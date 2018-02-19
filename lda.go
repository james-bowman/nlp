package nlp

import (
	"math"
	"runtime"
	"sync"
	"time"

	"github.com/james-bowman/sparse"
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
)

// LearningSchedule is used to calculate the learning rate for each iteration using a natural
// gradient descent algorithm.
type LearningSchedule struct {
	// S is the scale of the step size for the learning rate.
	S float64

	// Tau is the learning offset. The learning offset downweights the
	// learning rate from early iterations.
	Tau float64

	// Kappa controls the learning decay.  This is the amount the learning rate
	// reduces each iteration.  This is typically a value between 0.5 and 1.0.
	Kappa float64
}

// Calc returns the learning rate for the specified iteration
func (l LearningSchedule) Calc(iteration float64) float64 {
	return l.S / math.Pow(l.Tau+iteration, l.Kappa)
}

type ldaWorkspace struct {
	K          int
	topicPool  sync.Pool
	matrixPool sync.Pool
}

func newLdaWorkspace(k int) *ldaWorkspace {
	l := ldaWorkspace{
		K: k,
		topicPool: sync.Pool{
			New: func() interface{} {
				return make([]float64, k)
			},
		},
		matrixPool: sync.Pool{
			New: func() interface{} {
				return make([]float64, k*100)
			},
		},
	}

	return &l
}

func (l *ldaWorkspace) leaseFloatsForTopics(clear bool) []float64 {
	w := l.topicPool.Get().([]float64)
	if clear {
		for i := range w {
			w[i] = 0
		}
	}
	return w
}

func (l *ldaWorkspace) returnFloatsForTopics(w []float64) {
	l.topicPool.Put(w)
}

func (l *ldaWorkspace) leaseFloatsForMatrix(dims int, clear bool) []float64 {
	size := dims * l.K
	w := l.matrixPool.Get().([]float64)
	if size <= cap(w) {
		w = w[:size]
		if clear {
			for i := range w {
				w[i] = 0
			}
		}
		return w
	}
	w = make([]float64, size)
	return w
}

func (l *ldaWorkspace) returnFloatsForMatrix(w []float64) {
	l.matrixPool.Put(w)
}

// LatentDirichletAllocation (LDA) for fast unsupervised topic extraction.  Parallel implemention
// of the
// [SCVB0 (Stochastic Collapsed Variational Bayes) Algorithm](https://arxiv.org/pdf/1305.2452.pdf)
// by Jimmy Foulds with optional `clumping` optimisations.  Outputs matrices with a column for
// each document represented by columns in the input matrices.  Each document in the output
// is represented as the probability distribution over topics.
type LatentDirichletAllocation struct {
	// Iterations is the maximum number of training iterations
	Iterations int

	// PerplexityTolerance is the tolerance of perplexity below which the Fit method will stop iterating
	// and complete.  If the evaluated perplexity is is below the tolerance, fitting will terminate successfully
	// without necessarily completing all of the configured number of training iterations.
	PerplexityTolerance float64

	// PerplexityEvaluationFrquency is the frequency with which to test Perplexity against PerplexityTolerance inside
	// Fit.  A value <= 0 will not evaluate Perplexity at all and simply iterate for `Iterations` iterations.
	PerplexityEvaluationFrequency int

	// BatchSize is the size of mini batches used during training
	BatchSize int

	// K is the number of topics
	K int

	w, d int

	// NumBurnInPasses is the number of `burn-in` passes across the documents in the
	// training data to learn the document statistics before we start collecting topic statistics.
	BurnInPasses int

	// TransformationPasses is the number of passes to transform new documents given a previously
	// fitted topic model
	TransformationPasses int

	// MeanChangeTolerance is the tolerance of change to Theta between burn in passes.
	// If the level of change between passes is below the tolerance, the burn in will complete
	// without necessarily completing the configured number of passes.
	MeanChangeTolerance float64

	// ChangeEvaluationFrequency is the frequency with which to test Perplexity against
	// MeanChangeTolerance during burn-in and transformation.  A value <= 0 will not evaluate
	// the mean change at all and simply iterate for `BurnInPasses` iterations.
	ChangeEvaluationFrequency int

	// Alpha is the prior of theta (the documents over topics distribution)
	Alpha float64
	// Eta is the prior of phi (the topics over words distribution)
	Eta float64
	// RhoPhi is the learning rate for phi (the topics over words distribution)
	RhoPhi LearningSchedule
	// RhoTheta is the learning rate for theta (the documents over topics distribution)
	RhoTheta LearningSchedule

	rhoPhiT   float64
	rhoThetaT float64

	sum float64

	// Rnd is the random number generator used to generate the initial distributions
	// for nTheta (the document over topic distribution), nPhi (the topic over word
	// distribution) and nZ (the topic assignments).
	Rnd *rand.Rand

	// workspace holds sync.Pool instances for reusing allocated memory for temporary
	// working storage between iterations and go routines.
	workspace *ldaWorkspace

	// mutexes for updating global topic statistics
	phiMutex sync.RWMutex
	zMutex   sync.RWMutex

	// Processes is the degree of parallelisation, or more specifically, the number of
	// concurrent go routines to use during fitting.
	Processes int

	// nPhi is the topics over words distribution
	nPhi *mat.Dense

	// nZ is the topic assignments
	nZ []float64
}

// NewLatentDirichletAllocation returns a new LatentDirichletAllocation type initialised
// with default values for k topics.
func NewLatentDirichletAllocation(k int) *LatentDirichletAllocation {
	// TODO:
	// - Add FitPartial (and FitPartialTransform?) methods
	// - Add unit tests, documentation and benchmarks
	// - consider building nTheta as a slice rather than matrix then constructing the matrix
	// to avoid At() and Set() method calls
	// once finished transforming
	// - refactor word counting
	// - rename and check rhoTheta_t and rhoPhi_t
	// - Check visibilitiy of member variables
	// - Try parallelising:
	// 		- minibatches
	// 		- individual docs within minibatches
	// 		- M step
	//		- other areas
	// - investigate whetehr can combine/consolidate fitMiniBatch and burnIn
	// - Check whether nPhi could be sparse
	// - Add persistence methods

	l := LatentDirichletAllocation{
		Iterations:                    1000,
		PerplexityTolerance:           1e-2,
		PerplexityEvaluationFrequency: 30,
		BatchSize:                     100,
		K:                             k,
		BurnInPasses:                  1,
		TransformationPasses:          500,
		MeanChangeTolerance:           1e-5,
		ChangeEvaluationFrequency:     30,
		Alpha: 0.1,
		Eta:   0.01,
		RhoPhi: LearningSchedule{
			S:     10,
			Tau:   1000,
			Kappa: 0.9,
		},
		RhoTheta: LearningSchedule{
			S:     1,
			Tau:   10,
			Kappa: 0.9,
		},
		rhoPhiT:   1,
		rhoThetaT: 1,
		Rnd:       rand.New(rand.NewSource(uint64(time.Now().UnixNano()))),
		workspace: newLdaWorkspace(k),
		Processes: runtime.GOMAXPROCS(0),
	}

	return &l
}

// init initialises model for fitting allocating memory for distributions and
// randomising initial values.
func (l *LatentDirichletAllocation) init(m mat.Matrix) {
	r, c := m.Dims()
	l.w, l.d = r, c

	l.nPhi = mat.NewDense(l.K, r, nil)
	l.nZ = make([]float64, l.K)

	var v float64

	for i := 0; i < r; i++ {
		for k := 0; k < l.K; k++ {
			v = float64((l.Rnd.Int() % (r * l.K))) / float64(r*l.K)
			l.nPhi.Set(k, i, v)
			l.nZ[k] += v
		}
	}
}

// Fit fits the model to the specified matrix m.  The latent topics, and probability
// distribution of topics over words, are learnt and stored to be used for furture transformations
// and analysis.
func (l *LatentDirichletAllocation) Fit(m mat.Matrix) Transformer {
	l.FitTransform(m)
	return l
}

// burnInDoc calculates document statistics as part of fitting and transforming new
// documents
func (l *LatentDirichletAllocation) burnInDoc(j int, iterations int, m mat.Matrix, wc float64, nTheta *mat.Dense) {
	var rhoTheta float64
	//gamma := make([]float64, l.K)
	gamma := l.workspace.leaseFloatsForTopics(false)
	current := l.workspace.leaseFloatsForTopics(false)
	prev := l.workspace.leaseFloatsForTopics(false)
	defer func() {
		l.workspace.returnFloatsForTopics(prev)
		l.workspace.returnFloatsForTopics(current)
		l.workspace.returnFloatsForTopics(gamma)
	}()
	for counter := 1; counter <= iterations; counter++ {
		if l.ChangeEvaluationFrequency > 0 &&
			counter%l.ChangeEvaluationFrequency == 0 &&
			1 < iterations {
			prev = mat.Col(prev, j, nTheta)
		}
		rhoTheta = l.RhoTheta.Calc(l.rhoThetaT + float64(counter))
		ColNonZeroElemDo(m, j, func(i, j int, v float64) {
			var gammaSum float64
			for k := 0; k < l.K; k++ {
				// Eqn. 5.
				gamma[k] = ((l.nPhi.At(k, i) + l.Eta) * (nTheta.At(k, j) + l.Alpha) / (l.nZ[k] + l.Eta*float64(l.w)))
				gammaSum += gamma[k]
			}

			for k := 0; k < l.K; k++ {
				gamma[k] /= gammaSum
			}

			for k := 0; k < l.K; k++ {
				// Eqn. 9.
				nt := ((math.Pow((1.0-rhoTheta), v) * nTheta.At(k, j)) +
					((1 - math.Pow((1.0-rhoTheta), v)) * wc * gamma[k]))
				nTheta.Set(k, j, nt)
			}
		})
		if l.ChangeEvaluationFrequency > 0 &&
			counter%l.ChangeEvaluationFrequency == 0 &&
			counter < iterations {
			current = mat.Col(current, j, nTheta)
			var sum float64
			for i, v := range prev {
				sum += math.Abs(v - current[i])
			}
			if sum/float64(l.K) < l.MeanChangeTolerance {
				break
			}
		}
	}
}

// fitMiniBatch fits a proportion of the matrix using columns cStart to cEnd.  The
// algorithm is stochastic and so estimates across the batch and then applies those
// estimates to the global statistics.
func (l *LatentDirichletAllocation) fitMiniBatch(cStart, cEnd int, wc []float64, nTheta *mat.Dense, m mat.Matrix) {
	nPhiHatData := l.workspace.leaseFloatsForMatrix(l.w, true)
	nPhiHat := mat.NewDense(l.K, l.w, nPhiHatData)
	nZHat := l.workspace.leaseFloatsForTopics(true)
	gamma := l.workspace.leaseFloatsForTopics(false)
	defer func() {
		l.workspace.returnFloatsForTopics(gamma)
		l.workspace.returnFloatsForTopics(nZHat)
		l.workspace.returnFloatsForMatrix(nPhiHatData)
	}()

	var rhoTheta float64
	batchSize := cEnd - cStart

	for j := cStart; j < cEnd; j++ {
		l.burnInDoc(j, l.BurnInPasses, m, wc[j], nTheta)

		rhoTheta = l.RhoTheta.Calc(l.rhoThetaT + float64(l.BurnInPasses))
		ColNonZeroElemDo(m, j, func(i, j int, v float64) {
			var gammaSum float64
			for k := 0; k < l.K; k++ {
				// Eqn. 5.
				gamma[k] = ((l.nPhi.At(k, i) + l.Eta) * (nTheta.At(k, j) + l.Alpha) / (l.nZ[k] + l.Eta*float64(l.w)))
				gammaSum += gamma[k]
			}
			for k := 0; k < l.K; k++ {
				gamma[k] /= gammaSum
			}

			for k := 0; k < l.K; k++ {
				// Eqn. 9.
				nt := ((math.Pow((1.0-rhoTheta), v) * nTheta.At(k, j)) +
					((1 - math.Pow((1.0-rhoTheta), v)) * wc[j] * gamma[k]))
				nTheta.Set(k, j, nt)

				// calculate sufficient stats
				nv := l.sum * gamma[k] / float64(batchSize)
				nPhiHat.Set(k, i, nPhiHat.At(k, i)+nv)
				nZHat[k] += nv
			}
		})
	}
	rhoPhi := l.RhoPhi.Calc(l.rhoPhiT)
	l.rhoPhiT++
	for k := 0; k < l.K; k++ {
		l.phiMutex.Lock()
		// Eqn. 7.
		for w := 0; w < l.w; w++ {
			l.nPhi.Set(k, w, ((1.0-rhoPhi)*l.nPhi.At(k, w))+(rhoPhi*nPhiHat.At(k, w)))
		}
		l.phiMutex.Unlock()

		l.zMutex.Lock()
		// Eqn. 8.
		l.nZ[k] = ((1.0 - rhoPhi) * l.nZ[k]) + (rhoPhi * nZHat[k])
		l.zMutex.Unlock()
	}
}

// normaliseTheta normalises the document over topic distribution.  All values for each document
// are divided by the sum of all values for the document.
func (l *LatentDirichletAllocation) normaliseTheta(theta mat.Matrix, thetaProb *mat.Dense) *mat.Dense {
	//adjustment := l.Alpha
	adjustment := 0.0
	r, c := theta.Dims()
	if thetaProb == nil {
		thetaProb = mat.NewDense(r, c, nil)
	}
	for j := 0; j < c; j++ {
		var sum float64
		for k := 0; k < l.K; k++ {
			sum += theta.At(k, j) + adjustment
		}
		for k := 0; k < l.K; k++ {
			thetaProb.Set(k, j, (theta.At(k, j)+adjustment)/sum)
		}
	}
	return thetaProb
}

// normalisePhi normalises the topic over word distribution.  All values for each topic
// are divided by the sum of all values for the topic.
func (l *LatentDirichletAllocation) normalisePhi(phi mat.Matrix, phiProb *mat.Dense) *mat.Dense {
	//adjustment := l.Eta
	adjustment := 0.0
	r, c := phi.Dims()
	if phiProb == nil {
		phiProb = mat.NewDense(r, c, nil)
	}
	for k := 0; k < l.K; k++ {
		var sum float64
		for i := 0; i < c; i++ {
			sum += phi.At(k, i) + adjustment
		}
		for i := 0; i < c; i++ {
			phiProb.Set(k, i, (phi.At(k, i)+adjustment)/sum)
		}
	}
	return phiProb
}

// Perplexity calculates the perplexity of the matrix m against the trained model.
// m is first transformed into corresponding document over topic distributions and
// then used to calculate the perplexity.
func (l *LatentDirichletAllocation) Perplexity(m mat.Matrix) float64 {
	if t, isTypeConv := m.(sparse.TypeConverter); isTypeConv {
		m = t.ToCSC()
	}
	var sum float64

	if s, isSparse := m.(sparse.Sparser); isSparse {
		s.DoNonZero(func(i, j int, v float64) {
			sum += v
		})
	} else {
		r, c := m.Dims()
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				sum += m.At(i, j)
			}
		}
	}

	theta := l.unNormalisedTransform(m)
	return l.perplexity(m, sum, l.normaliseTheta(theta, theta), l.normalisePhi(l.nPhi, nil))
}

// perplexity returns the perplexity of the matrix against the model.
func (l *LatentDirichletAllocation) perplexity(m mat.Matrix, sum float64, nTheta *mat.Dense, nPhi *mat.Dense) float64 {
	_, c := m.Dims()
	var perplexity float64
	var ttlLogWordProb float64

	phiData := l.workspace.leaseFloatsForTopics(false)
	thetaData := l.workspace.leaseFloatsForTopics(false)

	phiCol := mat.NewVecDense(l.K, phiData)
	thetaCol := mat.NewVecDense(l.K, thetaData)

	defer func() {
		l.workspace.returnFloatsForTopics(thetaData)
		l.workspace.returnFloatsForTopics(phiData)
	}()

	for j := 0; j < c; j++ {
		ColNonZeroElemDo(m, j, func(i, j int, v float64) {
			phiCol.ColViewOf(nPhi, i)
			thetaCol.ColViewOf(nTheta, j)
			ttlLogWordProb += math.Log2(mat.Dot(phiCol, thetaCol)) * v
		})
	}
	perplexity = math.Exp2(-ttlLogWordProb / sum)
	return perplexity
}

// Components returns the topic over words probability distribution.  The returned
// matrix is of dimensions K x W where w was the number of rows in the training matrix
// and each column represents a unique words in the vocabulary and K is the number of
// topics.
func (l *LatentDirichletAllocation) Components() mat.Matrix {
	return l.normalisePhi(l.nPhi, nil)
}

// unNormalisedTransform performs an unNormalisedTransform - the output
// needs to be normalised using normaliseTheta before use.
func (l *LatentDirichletAllocation) unNormalisedTransform(m mat.Matrix) *mat.Dense {
	_, c := m.Dims()
	datalen := l.K * c
	data := make([]float64, datalen)
	for i := 0; i < datalen; i++ {
		//data[i] = rnd.Float64() + 0.5
		data[i] = float64((l.Rnd.Int() % (c * l.K))) / float64(c*l.K)
	}
	prod := mat.NewDense(l.K, c, data)

	for j := 0; j < c; j++ {
		var wc float64
		ColNonZeroElemDo(m, j, func(i, j int, v float64) {
			wc += v
		})
		l.burnInDoc(j, l.TransformationPasses, m, wc, prod)
	}
	return prod
}

// Transform transforms the input matrix into a matrix representing the distribution
// of the documents over topics.
// THe returned matrix contains the document over topic distributions where each element
// is the probability of the corresponding document being related to the corresponding
// topic.  The returned matrix is a Dense matrix of shape K x C where K is the number
// of topics and C is the number of columns in the input matrix (representing the
// documents).
func (l *LatentDirichletAllocation) Transform(m mat.Matrix) (mat.Matrix, error) {
	if t, isTypeConv := m.(sparse.TypeConverter); isTypeConv {
		m = t.ToCSC()
	}

	prod := l.unNormalisedTransform(m)

	return l.normaliseTheta(prod, prod), nil
}

// FitTransform is approximately equivalent to calling Fit() followed by Transform()
// on the same matrix.  This is a useful shortcut where separate training data is not being
// used to fit the model i.e. the model is fitted on the fly to the test data.
// THe returned matrix contains the document over topic distributions where each element
// is the probability of the corresponding document being related to the corresponding
// topic.  The returned matrix is a Dense matrix of shape K x C where K is the number
// of topics and C is the number of columns in the input matrix (representing the
// documents).
func (l *LatentDirichletAllocation) FitTransform(m mat.Matrix) (mat.Matrix, error) {
	if t, isTypeConv := m.(sparse.TypeConverter); isTypeConv {
		m = t.ToCSC()
	}

	l.init(m)

	_, c := m.Dims()

	nTheta := mat.NewDense(l.K, c, nil)
	for j := 0; j < c; j++ {
		for k := 0; k < l.K; k++ {
			nTheta.Set(k, j, float64((l.Rnd.Int()%(c*l.K)))/float64(c*l.K))
		}
	}
	var phiProb *mat.Dense
	var thetaProb *mat.Dense

	numMiniBatches := int(math.Ceil(float64(c) / float64(l.BatchSize)))

	wc := make([]float64, c)

	for j := 0; j < c; j++ {
		ColNonZeroElemDo(m, j, func(i, j int, v float64) {
			wc[j] += v
		})
		l.sum += wc[j]
	}

	l.rhoPhiT = 1
	var perplexity float64
	var prevPerplexity float64

	for it := 0; it < l.Iterations; it++ {
		l.rhoThetaT++

		mb := make(chan int)
		var wg sync.WaitGroup

		for process := 0; process < l.Processes; process++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for j := range mb {
					var end int
					if j < numMiniBatches-1 {
						end = j*l.BatchSize + l.BatchSize
					} else {
						end = c
					}
					l.fitMiniBatch(j*l.BatchSize, end, wc, nTheta, m)
				}
			}()
		}

		for j := 0; j < numMiniBatches; j++ {
			mb <- j
		}
		close(mb)
		wg.Wait()

		if l.PerplexityEvaluationFrequency > 0 &&
			(it+1)%l.PerplexityEvaluationFrequency == 0 {
			phiProb = l.normalisePhi(l.nPhi, phiProb)
			thetaProb = l.normaliseTheta(nTheta, thetaProb)
			perplexity = l.perplexity(m, l.sum, thetaProb, phiProb)

			if prevPerplexity != 0 && math.Abs(prevPerplexity-perplexity) < l.PerplexityTolerance {
				break
			}
			prevPerplexity = perplexity
		}
	}

	return l.normaliseTheta(nTheta, thetaProb), nil
}
