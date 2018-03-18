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

type ldaMiniBatch struct {
	start, end int
	nPhiHat    []float64
	nZHat      []float64
	gamma      []float64
}

func newLdaMiniBatch(topics int, words int) *ldaMiniBatch {
	l := ldaMiniBatch{
		nPhiHat: make([]float64, topics*words),
		nZHat:   make([]float64, topics),
		gamma:   make([]float64, topics),
	}
	return &l
}

func (l *ldaMiniBatch) reset() {
	for i := range l.nPhiHat {
		l.nPhiHat[i] = 0
	}
	for i := range l.nZHat {
		l.nZHat[i] = 0
	}
	// assume gamma does not need to be zeroed between mini batches
}

// LatentDirichletAllocation (LDA) for fast unsupervised topic extraction.  LDA processes
// documents and learns their latent topic model estimating the posterior document over topic
// probability distribution (the probabilities of each document being allocated to each
// topic) and the posterior topic over word probability distribution.
//
// This transformer uses a parallel implemention of the
// SCVB0 (Stochastic Collapsed Variational Bayes) Algorithm (https://arxiv.org/pdf/1305.2452.pdf)
// by Jimmy Foulds with optional `clumping` optimisations.
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

	wordsInCorpus float64
	w, d          int

	// Rnd is the random number generator used to generate the initial distributions
	// for nTheta (the document over topic distribution), nPhi (the topic over word
	// distribution) and nZ (the topic assignments).
	Rnd *rand.Rand

	// mutexes for updating global topic statistics
	phiMutex sync.Mutex
	zMutex   sync.Mutex

	// Processes is the degree of parallelisation, or more specifically, the number of
	// concurrent go routines to use during fitting.
	Processes int

	// nPhi is the topics over words distribution
	nPhi []float64

	// nZ is the topic assignments
	nZ []float64
}

// NewLatentDirichletAllocation returns a new LatentDirichletAllocation type initialised
// with default values for k topics.
func NewLatentDirichletAllocation(k int) *LatentDirichletAllocation {
	// TODO:
	// - Add FitPartial (and FitPartialTransform?) methods
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
		Processes: runtime.GOMAXPROCS(0),
	}

	return &l
}

// init initialises model for fitting allocating memory for distributions and
// randomising initial values.
func (l *LatentDirichletAllocation) init(m mat.Matrix) {
	r, c := m.Dims()
	l.w, l.d = r, c
	l.nPhi = make([]float64, l.K*r)
	l.nZ = make([]float64, l.K)
	var v float64
	for i := 0; i < r; i++ {
		for k := 0; k < l.K; k++ {
			v = float64((l.Rnd.Int() % (r * l.K))) / float64(r*l.K)
			l.nPhi[i*l.K+k] = v
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
func (l *LatentDirichletAllocation) burnInDoc(j int, iterations int, m mat.Matrix, wc float64, gamma *[]float64, nTheta []float64) {
	var rhoTheta float64
	var sum, prevSum float64
	var thetaInd int

	for counter := 1; counter <= iterations; counter++ {
		if l.ChangeEvaluationFrequency > 0 && counter%l.ChangeEvaluationFrequency == 0 && 1 < iterations {
			// take a copy of current column j
			prevSum = 0
			for k := 0; k < l.K; k++ {
				prevSum += nTheta[j*l.K+k]
			}
		}
		rhoTheta = l.RhoTheta.Calc(l.rhoThetaT + float64(counter))
		ColNonZeroElemDo(m, j, func(i, j int, v float64) {
			var gammaSum float64
			for k := 0; k < l.K; k++ {
				// Eqn. 5.
				(*gamma)[k] = ((l.nPhi[i*l.K+k] + l.Eta) * (nTheta[j*l.K+k] + l.Alpha) / (l.nZ[k] + l.Eta*float64(l.w)))
				gammaSum += (*gamma)[k]
			}

			for k := 0; k < l.K; k++ {
				(*gamma)[k] /= gammaSum
			}

			for k := 0; k < l.K; k++ {
				// Eqn. 9.
				thetaInd = j*l.K + k
				nTheta[thetaInd] = ((math.Pow((1.0-rhoTheta), v) * nTheta[thetaInd]) +
					((1 - math.Pow((1.0-rhoTheta), v)) * wc * (*gamma)[k]))
			}
		})
		if l.ChangeEvaluationFrequency > 0 && counter%l.ChangeEvaluationFrequency == 0 && counter < iterations {
			sum = 0
			for k := 0; k < l.K; k++ {
				sum += nTheta[j*l.K+k]
			}
			if math.Abs(sum-prevSum)/float64(l.K) < l.MeanChangeTolerance {
				break
			}
		}
	}
}

// fitMiniBatch fits a proportion of the matrix as specified by miniBatch.  The
// algorithm is stochastic and so estimates across the minibatch and then applies those
// estimates to the global statistics.
func (l *LatentDirichletAllocation) fitMiniBatch(miniBatch *ldaMiniBatch, wc []float64, nTheta []float64, m mat.Matrix) {
	var rhoTheta float64
	batchSize := miniBatch.end - miniBatch.start
	var phiInd, thetaInd int

	for j := miniBatch.start; j < miniBatch.end; j++ {
		l.burnInDoc(j, l.BurnInPasses, m, wc[j], &miniBatch.gamma, nTheta)

		rhoTheta = l.RhoTheta.Calc(l.rhoThetaT + float64(l.BurnInPasses))
		ColNonZeroElemDo(m, j, func(i, j int, v float64) {
			var gammaSum float64
			for k := 0; k < l.K; k++ {
				// Eqn. 5.
				miniBatch.gamma[k] = ((l.nPhi[i*l.K+k] + l.Eta) * (nTheta[j*l.K+k] + l.Alpha) / (l.nZ[k] + l.Eta*float64(l.w)))
				gammaSum += miniBatch.gamma[k]
			}
			for k := 0; k < l.K; k++ {
				miniBatch.gamma[k] /= gammaSum
			}

			for k := 0; k < l.K; k++ {
				// Eqn. 9.
				thetaInd = j*l.K + k
				nTheta[thetaInd] = ((math.Pow((1.0-rhoTheta), v) * nTheta[thetaInd]) +
					((1 - math.Pow((1.0-rhoTheta), v)) * wc[j] * miniBatch.gamma[k]))

				// calculate sufficient stats
				nv := l.wordsInCorpus * miniBatch.gamma[k] / float64(batchSize)
				miniBatch.nPhiHat[i*l.K+k] += nv
				miniBatch.nZHat[k] += nv
			}
		})
	}
	rhoPhi := l.RhoPhi.Calc(l.rhoPhiT)
	l.rhoPhiT++

	// Eqn. 7.
	l.phiMutex.Lock()
	for w := 0; w < l.w; w++ {
		for k := 0; k < l.K; k++ {
			phiInd = w*l.K + k
			l.nPhi[phiInd] = ((1.0 - rhoPhi) * l.nPhi[phiInd]) + (rhoPhi * miniBatch.nPhiHat[phiInd])
		}
	}
	l.phiMutex.Unlock()

	// Eqn. 8.
	l.zMutex.Lock()
	for k := 0; k < l.K; k++ {
		l.nZ[k] = ((1.0 - rhoPhi) * l.nZ[k]) + (rhoPhi * miniBatch.nZHat[k])
	}
	l.zMutex.Unlock()
}

// normaliseTheta normalises theta to derive the posterior probability estimates for
// documents over topics.  All values for each document are divided by the sum of all
// values for the document.
func (l *LatentDirichletAllocation) normaliseTheta(theta []float64, result []float64) []float64 {
	//adjustment := l.Alpha
	adjustment := 0.0
	c := len(theta) / l.K
	if result == nil {
		result = make([]float64, l.K*c)
	}
	for j := 0; j < c; j++ {
		var sum float64
		for k := 0; k < l.K; k++ {
			sum += theta[j*l.K+k] + adjustment
		}
		for k := 0; k < l.K; k++ {
			result[j*l.K+k] = (theta[j*l.K+k] + adjustment) / sum
		}
	}
	return result
}

// normalisePhi normalises phi to derive the posterior probability estimates for
// topics over words.  All values for each topic are divided by the sum of all values
// for the topic.
func (l *LatentDirichletAllocation) normalisePhi(phi []float64, result []float64) []float64 {
	//adjustment := l.Eta
	adjustment := 0.0
	if result == nil {
		result = make([]float64, l.K*l.w)
	}
	sum := make([]float64, l.K)
	for i := 0; i < l.w; i++ {
		for k := 0; k < l.K; k++ {
			sum[k] += phi[i*l.K+k] + adjustment
		}
	}
	for i := 0; i < l.w; i++ {
		for k := 0; k < l.K; k++ {
			result[i*l.K+k] = (phi[i*l.K+k] + adjustment) / sum[k]
		}
	}
	return result
}

// Perplexity calculates the perplexity of the matrix m against the trained model.
// m is first transformed into corresponding posterior estimates for document over topic
// distributions and then used to calculate the perplexity.
func (l *LatentDirichletAllocation) Perplexity(m mat.Matrix) float64 {
	if t, isTypeConv := m.(sparse.TypeConverter); isTypeConv {
		m = t.ToCSC()
	}
	var wordCount float64
	r, c := m.Dims()

	if s, isSparse := m.(sparse.Sparser); isSparse {
		s.DoNonZero(func(i, j int, v float64) {
			wordCount += v
		})
	} else {
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				wordCount += m.At(i, j)
			}
		}
	}

	theta := l.unNormalisedTransform(m)
	return l.perplexity(m, wordCount, l.normaliseTheta(theta, theta), l.normalisePhi(l.nPhi, nil))
}

// perplexity returns the perplexity of the matrix against the model.
func (l *LatentDirichletAllocation) perplexity(m mat.Matrix, sum float64, nTheta []float64, nPhi []float64) float64 {
	_, c := m.Dims()
	var perplexity float64
	var ttlLogWordProb float64

	for j := 0; j < c; j++ {
		ColNonZeroElemDo(m, j, func(i, j int, v float64) {
			var dot float64
			for k := 0; k < l.K; k++ {
				dot += nPhi[i*l.K+k] * nTheta[j*l.K+k]
			}
			ttlLogWordProb += math.Log2(dot) * v
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
	return mat.DenseCopyOf(mat.NewDense(l.w, l.K, l.normalisePhi(l.nPhi, nil)).T())
}

// unNormalisedTransform performs an unNormalisedTransform - the output
// needs to be normalised using normaliseTheta before use.
func (l *LatentDirichletAllocation) unNormalisedTransform(m mat.Matrix) []float64 {
	_, c := m.Dims()
	theta := make([]float64, l.K*c)
	for i := range theta {
		//data[i] = rnd.Float64() + 0.5
		theta[i] = float64((l.Rnd.Int() % (c * l.K))) / float64(c*l.K)
	}
	gamma := make([]float64, l.K)

	for j := 0; j < c; j++ {
		var wc float64
		ColNonZeroElemDo(m, j, func(i, j int, v float64) {
			wc += v
		})
		l.burnInDoc(j, l.TransformationPasses, m, wc, &gamma, theta)
	}
	return theta
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
	_, c := m.Dims()
	theta := l.unNormalisedTransform(m)
	return mat.DenseCopyOf(mat.NewDense(c, l.K, l.normaliseTheta(theta, theta)).T()), nil
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

	nTheta := make([]float64, l.K*c)
	for i := 0; i < l.K*c; i++ {
		nTheta[i] = float64((l.Rnd.Int() % (c * l.K))) / float64(c*l.K)
	}
	wc := make([]float64, c)
	for j := 0; j < c; j++ {
		ColNonZeroElemDo(m, j, func(i, j int, v float64) {
			wc[j] += v
		})
		l.wordsInCorpus += wc[j]
	}

	var phiProb []float64
	var thetaProb []float64

	numMiniBatches := int(math.Ceil(float64(c) / float64(l.BatchSize)))
	processes := l.Processes
	if numMiniBatches < l.Processes {
		processes = numMiniBatches
	}
	miniBatches := make([]*ldaMiniBatch, processes)
	for i := range miniBatches {
		miniBatches[i] = newLdaMiniBatch(l.K, l.w)
	}

	l.rhoPhiT = 1
	var perplexity float64
	var prevPerplexity float64

	for it := 0; it < l.Iterations; it++ {
		l.rhoThetaT++

		mb := make(chan int)
		var wg sync.WaitGroup

		for process := 0; process < processes; process++ {
			wg.Add(1)
			go func(miniBatch *ldaMiniBatch) {
				defer wg.Done()
				for j := range mb {
					miniBatch.reset()
					miniBatch.start = j * l.BatchSize
					if j < numMiniBatches-1 {
						miniBatch.end = miniBatch.start + l.BatchSize
					} else {
						miniBatch.end = c
					}
					l.fitMiniBatch(miniBatch, wc, nTheta, m)
				}
			}(miniBatches[process])
		}

		for j := 0; j < numMiniBatches; j++ {
			mb <- j
		}
		close(mb)
		wg.Wait()

		if l.PerplexityEvaluationFrequency > 0 && (it+1)%l.PerplexityEvaluationFrequency == 0 {
			phiProb = l.normalisePhi(l.nPhi, phiProb)
			thetaProb = l.normaliseTheta(nTheta, thetaProb)
			perplexity = l.perplexity(m, l.wordsInCorpus, thetaProb, phiProb)

			if prevPerplexity != 0 && math.Abs(prevPerplexity-perplexity) < l.PerplexityTolerance {
				break
			}
			prevPerplexity = perplexity
		}
	}
	return mat.DenseCopyOf(mat.NewDense(c, l.K, l.normaliseTheta(nTheta, thetaProb)).T()), nil
}
