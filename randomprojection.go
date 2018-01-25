package nlp

import (
	"math"
	"sort"
	"time"

	"golang.org/x/exp/rand"

	"github.com/james-bowman/sparse"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
	"gonum.org/v1/gonum/stat/sampleuv"
)

// SignRandomProjection represents a transform of a matrix into a lower
// dimensional space.  Sign Random Projection is a method of Locality
// Sensitive Hashing sometimes referred to as the random hyperplane method.
// A set of random hyperplanes are projected into dimensional
// space and then input matrices are expressed relative to the random
// projections as follows:
//	For each column vector in the input matrix, construct a corresponding output
// 	bit vector with each bit (i) calculated as follows:
//		if dot(vector, projection[i]) > 0
// 			bit[i] = 1
// 		else
//			bit[i] = 0
// Similar to other methods of random projection this method is unique in that
// it uses a single bit in the output matrix to represent the sign of result
// of the comparison (Dot product) with each projection.
// Hamming similarity (and distance) between the transformed vectors in this
// new space can approximate Angular similarity (and distance) (which is strongly
// related to Cosine similarity) of the associated vectors from the original space
// with significant reductions in both memory usage and processing time.
type SignRandomProjection struct {
	// Bits represents the number of bits the output vectors should
	// be in length and hence the number of random projections needed
	// for the transformation
	Bits int

	// simhash is the simhash LSH (Locality Sensitive Hashing) algorithm
	// used to perform the sign random projection
	simHash *SimHash
}

// NewSignRandomProjection constructs a new SignRandomProjection transformer
// to reduce the dimensionality.  The transformer uses a number of random hyperplanes
// represented by `bits` and is the dimensionality of the output, transformed
// matrices.
func NewSignRandomProjection(bits int) *SignRandomProjection {
	return &SignRandomProjection{Bits: bits}
}

// Fit creates the random hyperplanes from the input training data matrix, mat and
// stores the hyperplanes as a transform to apply to matrices.
func (s *SignRandomProjection) Fit(m mat.Matrix) Transformer {
	rows, _ := m.Dims()
	s.simHash = NewSimHash(s.Bits, rows)
	return s
}

// Transform applies the transform decomposed from the training data matrix in Fit()
// to the input matrix.  The columns in the resulting output matrix will be a low
// dimensional binary representation of the columns within the original
// i.e. a hash or fingerprint that can be quickly and efficiently compared with other
// similar vectors.  Hamming similarity in the new dimensional space can be
// used to approximate Cosine similarity between the vectors of the original space.
// The returned matrix is a Binary matrix or BinaryVec type depending
// upon whether m is Matrix or Vector.
func (s *SignRandomProjection) Transform(m mat.Matrix) (mat.Matrix, error) {
	_, cols := m.Dims()

	sigs := make([]sparse.BinaryVec, cols)
	ColDo(m, func(j int, v mat.Vector) {
		sigs[j] = *s.simHash.Hash(v)
	})
	return sparse.NewBinary(s.Bits, cols, sigs), nil
}

// FitTransform is approximately equivalent to calling Fit() followed by Transform()
// on the same matrix.  This is a useful shortcut where separate training data is not being
// used to fit the model i.e. the model is fitted on the fly to the test data.
// The returned matrix is a Binary matrix or BinaryVec type depending upon
// whether m is Matrix or Vector.
func (s *SignRandomProjection) FitTransform(m mat.Matrix) (mat.Matrix, error) {
	return s.Fit(m).Transform(m)
}

// RandomProjection is a method of dimensionality reduction based upon
// the Johnson–Lindenstrauss lemma stating that a small set of points
// in a high-dimensional space can be embedded into a space of much
// lower dimension in such a way that distances between the points
// are nearly preserved.
//
// The technique projects the original
// matrix orthogonally onto a random subspace, transforming the
// elements of the original matrix into a lower dimensional representation.
// Computing orthogonal matrices is expensive and so this technique
// uses specially generated random matrices (hence the name) following
// the principle that in high dimensional spaces, there are lots of
// nearly orthogonal matrices.
type RandomProjection struct {
	K           int
	Density     float64
	projections mat.Matrix
}

// NewRandomProjection creates and returns a new RandomProjection
// transformer.  The RandomProjection will use a specially generated
// random matrix of the specified density and dimensionality k to
// perform the transform to k dimensional space.
func NewRandomProjection(k int, density float64) *RandomProjection {
	r := RandomProjection{
		K:       k,
		Density: density,
	}

	return &r
}

// Fit creates the random (almost) orthogonal matrix used to project
// input matrices into the new reduced dimensional subspace.
func (r *RandomProjection) Fit(m mat.Matrix) Transformer {
	rows, _ := m.Dims()
	r.projections = CreateRandomProjectionTransform(r.K, rows, r.Density)
	return r
}

// Transform applies the transformation, projecting the input matrix
// into the reduced dimensional subspace.  The transformed matrix
// will be a sparse CSR format matrix of shape k x c.
func (r *RandomProjection) Transform(m mat.Matrix) (mat.Matrix, error) {
	var product sparse.CSR

	// projections will be dimensions k x r (k x t)
	// m will be dimensions r x c (t x d)
	// product will be of reduced dimensions k x c (k x d)

	var matrix mat.Matrix
	if t, canConv := m.(sparse.TypeConverter); canConv {
		matrix = t.ToCSC()
	} else {
		matrix = m
	}

	product.Mul(r.projections, matrix)

	return &product, nil
}

// FitTransform is approximately equivalent to calling Fit() followed by Transform()
// on the same matrix.  This is a useful shortcut where separate training data is not being
// used to fit the model i.e. the model is fitted on the fly to the test data.
// The returned matrix is a sparse CSR format matrix of shape k x c.
func (r *RandomProjection) FitTransform(m mat.Matrix) (mat.Matrix, error) {
	return r.Fit(m).Transform(m)
}

// RIBasis represents the initial basis for the elemental vectors
// used for Random Indexing
type RIBasis int

const (
	// RowBasedRI indicates rows (terms in a term-document matrix)
	// forming the initial basis for elemental vectors in Random Indexing.
	// This is basis used for Random Indexing of documents, Reflective
	// Random Indexing can use either rows or columns as the initial
	// basis for elemental vectors.
	RowBasedRI RIBasis = iota

	// ColBasedRI represents columns (documents/contexts in a term-document
	// matrix) forming the initial basis for elemental vectors in Random Indexing
	ColBasedRI
)

// RandomIndexing is a method of dimensionality reduction similar to
// random projection and locality sensitive hashing.  Random
// Indexing is designed to solve limitations of very high dimensional
// vector space model implementations for modelling term co-occurance
// in language processing such as SVD as used by LSA/LSI (Latent
// Semantic Analysis/Latent Semantic Indexing).
// The RandomIndexing type can also be used to perform Reflective
// Random Indexing which extends the Random Indexing model with additional
// training cycles to support indirect inferrences i.e. find synonyms
// where the words do not appear together in documents.
type RandomIndexing struct {
	// K specifies the number of dimensions for the semantic space
	K int

	// Density specifies the proportion of non-zero elements in the
	// elemental vectors
	Density float64

	// Type specifies the initial basis for the elemental vectors
	// i.e. whether they initially represent the rows or columns
	// For Random Indexing this should be RowBasedRI, for RRI
	// (Reflective Random Indexing) it can be either RowBasedRI or
	// ColBasedRI
	Type RIBasis

	// Reflections specifies the number of reflective training cycles
	// to run during fitting for RRI (Reflective Random Indexing).
	// If Type is ColBasedRI then Reflections must be >= 1
	Reflections   int
	elementalVecs mat.Matrix
}

// NewRandomIndexing returns a new RandomIndexing transformer
// configured to transform term document matrices into k dimensional
// space. The density parameter specifies the density of the elemental
// vectors used to project the input matrix into lower dimensional
// space i.e. the proportion of elements that are non-zero.
// As RandomIndexing makes use of sparse matrix formats, specifying
// lower values for density will result in lower memory usage.
func NewRandomIndexing(k int, density float64) *RandomIndexing {
	r := RandomIndexing{
		K:       k,
		Density: density,
	}

	return &r
}

// NewReflectiveRandomIndexing returns a new RandomIndexing type
// configured for Reflective Random Indexing.  Reflective Random
// Indexing applies additional (reflective) training cycles ontop
// of Random Indexing to capture indirect inferences (synonyms).
// i.e. similarity between terms that do not directly co-occur
// within the same context/document.
// t specifies the basis for the reflective random indexing i.e.
// whether the initial, random elemental vectors should represent
// columns or rows.
// reflections is the number of training cycles to apply.
// If t == RowBasedRI and reflections == 0 then the created type
// will perform conventional Random Indexing.
// NewReflectiveRandomIndexing will panic if t == ColBasedRI and
// reflections < 1 because column based Reflective Random Indexing
// requires at least one reflective training cycle to generate the
// row based elemental vectors required for RI/RRI.
func NewReflectiveRandomIndexing(k int, t RIBasis, reflections int, density float64) *RandomIndexing {
	if t == ColBasedRI && reflections < 1 {
		panic("nlp: At least 1 reflection required for Column Based Reflective Random Indexing")
	}

	r := RandomIndexing{
		K:           k,
		Density:     density,
		Type:        t,
		Reflections: reflections,
	}

	return &r
}

// Fit trains the model, creating random elemental vectors that will
// later be used to construct the new projected feature vectors in
// the reduced semantic dimensional space. If configured for
// Reflective Random Indexing then Fit may actually run multiple
// training cycles as specified during construction.
func (r *RandomIndexing) Fit(m mat.Matrix) Transformer {
	rows, cols := m.Dims()
	var p mat.Matrix
	var cyclesPerformed int

	if r.Type == ColBasedRI {
		p = CreateRandomProjectionTransform(r.K, cols, r.Density)
		r.elementalVecs = p.(sparse.TypeConverter).ToCSC()

		r.trainingCycle(m.T())
		cyclesPerformed = 1
	} else {
		p = CreateRandomProjectionTransform(r.K, rows, r.Density)
		r.elementalVecs = p.(sparse.TypeConverter).ToCSC()
	}

	for i := cyclesPerformed; i < r.Reflections; i++ {
		r.trainingCycle(m)
		r.trainingCycle(m.T())
	}

	return r
}

// trainingCycle carries out a single training cycle
func (r *RandomIndexing) trainingCycle(m mat.Matrix) {
	out, err := r.Transform(m)
	if err != nil {
		panic("nlp: Failed to fit reflective random indexing during training cycle")
	}
	r.elementalVecs = out
}

// Transform applies the transform, projecting matrix into the
// lower dimensional semantic space.  The output matrix will be of
// shape k x c and will be a sparse CSC format matrix.
func (r *RandomIndexing) Transform(matrix mat.Matrix) (mat.Matrix, error) {
	rows, cols := matrix.Dims()
	k, _ := r.elementalVecs.Dims()

	var ptr int
	var ind []int
	var data []float64
	indptr := make([]int, cols+1)

	var m mat.Matrix
	if t, isTypeConv := matrix.(sparse.TypeConverter); isTypeConv {
		m = t.ToCSC()
	} else {
		m = matrix
	}

	colDoer, isColDoer := m.(mat.ColNonZeroDoer)

	for j := 0; j < cols; j++ {
		featVec := sparse.NewVecCOO(k, []int{0}, []float64{0})
		if isColDoer {
			colDoer.DoColNonZero(j, func(i, j int, v float64) {
				idxVec := r.elementalVecs.(mat.ColViewer).ColView(i)
				featVec.AddScaledVec(featVec, v, idxVec)
			})
		} else {
			for i := 0; i < rows; i++ {
				v := m.At(i, j)
				if v != 0 {
					idxVec := r.elementalVecs.(mat.ColViewer).ColView(i)
					featVec.AddScaledVec(featVec, v, idxVec)
				}
			}
		}
		featVec.DoNonZero(func(i, j int, v float64) {
			data = append(data, v)
			ind = append(ind, i)
			ptr++
		})
		indptr[j+1] = ptr
	}

	product := sparse.NewCSC(r.K, cols, indptr, ind, data)

	return product, nil
}

// FitTransform is approximately equivalent to calling Fit() followed by Transform()
// on the same matrix.  This is a useful shortcut where separate training data is not being
// used to fit the model i.e. the model is fitted on the fly to the test data.
// The returned matrix is a sparse CSC format matrix of shape k x c.
func (r *RandomIndexing) FitTransform(m mat.Matrix) (mat.Matrix, error) {
	return r.Fit(m).Transform(m)
}

// CreateRandomProjectionTransform returns a new random matrix for
// Random Projections of shape newDims x origDims.  The matrix will
// be randomly populated using probability distributions where density
// is used as the probability that each element will be populated.
// Populated values will be randomly selected from [-1, 1] scaled
// according to the density and dimensions of the matrix.
func CreateRandomProjectionTransform(newDims, origDims int, density float64) mat.Matrix {
	rnd := rand.New(rand.NewSource(uint64(time.Now().UnixNano())))

	// TODO Possibly return a mat.Dense instead of sparse.CSR if
	// density == 1

	var ptr int
	var ind []int
	indptr := make([]int, newDims+1)

	for i := 0; i < newDims; i++ {
		nnz := binomial(origDims, density, rnd)
		idx := make([]int, nnz)
		sampleuv.WithoutReplacement(idx, origDims, rnd)
		sort.Ints(idx)
		ind = append(ind, idx...)
		ptr += nnz
		indptr[i+1] = ptr
	}

	vals := make([]float64, len(ind))
	values(vals, newDims, density, rnd)

	return sparse.NewCSR(newDims, origDims, indptr, ind, vals)
}

func binomial(n int, p float64, rnd *rand.Rand) int {
	if rnd == nil {
		rnd = rand.New(rand.NewSource(uint64(time.Now().UnixNano())))
	}
	dist := distuv.Bernoulli{
		P: p,
		// Should this be Source (Gonum code and docs seem out of sync)
		Src: rnd,
	}

	var x int
	for i := 0; i < n; i++ {
		x += int(dist.Rand())
	}
	return x
}

func values(idx []float64, dims int, density float64, rnd *rand.Rand) {
	if rnd == nil {
		rnd = rand.New(rand.NewSource(uint64(time.Now().UnixNano())))
	}
	dist := distuv.Bernoulli{
		P: 0.5,
		// Should this be Source (Gonum code and docs seem out of sync)
		Src: rnd,
	}

	factor := math.Sqrt(1.0/density) / math.Sqrt(float64(dims))
	for i := range idx {
		idx[i] = factor * (dist.Rand()*2 - 1)
	}
}
