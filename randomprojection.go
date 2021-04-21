package nlp

import (
	"math"
	"time"

	"golang.org/x/exp/rand"

	"github.com/james-bowman/sparse"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
	"gonum.org/v1/gonum/stat/sampleuv"
)

// SignRandomProjection represents a transform of a matrix into a lower
// dimensional space.  Sign Random Projection is a method of Locality
// Sensitive Hashing (LSH) sometimes referred to as the random hyperplane method.
// A set of random hyperplanes are created in the original dimensional
// space and then input matrices are expressed relative to the random
// hyperplanes as follows:
//	For each column vector in the input matrix, construct a corresponding output
// 	bit vector with each bit (i) calculated as follows:
//		if dot(vector, hyperplane[i]) > 0
// 			bit[i] = 1
// 		else
//			bit[i] = 0
// Whilst similar to other methods of random projection this method is unique in that
// it uses only a single bit in the output matrix to represent the sign of the result
// of the comparison (Dot product) with each hyperplane so encodes vector
// representations with very low memory and processor requirements whilst preserving
// relative distance between vectors from the original space.
// Hamming similarity (and distance) between the transformed vectors in the
// subspace can approximate Angular similarity (and distance) (which is strongly
// related to Cosine similarity) of the associated vectors from the original space.
type SignRandomProjection struct {
	// Bits represents the number of bits the output vectors should
	// be in length and hence the number of random hyperplanes needed
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
	rnd         *rand.Rand
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
	r.projections = CreateRandomProjectionTransform(r.K, rows, r.Density, r.rnd)
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
	if t, isTypeConv := m.(sparse.TypeConverter); isTypeConv {
		m = t.ToCSR()
	}

	product.Mul(r.projections, m)

	return &product, nil
}

// FitTransform is approximately equivalent to calling Fit() followed by Transform()
// on the same matrix.  This is a useful shortcut where separate training data is not being
// used to fit the model i.e. the model is fitted on the fly to the test data.
// The returned matrix is a sparse CSR format matrix of shape k x c.
func (r *RandomProjection) FitTransform(m mat.Matrix) (mat.Matrix, error) {
	return r.Fit(m).Transform(m)
}

// RRIBasis represents the initial basis for the index/elemental vectors
// used for Random Reflective Indexing
type RRIBasis int

const (
	// DocBasedRRI represents columns (documents/contexts in a term-document
	// matrix) forming the initial basis for index/elemental vectors in Random Indexing
	DocBasedRRI RRIBasis = iota

	// TermBasedRRI indicates rows (terms in a term-document matrix)
	// form the initial basis for index/elemental vectors in Reflective Random Indexing.
	TermBasedRRI
)

// RandomIndexing is a method of dimensionality reduction used for Latent Semantic
// Analysis in a similar way to TruncatedSVD and PCA.  Random
// Indexing is designed to solve limitations of very high dimensional
// vector space model implementations for modelling term co-occurance
// in language processing such as SVD typically used for LSA/LSI (Latent
// Semantic Analysis/Latent Semantic Indexing).  In implementation
// it bears some similarity to other random projection techniques
// such as those implemented in RandomProjection and SignRandomProjection
// within this package.
// The RandomIndexing type can also be used to perform Reflective
// Random Indexing which extends the Random Indexing model with additional
// training cycles to better support indirect inferrence i.e. find synonyms
// where the words do not appear together in documents.
type RandomIndexing struct {
	// K specifies the number of dimensions for the semantic space
	K int

	// Density specifies the proportion of non-zero elements in the
	// elemental vectors
	Density float64

	// Type specifies the initial basis for the elemental vectors
	// i.e. whether they initially represent the rows or columns
	// This is only relevent for Reflective Random Indexing
	Type RRIBasis

	// Reflections specifies the number of reflective training cycles
	// to run during fitting for RRI (Reflective Random Indexing). For
	// Randome Indexing (non-reflective) this is 0.
	Reflections int

	rnd *rand.Rand

	// components is a k x t matrix where `t` is the number of terms
	// (rows) in the training data matrix.  The columns in this matrix
	// contain the `context` vectors for RI where each column represents
	// a semantic representation of a term based upon the contexts
	// in which it has appeared within the training data.
	components mat.Matrix
}

// NewRandomIndexing returns a new RandomIndexing transformer
// configured to transform term document matrices into k dimensional
// space. The density parameter specifies the density of the index/elemental
// vectors used to project the input matrix into lower dimensional
// space i.e. the proportion of elements that are non-zero.
func NewRandomIndexing(k int, density float64) *RandomIndexing {
	return &RandomIndexing{
		K:       k,
		Density: density,
	}
}

// NewReflectiveRandomIndexing returns a new RandomIndexing type
// configured for Reflective Random Indexing.  Reflective Random
// Indexing applies additional (reflective) training cycles ontop
// of Random Indexing to capture indirect inferences (synonyms).
// i.e. similarity between terms that do not directly co-occur
// within the same context/document.
// basis specifies the basis for the reflective random indexing i.e.
// whether the initial, random index/elemental vectors should represent
// documents (columns) or terms (rows).
// reflections is the number of additional training cycles to apply
// to build the elemental vectors.
// Specifying basis == DocBasedRRI and reflections == 0 is equivalent
// to conventional Random Indexing.
func NewReflectiveRandomIndexing(k int, basis RRIBasis, reflections int, density float64) *RandomIndexing {
	return &RandomIndexing{
		K:           k,
		Type:        basis,
		Reflections: reflections,
		Density:     density,
	}
}

// PartialFit extends the model to take account of the specified matrix m. The
// context vectors are learnt and stored to be used for furture transformations
// and analysis.  PartialFit performs Random Indexing even if the Transformer is
// configured for Reflective Random Indexing so if RRI is required please train
// using the Fit() method as a batch operation.  Unlike the Fit() method, the
// PartialFit() method is designed to be called multiple times to support online
// and mini-batch learning whereas the Fit() method is only intended to be called
// once for batch learning.
func (r *RandomIndexing) PartialFit(m mat.Matrix) OnlineTransformer {
	rows, cols := m.Dims()

	if r.components == nil || r.components.(*sparse.CSR).IsZero() {
		r.components = sparse.NewCSR(r.K, rows, make([]int, r.K+1), []int{}, []float64{})
	}
	current := r.components

	// Create transform in transpose to get better randomised sparsity patterns
	// when partial fitting with small mini-batches e.g. single column/streaming
	idxVecs := CreateRandomProjectionTransform(cols, r.K, r.Density, r.rnd).T()
	ctxVecs := r.contextualise(m.T(), idxVecs)

	current.(*sparse.CSR).Add(current, ctxVecs)
	r.components = current

	return r
}

// Components returns a t x k matrix where `t` is the number of terms
// (rows) in the training data matrix.  The rows in this matrix
// are the `context` vectors for RI each one representing
// a semantic representation of a term based upon the contexts
// in which it has appeared within the training data.
func (r *RandomIndexing) Components() mat.Matrix {
	return r.components.T()
}

// SetComponents sets a t x k matrix where `t` is the number of terms
// (rows) in the training data matrix.
func (r *RandomIndexing) SetComponents(m mat.Matrix) {
	r.components = m
}

// Fit trains the model, creating random index/elemental vectors to
// be used to construct the new projected feature vectors ('context'
// vectors) in the reduced semantic dimensional space. If configured for
// Reflective Random Indexing then Fit may actually run multiple
// training cycles as specified during construction.  The Fit method
// trains the model in batch mode so is intended to be called once, for
// online/streaming or mini-batch training please consider the
// PartialFit method instead.
func (r *RandomIndexing) Fit(m mat.Matrix) Transformer {
	rows, cols := m.Dims()
	var idxVecs mat.Matrix

	if r.Type == TermBasedRRI {
		idxVecs = CreateRandomProjectionTransform(r.K, rows, r.Density, r.rnd)
	} else {
		idxVecs = CreateRandomProjectionTransform(r.K, cols, r.Density, r.rnd)
		idxVecs = r.contextualise(m.T(), idxVecs)
	}

	for i := 0; i < r.Reflections; i++ {
		idxVecs = r.contextualise(m, idxVecs)
		idxVecs = r.contextualise(m.T(), idxVecs)
	}

	r.components = idxVecs
	return r
}

// FitTransform is approximately equivalent to calling Fit() followed by Transform()
// on the same matrix.  This is a useful shortcut where separate training data is not being
// used to fit the model i.e. the model is fitted on the fly to the test data.
// The returned matrix is a sparse CSR format matrix of shape k x c.
func (r *RandomIndexing) FitTransform(m mat.Matrix) (mat.Matrix, error) {
	return r.Fit(m).Transform(m)
}

// Transform applies the transform, projecting matrix m into the
// lower dimensional semantic space.  The output matrix will be of
// shape k x c and will be a sparse CSR format matrix.  The transformation
// for each document vector is simply the accumulation of all trained context
// vectors relating to terms appearing in the document.  These are weighted by
// the frequency the term appears in the document.
func (r *RandomIndexing) Transform(m mat.Matrix) (mat.Matrix, error) {
	return r.contextualise(m, r.components), nil
}

// contextualise accumulates the vectors vectors for each column in matrix m weighting
// each row vector in vectors by its corresponding value in column of the matrix
func (r *RandomIndexing) contextualise(m mat.Matrix, vectors mat.Matrix) mat.Matrix {
	var product sparse.CSR

	product.Mul(vectors, m)

	return &product
}

// CreateRandomProjectionTransform returns a new random matrix for
// Random Projections of shape newDims x origDims.  The matrix will
// be randomly populated using probability distributions where density
// is used as the probability that each element will be populated.
// Populated values will be randomly selected from [-1, 1] scaled
// according to the density and dimensions of the matrix.  If rnd is
// nil then a new random number generator will be created and used.
func CreateRandomProjectionTransform(newDims, origDims int, density float64, rnd *rand.Rand) mat.Matrix {
	if rnd == nil {
		rnd = rand.New(rand.NewSource(uint64(time.Now().UnixNano())))
	}
	// TODO Possibly return a mat.Dense instead of sparse.CSR if
	// density == 1

	var ptr int
	var ind []int
	indptr := make([]int, newDims+1)

	for i := 0; i < newDims; i++ {
		nnz := binomial(origDims, density, rnd)
		if nnz > 0 {
			idx := make([]int, nnz)
			sampleuv.WithoutReplacement(idx, origDims, rnd)
			//sort.Ints(idx)
			ind = append(ind, idx...)
			ptr += nnz
		}
		indptr[i+1] = ptr
	}

	vals := make([]float64, len(ind))
	values(vals, newDims, density, rnd)

	return sparse.NewCSR(newDims, origDims, indptr, ind, vals)
}

func binomial(n int, p float64, rnd *rand.Rand) int {
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
