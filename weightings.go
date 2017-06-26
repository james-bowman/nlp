package nlp

import (
	"math"

	"github.com/gonum/matrix/mat64"
	"github.com/james-bowman/sparse"
)

// Transformer provides a common interface for transformer steps.
type Transformer interface {
	Fit(mat64.Matrix) Transformer
	Transform(mat mat64.Matrix) (mat64.Matrix, error)
	FitTransform(mat mat64.Matrix) (mat64.Matrix, error)
}

// TfidfTransformer takes a raw term document matrix and weights each raw term frequency
// value depending upon how commonly it occurs across all documents within the corpus.
// For example a very commonly occuring word like `the` is likely to occur in all documents
// and so would be weighted down.
// More precisely, TfidfTransformer applies a tf-idf algorithm to the matrix where each
// term frequency is multiplied by the inverse document frequency.  Inverse document
// frequency is calculated as log(n/df) where df is the number of documents in which the
// term occurs and n is the total number of documents within the corpus.  We add 1 to both n
// and df before division to prevent division by zero.
type TfidfTransformer struct {
	transform mat64.Matrix
}

// NewTfidfTransformer constructs a new TfidfTransformer.
func NewTfidfTransformer() *TfidfTransformer {
	return &TfidfTransformer{}
}

// Fit takes a training term document matrix, counts term occurances across all documents
// and constructs an inverse document frequency transform to apply to matrices in subsequent
// calls to Transform().
func (t *TfidfTransformer) Fit(mat mat64.Matrix) Transformer {
	m, n := mat.Dims()

	weights := make([]float64, m)

	csr, ok := mat.(*sparse.CSR)

	for i := 0; i < m; i++ {
		df := 0
		if ok {
			df = csr.RowNNZ(i)
		} else {
			for j := 0; j < n; j++ {
				if mat.At(i, j) != 0 {
					df++
				}
			}
		}
		idf := math.Log(float64(1+n) / float64(1+df))
		weights[i] = idf
	}

	// build a diagonal matrix from array of term weighting values for subsequent
	// multiplication with term document matrics
	t.transform = sparse.NewDIA(m, weights)

	return t
}

// Transform applies the inverse document frequency (IDF) transform by multiplying
// each term frequency by its corresponding IDF value.  This has the effect of weighting
// each term frequency according to how often it appears across the whole document corpus
// so that naturally frequent occuring words are given less weight than uncommon ones.
// The returned matrix is a sparse matrix type.
func (t *TfidfTransformer) Transform(mat mat64.Matrix) (mat64.Matrix, error) {
	product := &sparse.CSR{}

	// simply multiply the matrix by our idf transform (the diagonal matrix of term weights)
	product.Mul(t.transform, mat)

	// todo: possibly L2 norm matrix to remove any bias caused by documents of different
	// lengths where longer documents naturally have more words and so higher word counts

	return product, nil
}

// FitTransform is exactly equivalent to calling Fit() followed by Transform() on the
// same matrix.  This is a convenience where separate trianing data is not being
// used to fit the model i.e. the model is fitted on the fly to the test data.
// The returned matrix is a sparse matrix type.
func (t *TfidfTransformer) FitTransform(mat mat64.Matrix) (mat64.Matrix, error) {
	return t.Fit(mat).Transform(mat)
}
