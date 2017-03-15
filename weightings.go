package nlp

import (
	"math"

	"github.com/gonum/matrix/mat64"
)

// TfidfTransformer takes a raw term document matrix and weights each raw term frequency
// value depending upon how commonly it occurs across all documents within the corpus.
// For example a very commonly occuring word like `the` is likely to occur in all documents
// and so would be weighted down.
// More precisely, TfidfTransformer applies a tf-idf algorithm to the matrix where each
// term frequency is multiplied by the inverse document frequency.  Inverse document 
// frequency is calculated as log(n/df) where df is the number of documents in which the // 
// term occurs and n is the total number of documents within the corpus.  We add 1 to both n 
// and df before division to prevent division by zero.
type TfidfTransformer struct {
	transform *mat64.Dense
}

// NewTfidfTransformer constructs a new TfidfTransformer.
func NewTfidfTransformer() *TfidfTransformer {
	return &TfidfTransformer{}
}

// Fit takes a training term document matrix, counts term occurances across all documents
// and constructs an inverse document frequency transform to apply to matrices in subsequent 
// calls to Transform().
func (t *TfidfTransformer) Fit(mat mat64.Matrix) *TfidfTransformer {
	m, n := mat.Dims()

	// build a diagonal matrix from array of term weighting values for subsequent
	// multiplication with term document matrics
	t.transform = mat64.NewDense(m, m, nil)

	for i := 0; i < m; i++ {
		df := 0
		for j := 0; j < n; j++ {
			if mat.At(i, j) != 0 {
				df++
			}
		}
		idf := math.Log(float64(1+n) / float64(1+df))
		t.transform.Set(i, i, idf)
	}

	return t
}

func (t *TfidfTransformer) Transform(mat mat64.Matrix) (*mat64.Dense, error) {
	m, n := mat.Dims()
	product := mat64.NewDense(m, n, nil)

	product.Product(t.transform, mat)

	// todo: possibly L2 norm matrix

	return product, nil
}

// FitTransform is exactly equivalent to calling Fit() followed by Transform() on the 
// same matrix.  This is a convenience where separate trianing data is not being 
// used to fit the model i.e. the model is fitted on the fly to the test data.
func (t *TfidfTransformer) FitTransform(mat mat64.Matrix) (*mat64.Dense, error) {
	t.Fit(mat)
	return t.Transform(mat)
}
