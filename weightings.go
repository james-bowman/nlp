package nlp

import (
	"math"

	"github.com/gonum/matrix/mat64"
)

type TfidfTransformer struct {
	transform *mat64.Dense
}

func NewTfidfTransformer() *TfidfTransformer {
	return &TfidfTransformer{}
}

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

func (t *TfidfTransformer) FitTransform(mat mat64.Matrix) (*mat64.Dense, error) {
	t.Fit(mat)
	return t.Transform(mat)
}
