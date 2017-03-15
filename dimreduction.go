package nlp

import (
	"fmt"

	"github.com/gonum/matrix"
	"github.com/gonum/matrix/mat64"
)

type TruncatedSVD struct {
	transform *mat64.Dense
	K         int
}

func NewTruncatedSVD(k int) *TruncatedSVD {
	return &TruncatedSVD{K: k}
}

func (t *TruncatedSVD) Fit(mat *mat64.Dense) *TruncatedSVD {
	t.FitTransform(mat)
	return t
}

func (t *TruncatedSVD) Transform(mat *mat64.Dense) (*mat64.Dense, error) {
	var product mat64.Dense

	product.Product(t.transform.T(), mat)

	return &product, nil
}

func (t *TruncatedSVD) FitTransform(mat *mat64.Dense) (*mat64.Dense, error) {
	var svd mat64.SVD
	if ok := svd.Factorize(mat, matrix.SVDThin); !ok {
		return nil, fmt.Errorf("Failed SVD Factorisation of working matrix")
	}
	s, u, v := t.extractSVD(&svd)

	m, n := mat.Dims()
	min := minimum(t.K, m, n)

	// truncate matrix to k << min(m, n)
	uk, ok := u.Slice(0, m, 0, min).(*mat64.Dense)
	if !ok {
		return nil, fmt.Errorf("Failed to truncate U")
	}

	vk, ok := v.Slice(0, n, 0, min).(*mat64.Dense)
	if !ok {
		return nil, fmt.Errorf("Failed to truncate V")
	}

	// only build out eigenvalue matrix to k x k (truncate values) (or min(m, n) if lower)
	sigmak := mat64.NewDense(min, min, nil)
	for i := 0; i < min; i++ {
		sigmak.Set(i, i, s[i])
	}

	t.transform = uk

	var product mat64.Dense
	product.Product(sigmak, vk.T())

	return &product, nil
}

func minimum(k, m, n int) int {
	return min(k, min(m, n))
}

func min(m, n int) int {
	if m < n {
		return m
	}
	return n
}

func (t *TruncatedSVD) extractSVD(svd *mat64.SVD) (s []float64, u, v *mat64.Dense) {
	var um, vm mat64.Dense
	um.UFromSVD(svd)
	vm.VFromSVD(svd)
	s = svd.Values(nil)
	return s, &um, &vm
}
