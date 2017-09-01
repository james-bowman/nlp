package nlp

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

// TruncatedSVD implements the Singular Value Decomposition factorisation of matrices.
// This produces an approximation of the input matrix at a lower rank.  This is a core
// component of LSA (Latent Semantic Analsis)
type TruncatedSVD struct {
	// Components is the truncated term matrix (matrix U of the Singular Value Decomposition
	// (A=USV^T)).  The matrix will be of size m, k where m = the number of unique terms
	// in the training data and k = the number of elements to truncate to (specified by
	// attribute K) or m or n (the number of documents in the training data) whichever of
	// the 3 values is smaller.
	Components mat.Matrix

	// K is the number of dimensions to which the output, transformed, matrix should be
	// truncated to.  The matrix output by the FitTransform() and Transform() methods will
	// be n rows by min(m, n, K) columns, where n is the number of columns in the original,
	// input matrix and min(m, n, K) is the lowest value of m, n, K where m is the number of
	// rows in the original, input matrix.
	K int
}

// NewTruncatedSVD creates a new TruncatedSVD transformer with K (the truncated
// dimensionality) being set to the specified value k
func NewTruncatedSVD(k int) *TruncatedSVD {
	return &TruncatedSVD{K: k}
}

// Fit performs the SVD factorisation on the input training data matrix, mat and
// stores the output term matrix as a transform to apply to matrices in the Transform matrix.
func (t *TruncatedSVD) Fit(mat mat.Matrix) Transformer {
	t.FitTransform(mat)
	return t
}

// Transform applies the transform decomposed from the training data matrix in Fit()
// to the input matrix.  The resulting output matrix will be the closest approximation
// to the input matrix at a reduced rank.  The returned matrix is a dense matrix type.
func (t *TruncatedSVD) Transform(m mat.Matrix) (mat.Matrix, error) {
	var product mat.Dense

	product.Mul(t.Components.T(), m)

	return &product, nil
}

// FitTransform is approximately equivalent to calling Fit() followed by Transform()
// on the same matrix.  This is a useful shortcut where separate trianing data is not being
// used to fit the model i.e. the model is fitted on the fly to the test data.
// The returned matrix is a dense matrix type.
func (t *TruncatedSVD) FitTransform(m mat.Matrix) (mat.Matrix, error) {
	var svd mat.SVD
	if ok := svd.Factorize(m, mat.SVDThin); !ok {
		return nil, fmt.Errorf("Failed SVD Factorisation of working matrix")
	}
	s, u, v := t.extractSVD(&svd)

	r, c := m.Dims()
	min := minimum(t.K, r, c)

	// truncate U and V matrices to k << min(m, n)
	uk := u.Slice(0, r, 0, min)
	vk := v.Slice(0, c, 0, min)

	t.Components = uk

	// multiply Sigma by transpose of V.  As sigma is a symmetrical (square) diagonal matrix it is
	// more efficient to simply multiply each element from the array of diagonal values with each
	// element from the matrix V rather than multiplying out the non-zero values from off the diagonal.
	var product mat.Dense
	product.Apply(func(i, j int, v float64) float64 {
		return (v * s[i])
	}, vk.T())

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

func (t *TruncatedSVD) extractSVD(svd *mat.SVD) (s []float64, u, v *mat.Dense) {
	var um, vm mat.Dense
	svd.UTo(&um)
	svd.VTo(&vm)
	s = svd.Values(nil)
	return s, &um, &vm
}
