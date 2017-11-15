package nlp

import (
	"fmt"
	"io"
	"encoding/binary"
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
	Components *mat.Dense

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
// on the same matrix.  This is a useful shortcut where separate training data is not being
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

	t.Components = uk.(*mat.Dense)

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

// Save binary serialises the model and writes it into w.  This is useful for persisting
// a trained model to disk so that it may be loaded (using the Load() method)in another 
// context (e.g. production) for reproducable results.
func (t TruncatedSVD) Save(w io.Writer) error {
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], uint64(t.K))
	if _, err := w.Write(buf[:]); err != nil {
		return err
	}
	
	_, err := t.Components.MarshalBinaryTo(w)
		
	return err
}

// Load binary deserialises the previously serialised model into the receiver.  This is
// useful for loading a previously trained and saved model from another context 
// (e.g. offline training) for use within another context (e.g. production) for 
// reproducable results.  Load should only be performed with trusted data.
func (t *TruncatedSVD) Load(r io.Reader) error {
	var n   int
	var buf [8]byte
	var err error
	for n < len(buf) && err == nil {
		var nn int
		nn, err = r.Read(buf[n:])
		n += nn
	}
	if err == io.EOF {
		return io.ErrUnexpectedEOF
	}
	if err != nil {
		return err
	}
	k := int(binary.LittleEndian.Uint64(buf[:]))

	var model mat.Dense
	if _, err := model.UnmarshalBinaryFrom(r); err != nil {
		return err
	}

	t.K = k
	t.Components = &model

	return nil
}
