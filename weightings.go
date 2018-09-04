package nlp

import (
	"io"
	"math"

	"github.com/james-bowman/sparse"
	"gonum.org/v1/gonum/mat"
)

// TfidfTransformer takes a raw term document matrix and weights each raw term frequency
// value depending upon how commonly it occurs across all documents within the corpus.
// For example a very commonly occurring word like `the` is likely to occur in all documents
// and so would be weighted down.
// More precisely, TfidfTransformer applies a tf-idf algorithm to the matrix where each
// term frequency is multiplied by the inverse document frequency.  Inverse document
// frequency is calculated as log(n/df) where df is the number of documents in which the
// term occurs and n is the total number of documents within the corpus.  We add 1 to both n
// and df before division to prevent division by zero.
type TfidfTransformer struct {
	transform *sparse.DIA
}

// NewTfidfTransformer constructs a new TfidfTransformer.
func NewTfidfTransformer() *TfidfTransformer {
	return &TfidfTransformer{}
}

// Fit takes a training term document matrix, counts term occurrences across all documents
// and constructs an inverse document frequency transform to apply to matrices in subsequent
// calls to Transform().
func (t *TfidfTransformer) Fit(matrix mat.Matrix) Transformer {
	if t, isTypeConv := matrix.(sparse.TypeConverter); isTypeConv {
		matrix = t.ToCSR()
	}
	m, n := matrix.Dims()

	weights := make([]float64, m)
	var df int
	if csr, ok := matrix.(*sparse.CSR); ok {
		for i := 0; i < m; i++ {
			weights[i] = math.Log(float64(1+n) / float64(1+csr.RowNNZ(i)))
		}
	} else {
		for i := 0; i < m; i++ {
			df = 0
			for j := 0; j < n; j++ {
				if matrix.At(i, j) != 0 {
					df++
				}
			}
			weights[i] = math.Log(float64(1+n) / float64(1+df))
		}
	}

	// build a diagonal matrix from array of term weighting values for subsequent
	// multiplication with term document matrics
	t.transform = sparse.NewDIA(m, m, weights)

	return t
}

// Transform applies the inverse document frequency (IDF) transform by multiplying
// each term frequency by its corresponding IDF value.  This has the effect of weighting
// each term frequency according to how often it appears across the whole document corpus
// so that naturally frequent occurring words are given less weight than uncommon ones.
// The returned matrix is a sparse matrix type.
func (t *TfidfTransformer) Transform(matrix mat.Matrix) (mat.Matrix, error) {
	if t, isTypeConv := matrix.(sparse.TypeConverter); isTypeConv {
		matrix = t.ToCSR()
	}
	var product sparse.CSR

	// simply multiply the matrix by our idf transform (the diagonal matrix of term weights)
	product.Mul(t.transform, matrix)

	// todo: possibly L2 norm matrix to remove any bias caused by documents of different
	// lengths where longer documents naturally have more words and so higher word counts

	return &product, nil
}

// FitTransform is exactly equivalent to calling Fit() followed by Transform() on the
// same matrix.  This is a convenience where separate training data is not being
// used to fit the model i.e. the model is fitted on the fly to the test data.
// The returned matrix is a sparse matrix type.
func (t *TfidfTransformer) FitTransform(matrix mat.Matrix) (mat.Matrix, error) {
	if t, isTypeConv := matrix.(sparse.TypeConverter); isTypeConv {
		matrix = t.ToCSR()
	}
	return t.Fit(matrix).Transform(matrix)
}

// Save binary serialises the model and writes it into w.  This is useful for persisting
// a trained model to disk so that it may be loaded (using the Load() method)in another
// context (e.g. production) for reproducible results.
func (t TfidfTransformer) Save(w io.Writer) error {
	_, err := t.transform.MarshalBinaryTo(w)

	return err
}

// Load binary deserialises the previously serialised model into the receiver.  This is
// useful for loading a previously trained and saved model from another context
// (e.g. offline training) for use within another context (e.g. production) for
// reproducible results.  Load should only be performed with trusted data.
func (t *TfidfTransformer) Load(r io.Reader) error {
	var model sparse.DIA

	if _, err := model.UnmarshalBinaryFrom(r); err != nil {
		return err
	}
	t.transform = &model

	return nil
}
