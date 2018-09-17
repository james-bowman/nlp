package nlp

import (
	"github.com/james-bowman/sparse"
	"gonum.org/v1/gonum/mat"
)

// ColDo executes fn for each column j in m.  If the matrix implements the mat.ColViewer
// interface then this interface will be used to iterate over the column vectors more
// efficiently.  If the matrix implements the sparse.TypeConverter interface then the
// matrix will be converted to a CSC matrix (which implements the mat.ColViewer
// interface) so that it can benefit from the same optimisation.
func ColDo(m mat.Matrix, fn func(j int, vec mat.Vector)) {
	if v, isOk := m.(mat.Vector); isOk {
		fn(0, v)
		return
	}

	if cv, isOk := m.(mat.ColViewer); isOk {
		_, c := m.Dims()
		for j := 0; j < c; j++ {
			fn(j, cv.ColView(j))
		}
		return
	}

	if sv, isOk := m.(sparse.TypeConverter); isOk {
		csc := sv.ToCSC()
		_, c := csc.Dims()
		for j := 0; j < c; j++ {
			fn(j, csc.ColView(j))
		}
		return
	}

	r, c := m.Dims()
	for j := 0; j < c; j++ {
		fn(j, mat.NewVecDense(r, mat.Col(nil, j, m)))
	}
}

// ColNonZeroElemDo executes fn for each non-zero element in column j of matrix m.
// If m implements mat.ColNonZeroDoer then this interface will be used to perform
// the iteration.
func ColNonZeroElemDo(m mat.Matrix, j int, fn func(i, j int, v float64)) {
	colNonZeroDoer, isSparse := m.(mat.ColNonZeroDoer)
	r, _ := m.Dims()

	if isSparse {
		colNonZeroDoer.DoColNonZero(j, fn)
	} else {
		for i := 0; i < r; i++ {
			v := m.At(i, j)
			if v != 0 {
				fn(i, j, v)
			}
		}
	}
}
