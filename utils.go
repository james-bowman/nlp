package nlp

import "gonum.org/v1/gonum/mat"

// ColDo executes fn for each column j in m
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

	if cv, isOk := m.(mat.RawColViewer); isOk {
		r, c := m.Dims()
		for j := 0; j < c; j++ {
			fn(j, mat.NewVecDense(r, cv.RawColView(j)))
		}
		return
	}

	r, c := m.Dims()
	for j := 0; j < c; j++ {
		fn(j, mat.NewVecDense(r, mat.Col(nil, j, m)))
	}
}

// ColNonZeroElemDo executes fn for each element in column j of matrix m.
// If m implements mat.ColNonZeroDoer then only non-zero elements
// will be visited.
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
