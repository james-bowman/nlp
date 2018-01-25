package nlp

import "gonum.org/v1/gonum/mat"

// ColDo executes fn for each column in m
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
