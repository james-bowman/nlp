package nlp

import (
	"math"
	"testing"

	"github.com/james-bowman/nlp/measures/pairwise"
	"github.com/james-bowman/sparse"
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
)

func TestSignRandomProjection(t *testing.T) {
	tests := []struct {
		rows int
		cols int
		bits int
	}{
		{rows: 100, cols: 1000, bits: 1024},
		{rows: 100, cols: 1000, bits: 256},
	}

	for ti, test := range tests {
		// Given an input matrix and a query matching one column
		matrix := mat.NewDense(test.rows, test.cols, nil)
		for i := 0; i < test.rows; i++ {
			for j := 0; j < test.cols; j++ {
				matrix.Set(i, j, rand.Float64())
			}
		}

		query := matrix.ColView(0)

		// When transformed using sign random projections
		transformer := NewSignRandomProjection(test.bits)
		reducedDimMatrix, err := transformer.FitTransform(matrix)
		if err != nil {
			t.Errorf("Failed to transform matrix because %v\n", err)
		}
		m := reducedDimMatrix.(*sparse.Binary)

		reducedDimQuery, err := transformer.Transform(query)
		if err != nil {
			t.Errorf("Failed to transform query because %v\n", err)
		}
		q := reducedDimQuery.(*sparse.Binary).ColView(0)

		var culmDiff float64
		for i := 0; i < test.cols; i++ {
			angSim := pairwise.AngularSimilarity(query, matrix.ColView(i))
			lshSim := pairwise.HammingSimilarity(q, m.ColView(i))

			if i == 0 {
				if math.Abs(angSim-lshSim) >= 0.0000001 {
					t.Errorf("Test %d: Expected matching similarity but found %.10f (Ang) and %.10f (LSH)\n", ti, angSim, lshSim)
				}
			}

			diff := math.Abs(lshSim-angSim) / angSim
			culmDiff += diff
		}
		avgDiff := culmDiff / float64(test.cols)

		// Then output matrix should be of specified length,
		// matching column should still have similarity of ~1.0 and
		// avg difference betwen angular and hamming similarities should
		// be less than 0.03
		r, c := m.Dims()
		if r != test.bits || c != test.cols {
			t.Errorf("Test %d: Expected output matrix to be %dx%d but was %dx%d\n", ti, test.bits, test.cols, r, c)
		}
		if avgDiff >= 0.03 {
			t.Errorf("Test %d: Expected difference between vector spaces %f but was %f\n", ti, 0.03, avgDiff)
		}
	}
}

func TestRandomProjection(t *testing.T) {
	tests := []struct {
		k       int
		rows    int
		cols    int
		density float32
	}{
		{k: 400, rows: 700, cols: 600, density: 0.02},
		{k: 400, rows: 800, cols: 800, density: 0.02},
	}

	for ti, test := range tests {
		matrix := sparse.Random(sparse.CSRFormat, test.rows, test.cols, test.density).(sparse.TypeConverter).ToCSR()
		query := matrix.ToCSC().ColView(0)

		// When transformed using sign random projections
		transformer := NewRandomProjection(test.k, float64(test.density))
		transformer.rnd = rand.New(rand.NewSource(uint64(0)))
		reducedDimMatrix, err := transformer.FitTransform(matrix)
		if err != nil {
			t.Errorf("Failed to transform matrix because %v\n", err)
		}
		m := reducedDimMatrix.(*sparse.CSR).ToCSC()

		reducedDimQuery, err := transformer.Transform(query)
		if err != nil {
			t.Errorf("Failed to transform query because %v\n", err)
		}
		q := reducedDimQuery.(*sparse.CSR).ToCSC().ColView(0)

		var culmDiff float64
		ColDo(matrix, func(j int, v mat.Vector) {
			angSim := pairwise.CosineSimilarity(query, v)
			lshSim := pairwise.CosineSimilarity(q, m.ColView(j))

			if j == 0 {
				if math.Abs(angSim-lshSim) >= 0.0000001 {
					t.Errorf("Test %d: Expected matching similarity but found %.10f (Ang) and %.10f (LSH)\n", ti, angSim, lshSim)
				}
			}

			//diff := math.Abs(lshSim-angSim) / angSim
			diff := math.Abs(lshSim - angSim)
			culmDiff += diff
		})
		t.Logf("CulmDiff = %f\n", culmDiff)
		avgDiff := culmDiff / float64(test.cols)

		// Then output matrix should be of specified length,
		// matching column should still have similarity of ~1.0 and
		// avg difference betwen angular and hamming similarities should
		// be less than 0.03
		r, c := reducedDimMatrix.Dims()
		if r != test.k || c != test.cols {
			t.Errorf("Test %d: Expected output matrix to be %dx%d but was %dx%d\n", ti, test.k, test.cols, r, c)
		}
		if avgDiff >= 0.05 {
			t.Errorf("Test %d: Expected difference between vector spaces %f but was %f\n", ti, 0.03, avgDiff)
		}
	}
}

func TestRandomIndexing(t *testing.T) {
	tests := []struct {
		k       int
		rows    int
		cols    int
		density float32
	}{
		{k: 400, rows: 700, cols: 600, density: 0.02},
		{k: 400, rows: 800, cols: 800, density: 0.02},
	}

	for ti, test := range tests {
		matrix := sparse.Random(sparse.CSRFormat, test.rows, test.cols, test.density).(sparse.TypeConverter).ToCSR()
		query := matrix.ToCSC().ColView(0)

		// When transformed using sign random projections
		transformer := NewRandomIndexing(test.k, float64(test.density))
		transformer.rnd = rand.New(rand.NewSource(uint64(0)))
		reducedDimMatrix, err := transformer.FitTransform(matrix)
		if err != nil {
			t.Errorf("Failed to transform matrix because %v\n", err)
		}
		m := reducedDimMatrix.(*sparse.CSC)

		reducedDimQuery, err := transformer.Transform(query)
		if err != nil {
			t.Errorf("Failed to transform query because %v\n", err)
		}
		q := reducedDimQuery.(mat.ColViewer).ColView(0)

		var culmDiff float64
		ColDo(matrix, func(j int, v mat.Vector) {
			angSim := pairwise.CosineSimilarity(query, v)
			lshSim := pairwise.CosineSimilarity(q, m.ColView(j))

			if j == 0 {
				if math.Abs(angSim-lshSim) >= 0.05 {
					t.Errorf("Test %d: Expected matching similarity but found %.10f (Ang) and %.10f (LSH)\n", ti, angSim, lshSim)
				}
			}

			//diff := math.Abs(lshSim-angSim) / angSim
			diff := math.Abs(lshSim - angSim)
			culmDiff += diff
		})
		t.Logf("CulmDiff = %f\n", culmDiff)
		avgDiff := culmDiff / float64(test.cols)

		// Then output matrix should be of specified length,
		// matching column should still have similarity of ~1.0 and
		// avg difference betwen angular and hamming similarities should
		// be less than 0.03
		r, c := reducedDimMatrix.Dims()
		if r != test.k || c != test.cols {
			t.Errorf("Test %d: Expected output matrix to be %dx%d but was %dx%d\n", ti, test.k, test.cols, r, c)
		}
		if avgDiff >= 0.05 {
			t.Errorf("Test %d: Expected difference between vector spaces %f but was %f\n", ti, 0.03, avgDiff)
		}
	}
}

func TestReflectiveRandomIndexing(t *testing.T) {
	tests := []struct {
		k       int
		rows    int
		cols    int
		density float32
	}{
		{k: 400, rows: 700, cols: 600, density: 0.02},
		{k: 400, rows: 800, cols: 800, density: 0.02},
	}

	for ti, test := range tests {
		matrix := sparse.Random(sparse.CSRFormat, test.rows, test.cols, test.density).(sparse.TypeConverter).ToCSR()
		query := matrix.ToCSC().ColView(0)

		// When transformed using Reflective Random Indexing
		transformer := NewReflectiveRandomIndexing(test.k, ColBasedRI, 0, float64(test.density))
		transformer.rnd = rand.New(rand.NewSource(uint64(0)))
		reducedDimMatrix, err := transformer.FitTransform(matrix)
		if err != nil {
			t.Errorf("Failed to transform matrix because %v\n", err)
		}
		m := reducedDimMatrix.(mat.ColViewer)

		reducedDimQuery, err := transformer.Transform(query)
		if err != nil {
			t.Errorf("Failed to transform query because %v\n", err)
		}
		q := reducedDimQuery.(mat.ColViewer).ColView(0)

		var culmDiff float64
		ColDo(matrix, func(j int, v mat.Vector) {
			origSim := pairwise.CosineSimilarity(query, v)
			redSim := pairwise.CosineSimilarity(q, m.ColView(j))

			if j == 0 {
				if math.Abs(origSim-redSim) >= 0.0000001 {
					t.Errorf("Test %d: Expected matching similarity but found %.10f (Original) and %.10f (Reduced)\n", ti, origSim, redSim)
				}
			}

			diff := math.Abs(redSim - origSim)
			culmDiff += diff
		})
		t.Logf("CulmDiff = %f\n", culmDiff)
		avgDiff := culmDiff / float64(test.cols)

		// Then output matrix should be of specified length,
		// matching column should still have similarity of ~1.0 and
		// avg difference betwen angular and hamming similarities should
		// be less than 0.03
		r, c := reducedDimMatrix.Dims()
		if r != test.k || c != test.cols {
			t.Errorf("Test %d: Expected output matrix to be %dx%d but was %dx%d\n", ti, test.k, test.cols, r, c)
		}
		if avgDiff >= 0.12 {
			t.Errorf("Test %d: Expected difference between vector spaces %f but was %f\n", ti, 0.12, avgDiff)
		}
	}
}
