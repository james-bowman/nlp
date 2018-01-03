package nlp

import (
	"bytes"
	"math"
	"math/rand"
	"testing"

	"github.com/james-bowman/nlp/measures/pairwise"
	"github.com/james-bowman/sparse"
	"gonum.org/v1/gonum/mat"
)

func TestTruncatedSVDFitTransform(t *testing.T) {
	var tests = []struct {
		m      int
		n      int
		input  []float64
		k      int
		r      int
		c      int
		result []float64
	}{
		{
			m: 6, n: 4,
			input: []float64{
				1, 3, 5, 2,
				8, 1, 0, 0,
				2, 1, 0, 1,
				0, 0, 0, 0,
				0, 0, 0, 1,
				0, 1, 0, 0,
			},
			k: 2,
			r: 2, c: 4,
			result: []float64{
				-8.090, -2.212, -1.695, -0.955,
				1.888, -2.524, -4.649, -1.930,
			},
		},
	}

	for _, test := range tests {
		transformer := NewTruncatedSVD(test.k)
		input := mat.NewDense(test.m, test.n, test.input)
		expResult := mat.NewDense(test.r, test.c, test.result)

		result, err := transformer.FitTransform(input)

		if err != nil {
			t.Errorf("Failed Truncated SVD transform caused by %v", err)
		}

		if !mat.EqualApprox(expResult, result, 0.01) {
			t.Logf("Expected matrix: \n%v\n but found: \n%v\n",
				mat.Formatted(expResult),
				mat.Formatted(result))
			t.Fail()
		}

		result2, err := transformer.Transform(input)

		if err != nil {
			t.Errorf("Failed Truncated SVD transform caused by %v", err)
		}

		if !mat.EqualApprox(result, result2, 0.001) {
			t.Logf("First matrix: \n%v\n but second matrix: \n%v\n",
				mat.Formatted(result),
				mat.Formatted(result2))
			t.Fail()
		}
	}
}

func TestTruncatedSVDSaveLoad(t *testing.T) {
	var transforms = []struct {
		wanted *TruncatedSVD
	}{
		{
			wanted: &TruncatedSVD{
				Components: mat.NewDense(4, 2, []float64{
					1, 5,
					3, 2,
					9, 0,
					8, 4,
				}),
				K: 2,
			},
		},
	}

	for ti, test := range transforms {
		t.Logf("**** TestTruncatedSVDSaveLoad - Test Run %d.\n", ti+1)

		buf := new(bytes.Buffer)
		if err := test.wanted.Save(buf); err != nil {
			t.Errorf("Error encoding: %v\n", err)
			continue
		}

		var b TruncatedSVD
		if err := b.Load(buf); err != nil {
			t.Errorf("Error unencoding: %v\n", err)
			continue
		}

		if !mat.Equal(test.wanted.Components, b.Components) {
			t.Logf("Components mismatch: Wanted %v but got %v\n", mat.Formatted(test.wanted.Components), mat.Formatted(b.Components))
			t.Fail()
		}
		if test.wanted.K != b.K {
			t.Logf("K value mismatch: Wanted %d but got %d\n", test.wanted.K, b.K)
			t.Fail()
		}
	}
}

func TestSignRandomProjections(t *testing.T) {
	tests := []struct {
		rows int
		cols int
		bits int
	}{
		{rows: 100, cols: 1000, bits: 1024},
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
		q := reducedDimQuery.(*sparse.BinaryVec)

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
		// be less than 0.3
		r, c := m.Dims()
		if r != test.bits || c != test.cols {
			t.Errorf("Test %d: Expected output matrix to be %dx%d but was %dx%d\n", ti, test.bits, test.cols, r, c)
		}
		if avgDiff >= 0.03 {
			t.Errorf("Test %d: Expected difference between vector spaces %f but was %f\n", ti, 0.3, avgDiff)
		}
	}
}

/*
func TestSignRandomProjections2(t *testing.T) {
	//var density float32
	//density = 1

	tests := []struct {
		r           int
		comparisons int
		bits        int
	}{
		{100, 100, 1},
		{100, 100, 2},
		{100, 100, 4},
		{100, 100, 8},
		{100, 100, 16},
		{100, 100, 32},
		{100, 100, 64},
		{100, 100, 128},
		{100, 100, 256},
		{100, 100, 512},
		{100, 100, 1024},
		{100, 100, 2048},

		{100, 1000, 1},
		{100, 1000, 2},
		{100, 1000, 4},
		{100, 1000, 8},
		{100, 1000, 16},
		{100, 1000, 32},
		{100, 1000, 64},
		{100, 1000, 128},
		{100, 1000, 256},
		{100, 1000, 512},
		{100, 1000, 1024},
		{100, 1000, 2048},

		{100, 10000, 1},
		{100, 10000, 2},
		{100, 10000, 4},
		{100, 10000, 8},
		{100, 10000, 16},
		{100, 10000, 32},
		{100, 10000, 64},
		{100, 10000, 128},
		{100, 10000, 256},
		{100, 10000, 512},
		{100, 10000, 1024},
		{100, 10000, 2048},

		{100, 100000, 1},
		{100, 100000, 2},
		{100, 100000, 4},
		{100, 100000, 8},
		{100, 100000, 16},
		{100, 100000, 32},
		{100, 100000, 64},
		{100, 100000, 128},
		{100, 100000, 256},
		{100, 100000, 512},
		{100, 100000, 1024},
		{100, 100000, 2048},
	}

	for _, test := range tests {
		//matrix := normalise(sparse.Random(sparse.DenseFormat, test.r, test.comparisons + 1, density).(*mat.Dense))
		matrix := mat.NewDense(test.r, test.comparisons+1, nil)
		for i := 0; i < test.r; i++ {
			for j := 0; j < test.comparisons+1; j++ {
				matrix.Set(i, j, rand.NormFloat64())
			}
		}
		//matrix = normalise(matrix)

		searchSpace := matrix.Slice(0, test.r, 0, test.comparisons+1).(*mat.Dense)
		query := matrix.ColView(test.comparisons)

		transformer := NewSignRandomProjection(test.bits)
		binSearchSpace, err := transformer.FitTransform(searchSpace)
		if err != nil {
			t.Error(err)
		}
		hashSearchSpace := binSearchSpace.(*sparse.Binary)
		binQuery, err := transformer.Transform(query)
		if err != nil {
			t.Error(err)
		}
		hashQuery := binQuery.(*sparse.BinaryVec)

		hashes := make(map[string][]int)
		var culmDiff float64
		for j := 0; j < test.comparisons+1; j++ {
			lshSim := pairwise.HammingSimilarity(hashQuery, hashSearchSpace.ColView(j))
			//cosSim := pairwise.CosineSimilarity(query, searchSpace.ColView(j))
			angSim := pairwise.AngularSimilarity(query, searchSpace.ColView(j))

			diff := math.Abs(lshSim-angSim) / angSim
			culmDiff += diff
			//fmt.Printf("\tComparing - Cos: %.4f, Ang: %.4f, LSH: %.4f, Diff: %.4f\n", cosSim, angSim, lshSim, diff)

			key := hashSearchSpace.ColView(j).(*sparse.BinaryVec).String()
			hashes[key] = append(hashes[key], j)
		}

		var ttlCollisions int
		var maxCollision int
		for _, v := range hashes {
			pop := len(v)
			ttlCollisions += pop - 1
			if pop > maxCollision {
				maxCollision = pop
			}
		}

		avgCollision := (test.comparisons + 1) / len(hashes)

		avgDiff := culmDiff / float64(test.comparisons+1)
		fmt.Printf("%d dimensions, %d observations (%d bit hashes) : %.4f avg diff, distinct hashes: %d, Collisions: %d ttl, %d avg, %d max\n", test.r, test.comparisons+1, test.bits, avgDiff, len(hashes), ttlCollisions, avgCollision, maxCollision)
	}
}
*/
