package nlp

import (
	"bytes"
	"testing"

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

func TestPCAFitTransform(t *testing.T) {
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
				-7.478, -0.128, 1.591, 0.496,
				2.937, 2.581, 4.240, 1.110,
			},
		},
	}

	for _, test := range tests {
		transformer := NewPCA(test.k)
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
