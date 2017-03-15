package nlp

import (
	"testing"

	"github.com/gonum/matrix/mat64"
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
		input := mat64.NewDense(test.m, test.n, test.input)
		expResult := mat64.NewDense(test.r, test.c, test.result)

		result, err := transformer.FitTransform(input)

		if err != nil {
			t.Errorf("Failed Truncated SVD transform caused by %v", err)
		}

		if !mat64.EqualApprox(expResult, result, 0.01) {
			t.Logf("Expected matrix: \n%v\n but found: \n%v\n",
				mat64.Formatted(expResult),
				mat64.Formatted(result))
			t.Fail()
		}

		result2, err := transformer.Transform(input)

		if err != nil {
			t.Errorf("Failed Truncated SVD transform caused by %v", err)
		}

		if !mat64.EqualApprox(result, result2, 0.001) {
			t.Logf("First matrix: \n%v\n but second matrix: \n%v\n",
				mat64.Formatted(result),
				mat64.Formatted(result2))
			t.Fail()
		}
	}
}
