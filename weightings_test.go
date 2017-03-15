package nlp

import (
	"testing"

	"github.com/gonum/matrix/mat64"
)

func TestTfidfTransformerFit(t *testing.T) {
	var tests = []struct {
		m         int
		n         int
		input     []float64
		dim       int
		transform []float64
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
			dim: 6,
			transform: []float64{
				0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
				0.000, 0.511, 0.000, 0.000, 0.000, 0.000,
				0.000, 0.000, 0.223, 0.000, 0.000, 0.000,
				0.000, 0.000, 0.000, 1.609, 0.000, 0.000,
				0.000, 0.000, 0.000, 0.000, 0.916, 0.000,
				0.000, 0.000, 0.000, 0.000, 0.000, 0.916,
			},
		},
	}

	for _, test := range tests {
		transformer := NewTfidfTransformer()
		input := mat64.NewDense(test.m, test.n, test.input)
		output := mat64.NewDense(test.dim, test.dim, test.transform)

		transformer.Fit(input)

		if !mat64.EqualApprox(output, transformer.transform, 0.001) {
			t.Logf("Expected matrix: \n%v\n but found: \n%v\n",
				mat64.Formatted(output),
				mat64.Formatted(transformer.transform))
			t.Fail()
		}
	}
}

func TestTfidfTransformerTransform(t *testing.T) {
	var tests = []struct {
		m      int
		n      int
		input  []float64
		tm     int
		tn     int
		output []float64
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
			tm: 6, tn: 4,
			output: []float64{
				0.000, 0.000, 0.000, 0.000,
				4.087, 0.511, 0.000, 0.000,
				0.446, 0.223, 0.000, 0.223,
				0.000, 0.000, 0.000, 0.000,
				0.000, 0.000, 0.000, 0.916,
				0.000, 0.916, 0.000, 0.000,
			},
		},
	}

	for _, test := range tests {
		transformer := NewTfidfTransformer()
		input := mat64.NewDense(test.m, test.n, test.input)
		output := mat64.NewDense(test.tm, test.tn, test.output)

		result, err := transformer.FitTransform(input)

		if err != nil {
			t.Errorf("Failed tfidf fit transform caused by %v", err)
		}

		if !mat64.EqualApprox(output, result, 0.001) {
			t.Logf("Expected matrix: \n%v\n but found: \n%v\n",
				mat64.Formatted(output),
				mat64.Formatted(result))
			t.Fail()
		}

		// test that subsequent transforms produce same result as initial
		result2, err := transformer.Transform(input)

		if err != nil {
			t.Errorf("Failed tfidf fit transform caused by %v", err)
		}

		if !mat64.Equal(result, result2) {
			t.Logf("Expected matrix: \n%v\n but found: \n%v\n",
				mat64.Formatted(result),
				mat64.Formatted(result2))
			t.Fail()
		}
	}
}
