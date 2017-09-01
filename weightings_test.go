package nlp

import (
	"testing"

	"github.com/james-bowman/sparse"
	"gonum.org/v1/gonum/mat"
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
				0,
				0.5108256237659907,
				0.22314355131420976,
				1.6094379124341003,
				0.9162907318741551,
				0.9162907318741551,
			},
		},
	}

	for _, test := range tests {
		transformer := NewTfidfTransformer()
		input := mat.NewDense(test.m, test.n, test.input)

		transformer.Fit(input)

		weights := transformer.transform.(*sparse.DIA).Diagonal()
		for i, v := range weights {
			if v != test.transform[i] {
				t.Logf("Expected weights: \n%v\n but found: \n%v\n",
					test.transform, weights)
				t.Fail()
			}
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
		input := mat.NewDense(test.m, test.n, test.input)
		output := mat.NewDense(test.tm, test.tn, test.output)

		result, err := transformer.FitTransform(input)

		if err != nil {
			t.Errorf("Failed tfidf fit transform caused by %v", err)
		}

		if !mat.EqualApprox(output, result, 0.001) {
			t.Logf("Expected matrix: \n%v\n but found: \n%v\n",
				mat.Formatted(output),
				mat.Formatted(result))
			t.Fail()
		}

		// test that subsequent transforms produce same result as initial
		result2, err := transformer.Transform(input)

		if err != nil {
			t.Errorf("Failed tfidf fit transform caused by %v", err)
		}

		if !mat.Equal(result, result2) {
			t.Logf("Expected matrix: \n%v\n but found: \n%v\n",
				mat.Formatted(result),
				mat.Formatted(result2))
			t.Fail()
		}
	}
}

func benchmarkTFIDFFitTransform(t Transformer, m, n int, b *testing.B) {
	mat := mat.NewDense(m, n, nil)

	for n := 0; n < b.N; n++ {
		t.FitTransform(mat)
	}
}

func BenchmarkTFIDFFitTransform20x10(b *testing.B) {
	benchmarkTFIDFFitTransform(NewTfidfTransformer(), 20, 10, b)
}
func BenchmarkTFIDFFitTransform200x100(b *testing.B) {
	benchmarkTFIDFFitTransform(NewTfidfTransformer(), 200, 100, b)
}
func BenchmarkTFIDFFitTransform2000x1000(b *testing.B) {
	benchmarkTFIDFFitTransform(NewTfidfTransformer(), 2000, 1000, b)
}
func BenchmarkTFIDFFitTransform20000x10000(b *testing.B) {
	benchmarkTFIDFFitTransform(NewTfidfTransformer(), 20000, 10000, b)
}
