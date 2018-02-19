package nlp

import (
	"math"
	"testing"

	"golang.org/x/exp/rand"

	"gonum.org/v1/gonum/mat"
)

func TestLDAFit(t *testing.T) {
	tests := []struct {
		topics         int
		r, c           int
		data           []float64
		expectedTopics [][]float64
	}{
		{
			topics: 3,
			r:      9, c: 9,
			data: []float64{
				3, 3, 3, 0, 0, 0, 0, 0, 0,
				3, 3, 3, 0, 0, 0, 0, 0, 0,
				3, 3, 3, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 3, 3, 3, 0, 0, 0,
				0, 0, 0, 3, 3, 3, 0, 0, 0,
				0, 0, 0, 3, 3, 3, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 4, 4, 4,
				0, 0, 0, 0, 0, 0, 4, 4, 4,
				0, 0, 0, 0, 0, 0, 4, 4, 4,
			},
			expectedTopics: [][]float64{
				{0.33, 0.33, 0.33, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0.33, 0.33, 0.33},
				{0, 0, 0, 0.33, 0.33, 0.33, 0, 0, 0},
			},
		},
		{
			topics: 3,
			r:      9, c: 9,
			data: []float64{
				3, 3, 3, 0, 0, 0, 0, 0, 0,
				3, 3, 3, 0, 0, 0, 0, 0, 0,
				3, 3, 3, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 3, 5, 1, 0, 0, 0,
				0, 0, 0, 3, 5, 0, 0, 0, 0,
				0, 0, 0, 3, 5, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 4, 4, 4,
				0, 0, 0, 0, 0, 0, 4, 4, 4,
				0, 0, 0, 0, 0, 0, 4, 4, 4,
			},
			expectedTopics: [][]float64{
				{0.33, 0.33, 0.33, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0.33, 0.33, 0.33},
				{0, 0, 0, 0.428, 0.285, 0.285, 0, 0, 0},
			},
		},
	}

	for ti, test := range tests {
		// set Rnd to fixed constant seed for deterministic results
		lda := NewLatentDirichletAllocation(test.topics)
		lda.Rnd = rand.New(rand.NewSource(uint64(0)))

		in := mat.NewDense(test.r, test.c, test.data)
		lda.Fit(in)

		components := lda.Components()

		for i := 0; i < test.topics; i++ {
			var sum float64
			for ri, v := range test.expectedTopics[i] {
				cv := components.At(i, ri)
				sum += cv
				if math.Abs(cv-v) > 0.01 {
					t.Errorf("Test %d: Topic (%d) over word (%d) distribution incorrect. Expected %f but received %f\n", ti, i, ri, v, cv)
				}
			}
			if math.Abs(1-sum) > 0.00000001 {
				t.Errorf("Test %d: values in topic (%d) over word distributions should sum to 1 but summed to %f\n", ti, i, sum)
			}
		}
	}
}

func TestLDAFitTransform(t *testing.T) {
	tests := []struct {
		topics       int
		r, c         int
		data         []float64
		expectedDocs [][]float64
	}{
		{
			topics: 3,
			r:      9, c: 9,
			data: []float64{
				3, 3, 3, 0, 0, 0, 0, 0, 0,
				3, 3, 3, 0, 0, 0, 0, 0, 0,
				3, 3, 3, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 3, 3, 3, 0, 0, 0,
				0, 0, 0, 3, 3, 3, 0, 0, 0,
				0, 0, 0, 3, 3, 3, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 4, 4, 4,
				0, 0, 0, 0, 0, 0, 4, 4, 4,
				0, 0, 0, 0, 0, 0, 4, 4, 4,
			},
			expectedDocs: [][]float64{
				{1, 0, 0},
				{1, 0, 0},
				{1, 0, 0},
				{0, 0, 1},
				{0, 0, 1},
				{0, 0, 1},
				{0, 1, 0},
				{0, 1, 0},
				{0, 1, 0},
			},
		},
		{
			topics: 3,
			r:      9, c: 9,
			data: []float64{
				3, 3, 3, 0, 0, 0, 0, 0, 0,
				3, 3, 3, 0, 0, 0, 0, 0, 0,
				3, 3, 3, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 3, 5, 1, 0, 0, 0,
				0, 0, 0, 3, 5, 0, 0, 0, 0,
				0, 0, 0, 3, 5, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 4, 4, 4,
				0, 0, 0, 0, 0, 0, 4, 4, 4,
				0, 0, 0, 0, 0, 0, 4, 4, 4,
			},
			expectedDocs: [][]float64{
				{1, 0, 0},
				{1, 0, 0},
				{1, 0, 0},
				{0, 0, 1},
				{0, 0, 1},
				{0, 0, 1},
				{0, 1, 0},
				{0, 1, 0},
				{0, 1, 0},
			},
		},
	}

	for ti, test := range tests {
		// set Rnd to fixed constant seed for deterministic results
		lda := NewLatentDirichletAllocation(test.topics)
		lda.Rnd = rand.New(rand.NewSource(uint64(0)))

		in := mat.NewDense(test.r, test.c, test.data)
		theta, err := lda.FitTransform(in)
		if err != nil {
			t.Error(err)
		}

		for j := 0; j < test.c; j++ {
			var sum float64
			for ri, v := range test.expectedDocs[j] {
				cv := theta.At(ri, j)
				sum += cv
				if math.Abs(cv-v) > 0.01 {
					t.Errorf("Test %d: Document (%d) over topic (%d) distribution incorrect. Expected %f but received %f\n", ti, j, ri, v, cv)
				}
			}
			if math.Abs(1-sum) > 0.00000001 {
				t.Errorf("Test %d: values in document (%d) over topic distributions should sum to 1 but summed to %f\n", ti, j, sum)
			}
		}
	}
}

func TestLDATransform(t *testing.T) {
	tests := []struct {
		topics int
		r, c   int
		data   []float64
	}{
		{
			topics: 3,
			r:      9, c: 9,
			data: []float64{
				3, 3, 3, 0, 0, 0, 0, 0, 0,
				3, 3, 3, 0, 0, 0, 0, 0, 0,
				3, 3, 3, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 3, 3, 3, 0, 0, 0,
				0, 0, 0, 3, 3, 3, 0, 0, 0,
				0, 0, 0, 3, 3, 3, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 4, 4, 4,
				0, 0, 0, 0, 0, 0, 4, 4, 4,
				0, 0, 0, 0, 0, 0, 4, 4, 4,
			},
		},
		{
			topics: 3,
			r:      9, c: 9,
			data: []float64{
				3, 3, 3, 0, 0, 0, 0, 0, 0,
				3, 3, 3, 0, 0, 0, 0, 0, 0,
				3, 3, 3, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 3, 5, 1, 0, 0, 0,
				0, 0, 0, 3, 5, 0, 0, 0, 0,
				0, 0, 0, 3, 5, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 4, 4, 4,
				0, 0, 0, 0, 0, 0, 4, 4, 4,
				0, 0, 0, 0, 0, 0, 4, 4, 4,
			},
		},
	}

	for ti, test := range tests {
		// set Rnd to fixed constant seed for deterministic results
		lda := NewLatentDirichletAllocation(test.topics)
		lda.Rnd = rand.New(rand.NewSource(uint64(0)))
		lda.PerplexityEvaluationFrequency = 2

		in := mat.NewDense(test.r, test.c, test.data)
		theta, err := lda.FitTransform(in)
		if err != nil {
			t.Error(err)
		}

		tTheta, err := lda.Transform(in)

		if !mat.EqualApprox(theta, tTheta, 0.01) {
			t.Errorf("Test %d: Transformed matrix not equal to FitTransformed\nExpected:\n %v\nbut received:\n %v\n", ti, mat.Formatted(theta), mat.Formatted(tTheta))
		}
	}
}
