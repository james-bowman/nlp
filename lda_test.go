package nlp_test

import (
	"fmt"
	"math"
	"testing"

	"golang.org/x/exp/rand"

	"github.com/james-bowman/nlp"
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
		lda := nlp.NewLatentDirichletAllocation(test.topics)
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
		lda := nlp.NewLatentDirichletAllocation(test.topics)
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
		lda := nlp.NewLatentDirichletAllocation(test.topics)
		lda.Rnd = rand.New(rand.NewSource(uint64(0)))
		lda.PerplexityEvaluationFrequency = 2

		in := mat.NewDense(test.r, test.c, test.data)
		theta, err := lda.FitTransform(in)
		if err != nil {
			t.Error(err)
		}

		tTheta, err := lda.Transform(in)

		if !mat.EqualApprox(theta, tTheta, 0.035) {
			t.Errorf("Test %d: Transformed matrix not equal to FitTransformed\nExpected:\n %v\nbut received:\n %v\n", ti, mat.Formatted(theta), mat.Formatted(tTheta))
		}
	}
}

func ExampleLatentDirichletAllocation() {
	corpus := []string{
		"The quick brown fox jumped over the lazy dog",
		"The cow jumped over the moon",
		"The little dog laughed to see such fun",
	}

	// Create a pipeline with a count vectoriser and LDA transformer for 2 topics
	vectoriser := nlp.NewCountVectoriser(true)
	lda := nlp.NewLatentDirichletAllocation(2)
	pipeline := nlp.NewPipeline(vectoriser, lda)

	docsOverTopics, err := pipeline.FitTransform(corpus...)
	if err != nil {
		fmt.Printf("Failed to model topics for documents because %v", err)
		return
	}

	// Examine Document over topic probability distribution
	dr, dc := docsOverTopics.Dims()
	for doc := 0; doc < dc; doc++ {
		fmt.Printf("\nTopic distribution for document: '%s' -", corpus[doc])
		for topic := 0; topic < dr; topic++ {
			if topic > 0 {
				fmt.Printf(",")
			}
			fmt.Printf(" Topic #%d=%f", topic, docsOverTopics.At(topic, doc))
		}
	}

	// Examine Topic over word probability distribution
	topicsOverWords := lda.Components()
	tr, tc := topicsOverWords.Dims()

	vocab := make([]string, len(vectoriser.Vocabulary))
	for k, v := range vectoriser.Vocabulary {
		vocab[v] = k
	}
	for topic := 0; topic < tr; topic++ {
		fmt.Printf("\nWord distribution for Topic #%d -", topic)
		for word := 0; word < tc; word++ {
			if word > 0 {
				fmt.Printf(",")
			}
			fmt.Printf(" '%s'=%f", vocab[word], topicsOverWords.At(topic, word))
		}
	}
}
