package nlp_test

import (
	"fmt"
	"math"
	"testing"

	"golang.org/x/exp/rand"

	"github.com/james-bowman/nlp"
	"gonum.org/v1/gonum/mat"
)

var stopWords = []string{"a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "around", "as", "at", "back", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves"}

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
	vectoriser := nlp.NewCountVectoriser(stopWords)
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
