package nlp

import "testing"

var trainSet = []string{
	"The quick brown fox jumped over the. Lazy dog",
	"the brown Cat sat on the mat",
	"the little dog laughed to see such fun",
	"laughing cow",
	"the cow ran around the dog",
	"spoon dish and plate",
}

var testSet = []string{
	"hey diddle diddle",
	"the cat and the fiddle",
	"the cow jumped over the moon",
	"the quick brown fox jumped over the. Lazy dog",
	"The little dog laughed to see such fun",
	"The dish ran away with the spoon",
}

func TestCountVectoriserFit(t *testing.T) {
	var tests = []struct {
		train     []string
		vocabSize int
	}{
		{trainSet, 26},
		{trainSet[0:1], 8},
	}

	for _, test := range tests {
		vectoriser := NewCountVectoriser()

		vectoriser.Fit(test.train...)

		if len(vectoriser.Vocabulary) != test.vocabSize {
			t.Logf("Expected training dataset %v of size %d but found vocabulary %v of size %d",
				test.train, test.vocabSize, vectoriser.Vocabulary, len(vectoriser.Vocabulary))
			t.Fail()
		}
	}
}

func TestCountVectoriserTransform(t *testing.T) {
	var tests = []struct {
		train     []string
		vocabSize int
		test      []string
	}{
		{trainSet, 26, testSet},
		{trainSet[0:1], 8, testSet[0:3]},
		{testSet, 26, testSet},
	}

	for _, test := range tests {
		vectoriser := NewCountVectoriser()
		vectoriser.Fit(test.train...)

		vec, err := vectoriser.Transform(test.test...)

		if err != nil {
			t.Errorf("Error fitting and applying vectoriser caused by %v", err)
		}

		m, n := vec.Dims()

		if m != test.vocabSize || n != len(test.test) {
			t.Logf("Expected matrix %d x %d but found %d x %d", test.vocabSize, len(test.test), m, n)
			t.Fail()
		}
	}
}
