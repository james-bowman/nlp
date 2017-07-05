package nlp

import (
	"testing"

	"github.com/james-bowman/sparse"
)

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
		stop      bool
		vocabSize int
	}{
		{trainSet, false, 26},
		{trainSet[0:1], false, 8},
		{trainSet, true, 18},
	}

	for testRun, test := range tests {
		t.Logf("**** Test Run %d.\n", testRun+1)
		vectoriser := NewCountVectoriser(test.stop)

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
		stop      bool
		test      []string
	}{
		{trainSet, 26, false, testSet},
		{trainSet[0:1], 8, false, testSet[0:3]},
		{testSet, 26, false, testSet},
		{testSet, 19, true, testSet},
	}

	for testRun, test := range tests {
		t.Logf("**** Test Run %d.\n", testRun+1)

		vectoriser := NewCountVectoriser(test.stop)
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

func TestHashingVectoriserTransform(t *testing.T) {
	var tests = []struct {
		train    []string
		nnz      int
		features int
		stop     bool
		test     []string
	}{
		{trainSet, 33, 260000, false, testSet},
		{trainSet[0:1], 11, 260000, false, testSet[0:3]},
		{testSet, 33, 260001, false, testSet},
		{testSet, 21, 260000, true, testSet},
	}

	for testRun, test := range tests {
		t.Logf("**** Test Run %d.\n", testRun+1)
		vectoriser := NewHashingVectoriser(test.stop, test.features)
		vectoriser.Fit(test.train...)

		vec, err := vectoriser.Transform(test.test...)

		if err != nil {
			t.Errorf("Error fitting and applying vectoriser caused by %v", err)
		}

		m, n := vec.Dims()

		if m != test.features || n != len(test.test) || vec.(sparse.Sparser).NNZ() != test.nnz {
			t.Logf("Expected matrix %d x %d with NNZ = %d but found %d x %d with NNZ = %d",
				test.features,
				len(test.test),
				test.nnz,
				m, n,
				vec.(sparse.Sparser).NNZ())
			t.Fail()
		}
	}
}
