package nlp

import (
	"testing"

	"github.com/james-bowman/sparse"
)

var stopWords = []string{"a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "around", "as", "at", "back", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves"}

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
		stop      []string
		vocabSize int
	}{
		{trainSet, []string{}, 26},
		{trainSet[0:1], []string{}, 8},
		{trainSet, stopWords, 18},
	}

	for testRun, test := range tests {
		t.Logf("**** Test Run %d.\n", testRun+1)
		vectoriser := NewCountVectoriser(test.stop...)

		vectoriser.Fit(test.train...)

		if len(vectoriser.Vocabulary) != test.vocabSize {
			t.Logf("Expected training dataset %v of size %d but found vocabulary %v of size %d",
				test.train, test.vocabSize, vectoriser.Vocabulary, len(vectoriser.Vocabulary))
			t.Fail()
		}
	}
}

func TestCountVectoriserPartialFit(t *testing.T) {
	var tests = []struct {
		train     []string
		stop      []string
		vocabSize int
	}{
		{trainSet, []string{}, 26},
		{trainSet[0:1], []string{}, 8},
		{trainSet, stopWords, 18},
	}

	for testRun, test := range tests {
		t.Logf("**** Test Run %d.\n", testRun+1)
		vectoriser := NewCountVectoriser(test.stop...)

		for _, v := range test.train {
			vectoriser.PartialFit(v)
		}

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
		stop      []string
		test      []string
	}{
		{trainSet, 26, []string{}, testSet},
		{trainSet[0:1], 8, []string{}, testSet[0:3]},
		{testSet, 26, []string{}, testSet},
		{testSet, 19, stopWords, testSet},
	}

	for testRun, test := range tests {
		t.Logf("**** Test Run %d.\n", testRun+1)

		vectoriser := NewCountVectoriser(test.stop...)
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
		stop     []string
		test     []string
	}{
		{trainSet, 33, 260000, []string{}, testSet},
		{trainSet[0:1], 11, 260000, []string{}, testSet[0:3]},
		{testSet, 33, 260001, []string{}, testSet},
		{testSet, 21, 260000, stopWords, testSet},
	}

	for testRun, test := range tests {
		t.Logf("**** Test Run %d.\n", testRun+1)
		vectoriser := NewHashingVectoriser(test.features, test.stop...)
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
