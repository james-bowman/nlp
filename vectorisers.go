package nlp

import (
	"regexp"
	"strings"

	"github.com/gonum/matrix/mat64"
)

var (
	stopWords = []string{"a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "around", "as", "at", "back", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves"}
)

// CountVectoriser can be used to encode one or more text documents into a term document
// matrix where each column represents a document within the corpus and each row represents
// a term present in the training data set.  Each element represents the frequency the
// corresponding term appears in the corresponding document e.g. tf(t, d) = 5 would mean
// that term t (perhaps the word "dog") appears 5 times in the document d.
type CountVectoriser struct {
	// Vocabulary is a map of words to indices that point to the row number representing
	// that word in the term document matrix output from the Transform() and FitTransform()
	// methods.  The Vocabulary map is populated by the Fit() or FitTransform() methods
	// based upon the words occuring in the datasets supplied to those methods.  Within
	// Transform(), any words found in the test data set that were not present in the
	// training data set supplied to Fit() will not have an entry in the Vocabulary
	// and will be ignored.
	Vocabulary    map[string]int
	wordTokeniser *regexp.Regexp
	stopWords     *regexp.Regexp
}

// NewCountVectoriser creates a new CountVectoriser.  If removeStopwords is true then english stop words will be removed.
func NewCountVectoriser(removeStopwords bool) *CountVectoriser {
	var stop *regexp.Regexp

	if removeStopwords {
		reStr := "\\A("

		for i, word := range stopWords {
			if i != 0 {
				reStr += `|`
			}
			reStr += `\Q` + word + `\E`
		}
		reStr += ")\\z"
		stop = regexp.MustCompile(reStr)
	}
	return &CountVectoriser{Vocabulary: make(map[string]int), wordTokeniser: regexp.MustCompile("\\w+"), stopWords: stop}
}

// Fit processes the supplied training data (a variable number of strings representing
// documents).  Each word appearing inside the training data will be added to the
// Vocabulary
func (v *CountVectoriser) Fit(train ...string) *CountVectoriser {
	i := 0
	for _, doc := range train {
		words := v.tokenise(doc)

		for _, word := range words {
			_, exists := v.Vocabulary[word]
			if !exists {
				// if enabled, remove stop words
				if v.stopWords != nil {
					if v.stopWords.MatchString(word) {
						continue
					}
				}
				v.Vocabulary[word] = i
				i++
			}
		}
	}

	return v
}

// Transform transforms the supplied documents into a term document matrix where each
// column is a feature vector representing one of the supplied documents.  Each element
// represents the frequency with which the associated term for that row occured within
// that document.
func (v *CountVectoriser) Transform(docs ...string) (*mat64.Dense, error) {
	mat := mat64.NewDense(len(v.Vocabulary), len(docs), nil)

	for d, doc := range docs {
		words := v.tokenise(doc)

		for _, word := range words {
			i, exists := v.Vocabulary[word]

			if exists {
				mat.Set(i, d, mat.At(i, d)+1)
			}
		}
	}
	return mat, nil
}

// FitTransform is exactly equivalent to calling Fit() followed by Transform() on the
// same matrix.  This is a convenience where separate trianing data is not being
// used to fit the model i.e. the model is fitted on the fly to the test data.
func (v *CountVectoriser) FitTransform(docs ...string) (*mat64.Dense, error) {
	v.Fit(docs...)
	return v.Transform(docs...)
}

func (v *CountVectoriser) tokenise(text string) []string {
	// convert content to lower case
	c := strings.ToLower(text)

	// match whole words, removing any punctuation/whitespace
	words := v.wordTokeniser.FindAllString(c, -1)

	return words
}
