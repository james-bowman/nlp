package nlp

import (
	"regexp"
	"strings"

	"github.com/gonum/matrix/mat64"
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
}

// NewCountVectoriser creates a new CountVectoriser
func NewCountVectoriser() *CountVectoriser {
	return &CountVectoriser{Vocabulary: make(map[string]int), wordTokeniser: regexp.MustCompile("\\w+")}
}

// Fit processes the supplied training data (a variable number of strings representing 
// documents).  Each word appearing inside the training data will be added to the 
// Vocabulary
func (v *CountVectoriser) Fit(train ...string) *CountVectoriser {
	i := 0
	for _, doc := range train {
		words := tokenise(doc)

		for _, word := range words {
			_, exists := v.Vocabulary[word]
			if !exists {
				v.Vocabulary[word] = i
				i++
			}
			// todo: add optioinal stop word removal and stemming function callbacks
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
		words := tokenise(doc)

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
	// convert content to lower case, remove punctuation and split into words
	c := strings.ToLower(text)
	// match whole words, removing any punctuation/whitespace
	//re := regexp.MustCompile("\\w+")
	words := v.wordTokeniser.FindAllString(c, -1)

	return words
}
