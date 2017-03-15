package nlp

import (
	"regexp"
	"strings"

	"github.com/gonum/matrix/mat64"
)

type CountVectoriser struct {
	Vocabulary    map[string]int
	wordTokeniser *regexp.Regexp
}

func NewCountVectoriser() *CountVectoriser {
	return &CountVectoriser{Vocabulary: make(map[string]int), wordTokeniser: regexp.MustCompile("\\w+")}
}

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

// Add the document to the term document matrix under construction
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
