package nlp

import "fmt"

func Example() {
	testCorpus := []string{
		"The quick brown fox jumped over the lazy dog",
		"hey diddle diddle, the cat and the fiddle",
		"the cow jumped over the moon",
		"the little dog laughed to see such fun",
		"and the dish ran away with the spoon",
	}

	query := "the brown fox ran around the dog"

	vectoriser := NewCountVectoriser()
	transformer := NewTfidfTransformer()

	// set k (the number of dimensions following truncation) to 4
	reducer := NewTruncatedSVD(4)

	// Transform the corpus into an LSI fitting the model to the documents in the process
	mat, _ := vectoriser.FitTransform(testCorpus...)
	mat, _ = transformer.FitTransform(mat)
	lsi, _ := reducer.FitTransform(mat)

	// run the query through the same pipeline that was fitted to the corpus and
	// to project it into the same dimensional space
	mat, _ = vectoriser.Transform(query)
	mat, _ = transformer.Transform(mat)
	queryVector, _ := reducer.Transform(mat)

	// iterate over document feature vectors (columns) in the LSI and compare with the
	// query vector for similarity.  Similarity is determined by the difference between
	// the angles of the vectors known as the cosine similarity
	highestSimilarity := -1.0
	var matched int
	_, docs := lsi.Dims()
	for i := 0; i < docs; i++ {
		similarity := CosineSimilarity(queryVector.ColView(0), lsi.ColView(i))
		if similarity > highestSimilarity {
			matched = i
			highestSimilarity = similarity
		}
	}

	fmt.Printf("Matched '%s'", testCorpus[matched])
	// Output: Matched 'The quick brown fox jumped over the lazy dog'
}
