package nlp

import "github.com/gonum/matrix/mat64"

// CosineSimilarity calculates the distance between the angles of 2 vectors i.e. how
// similar they are.  Possible values range up to 1 (exact match)
func CosineSimilarity(a, b *mat64.Vector) float64 {
	// Cosine angle between two vectors is equal to their dot product divided by
	// the product of their norms
	dotProduct := mat64.Dot(a, b)
	norma := mat64.Norm(a, 2.0)
	normb := mat64.Norm(b, 2.0)

	return (dotProduct / (norma * normb))
}
