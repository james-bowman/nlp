package nlp

import (
	"container/heap"

	"fmt"

	"gonum.org/v1/gonum/mat"
)

// Comparer is a type of function that compares two *mat.Vector types and
// returns a value indicating how similar they are.
type Comparer func(a, b mat.Vector) float64

// CosineSimilarity calculates the distance between the angles of 2 vectors i.e. how
// similar they are.  Possible values range up to 1 (exact match)
func CosineSimilarity(a, b mat.Vector) float64 {
	// Cosine angle between two vectors is equal to their dot product divided by
	// the product of their norms
	dotProduct := mat.Dot(a, b)
	norma := mat.Norm(a, 2.0)
	normb := mat.Norm(b, 2.0)

	return (dotProduct / (norma * normb))
}

// Find returns the top k (unordered) most similar items from the
// input matrix mat compared to the input matrix vec using the specified similarity
// function f to perform the pairwise comparison.
func Find(vec mat.Matrix, matrix mat.Matrix, k int, f Comparer) []Match {
	_, corpusSize := matrix.Dims()
	results := resultHeap{matches: make([]Match, k)}

	v, ok := vec.(mat.ColViewer)
	if !ok {
		panic(fmt.Errorf("No way of indexing specified Vector"))
	}
	vc := v.ColView(0)

	m, ok := matrix.(mat.ColViewer)
	if !ok {
		panic(fmt.Errorf("No way of indexing specified matrix as a Vector"))
	}
	var point int
	for point = 0; point < k && point < corpusSize; point++ {
		mv := m.ColView(point)
		results.matches[point] = Match{Score: f(vc, mv), ID: point, Vec: mv}
	}
	heap.Init(&results)
	var sim float64
	for i := point; i < corpusSize; i++ {
		mv := m.ColView(i)
		sim = f(vc, mv)
		if sim >= results.matches[0].Score {
			heap.Pop(&results)
			heap.Push(&results, Match{Score: sim, ID: i, Vec: mv})
		}
	}

	return results.matches
}

// Match is a type representing matched items from a matrix including the score
// (how closely the item matched/similarity), the ID of the item and the feature
// vector representing the item for ongoing analysis.
type Match struct {
	Score float64
	ID    int
	Vec   mat.Matrix
}

type resultHeap struct {
	matches []Match
}

func (r resultHeap) Len() int { return len(r.matches) }

func (r resultHeap) Less(i, j int) bool { return r.matches[i].Score < r.matches[j].Score }

func (r resultHeap) Swap(i, j int) { r.matches[i], r.matches[j] = r.matches[j], r.matches[i] }

func (r *resultHeap) Push(x interface{}) {
	r.matches = append(r.matches, x.(Match))
}

func (r *resultHeap) Pop() interface{} {
	old := r.matches
	n := len(old)
	x := old[n-1]
	r.matches = old[0 : n-1]
	return x
}
