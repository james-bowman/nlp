package nlp

import (
	"sort"
	"testing"

	"github.com/james-bowman/nlp/measures/pairwise"
	"github.com/james-bowman/sparse"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func TestIndexerIndex(t *testing.T) {
	m := sparse.Random(sparse.DenseFormat, 100, 10, 1.0)

	tests := []struct {
		index Indexer
	}{
		{index: NewLinearScanIndex(pairwise.CosineDistance)},
		{index: NewLSHIndex(false, NewSimHash(1000, 100), NewClassicLSH(50, 20), pairwise.CosineDistance)},
		{index: NewLSHIndex(true, NewSimHash(1000, 100), NewClassicLSH(50, 20), pairwise.HammingDistance)},
	}

	for ti, test := range tests {
		ColDo(m, func(j int, v mat.Vector) {
			test.index.Index(v, j)
		})

		ColDo(m, func(j int, v mat.Vector) {
			matches := test.index.Search(v, 1)

			if len(matches) != 1 {
				t.Errorf("Test %d: Search expected 1 result but received %d", ti+1, len(matches))
			}
			if matches[0].ID != j {
				t.Errorf("Test %d: Search expected to find %d but found %d", ti+1, j, matches[0].ID)
			}
			if matches[0].Distance < -0.0000001 || matches[0].Distance > 0.0000001 {
				t.Errorf("Test %d: Search match distance expected 0.0 but received %f", ti+1, matches[0].Distance)
			}
		})
	}
}

func TestIndexerSearch(t *testing.T) {
	numCols := 10
	m := sparse.Random(sparse.DenseFormat, 100, numCols, 1.0)

	// build similarity matrix
	similarityMatrix := make([]float64, numCols*numCols)
	inds := make([][]int, numCols)
	ColDo(m, func(j int, v1 mat.Vector) {
		ColDo(m, func(i int, v2 mat.Vector) {
			similarityMatrix[j*numCols+i] = pairwise.CosineDistance(v1, v2)
		})
		inds[j] = make([]int, numCols)
		floats.Argsort(similarityMatrix[j*numCols:(j+1)*numCols], inds[j])
		for left, right := 0, len(inds[j])-1; left < right; left, right = left+1, right-1 {
			inds[j][left], inds[j][right] = inds[j][right], inds[j][left]
			similarityMatrix[j*numCols+left], similarityMatrix[j*numCols+right] = similarityMatrix[j*numCols+right], similarityMatrix[j*numCols+left]
		}
	})

	tests := []struct {
		k     int
		index Indexer
	}{
		{k: numCols, index: NewLinearScanIndex(pairwise.CosineDistance)},
		{k: numCols, index: NewLSHIndex(false, NewSimHash(800, 100), NewClassicLSH(8, 100), pairwise.CosineDistance)},
		//{k: numCols, index: NewLSHIndex(true, NewSimHash(5000, 100), NewClassicLSH(10, 500), pairwise.HammingDistance)},
	}

	for ti, test := range tests {
		ColDo(m, func(j int, v mat.Vector) {
			test.index.Index(v, j)
		})

		ColDo(m, func(j int, v mat.Vector) {

			matches := test.index.Search(v, test.k)

			if len(matches) != test.k {
				t.Errorf("Test %d: Search expected %d result but received %d", ti+1, test.k, len(matches))
			}
			heap := resultHeap{matches: matches}
			sort.Sort(heap)

			for i, match := range matches {
				if match.ID != inds[j][i] {
					t.Errorf("Test %d: For col #%d, Rank #%d - expected %v but found %v", ti+1, j, i, inds[j], matches)
					return
				}
			}
		})
	}
}

func TestIndexerRemove(t *testing.T) {
	m := sparse.Random(sparse.DenseFormat, 100, 10, 1.0)

	tests := []struct {
		index Indexer
	}{
		{index: NewLinearScanIndex(pairwise.CosineDistance)},
		{index: NewLSHIndex(false, NewSimHash(1000, 100), NewClassicLSH(50, 20), pairwise.CosineDistance)},
		{index: NewLSHIndex(true, NewSimHash(1000, 100), NewClassicLSH(50, 20), pairwise.HammingDistance)},
	}

	for ti, test := range tests {
		ColDo(m, func(j int, v mat.Vector) {
			test.index.Index(v, j)
		})

		ColDo(m, func(j int, v mat.Vector) {
			test.index.Remove(j)
			matches := test.index.Search(v, 1)

			if len(matches) > 1 {
				t.Errorf("Test %d: Search expected less than 1 result but received %d", ti+1, len(matches))
			}
			if len(matches) == 1 {
				if matches[0].ID == j {
					t.Errorf("Test %d: Search expected not to find %d but found %d", ti+1, j, matches[0].ID)
				}
			}
		})
	}
}
