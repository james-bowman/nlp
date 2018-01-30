package nlp

import (
	"math/rand"

	"github.com/james-bowman/sparse"
	"gonum.org/v1/gonum/mat"
)

// SimHash implements the SimHash Locality Sensitive Hashing (LSH) algorithm
// using sign random projections (Moses S. Charikar, https://www.cs.princeton.edu/courses/archive/spr04/cos598B/bib/CharikarEstim.pdf)
// The distance between the original vectors is preserved through the hashing process such that
// hashed vectors can be compared using Hamming Similarity for a faster, more space
// efficient, approximation of Cosine Similarity for the original vectors.
type SimHash struct {
	projections []*mat.VecDense
}

// NewSimHash constructs a new SimHash creating a set of locality sensitive
// hash functions which are combined to accept input vectors of length dim
// and produce hashed binary vector fingerprints of length bits.  This method
// projects a series of random hyperplanes which are then compared to each
// input vector to produce the output hashed binary vector encoding the input
// vector's location in vector space relative to the hyperplanes.  Each bit in
// the output vector corresponds to the sign (1/0 for +/-) of the result of
// the dot product comparison with each random hyperplane.
func NewSimHash(bits int, dim int) *SimHash {
	// Generate random hyperplane projections
	projections := make([]*mat.VecDense, bits)

	for j := 0; j < bits; j++ {
		p := make([]float64, dim)
		for i := 0; i < dim; i++ {
			p[i] = rand.NormFloat64()
		}
		projections[j] = mat.NewVecDense(dim, p)
	}
	return &SimHash{projections: projections}
}

// Hash accepts a Vector and outputs a BinaryVec (which also implements the
// Gonum Vector interface).  This method will panic if the input vector is of a
// different length than the dim parameter used when constructing the SimHash.
func (h *SimHash) Hash(v mat.Vector) *sparse.BinaryVec {
	bits := len(h.projections)
	dim := h.projections[0].Len()
	if dim != v.Len() {
		panic("The supplied vector has a different number of dimensions from the projected hyperplanes")
	}
	sig := sparse.NewBinaryVec(bits)
	for i := 0; i < bits; i++ {
		if sparse.Dot(v, h.projections[i]) >= 0 {
			sig.SetBit(i)
		}
	}
	return sig
}
