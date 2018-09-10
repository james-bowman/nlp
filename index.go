package nlp

import (
	"container/heap"
	"fmt"
	"sync"

	"github.com/james-bowman/nlp/measures/pairwise"
	"github.com/james-bowman/sparse"
	"gonum.org/v1/gonum/mat"
)

// Match represents a matching item for nearest neighbour similarity searches.
// It contains both the ID of the matching item and the distance from the queried item.
// The distance is represented as a score from 0 (exact match) to 1 (orthogonal)
// depending upon the metric used.
type Match struct {
	Distance float64
	ID       interface{}
}

// resultHeap is a min heap (priority queue) used to compile the top-k matches whilst
// performing nearest neighbour similarity searches.
type resultHeap struct {
	matches []Match
}

func (r resultHeap) Len() int { return len(r.matches) }

func (r resultHeap) Less(i, j int) bool { return r.matches[i].Distance > r.matches[j].Distance }

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

// Indexer indexes vectors to support Nearest Neighbour (NN) similarity searches across
// the indexed vectors.
type Indexer interface {
	Index(v mat.Vector, id interface{})
	Search(q mat.Vector, k int) []Match
	Remove(ids interface{})
}

// LinearScanIndex supports Nearest Neighbour (NN) similarity searches across indexed
// vectors performing queries in O(n) and requiring O(n) storage.  As the name implies,
// LinearScanIndex performs a linear scan across all indexed vectors comparing them
// each in turn with the specified query vector using the configured pairwise distance
// metric.  LinearScanIndex is accurate and will always return the true top-k nearest
// neighbours as opposed to some other types of index, like LSHIndex,
// which perform Approximate Nearest Neighbour (ANN) searches and trade some recall
// accuracy for performance over large scale datasets.
type LinearScanIndex struct {
	lock       sync.RWMutex
	signatures []mat.Vector
	ids        []interface{}
	distance   pairwise.Comparer
}

// NewLinearScanIndex construct a new empty LinearScanIndex which will use the specified
// pairwise distance metric to determine nearest neighbours based on similarity.
func NewLinearScanIndex(compareFN pairwise.Comparer) *LinearScanIndex {
	return &LinearScanIndex{distance: compareFN}
}

// Index adds the specified vector v with associated id to the index.
func (b *LinearScanIndex) Index(v mat.Vector, id interface{}) {
	b.lock.Lock()
	b.signatures = append(b.signatures, v)
	b.ids = append(b.ids, id)
	b.lock.Unlock()
}

// Search searches for the top-k nearest neighbours in the index.  The method
// returns up to the top-k most similar items in unsorted order.  The method may
// return fewer than k items if less than k neighbours are found.
func (b *LinearScanIndex) Search(qv mat.Vector, k int) []Match {
	b.lock.RLock()
	defer b.lock.RUnlock()
	size := len(b.signatures)

	var point int
	var results resultHeap
	results.matches = make([]Match, 0, k)
	for point = 0; point < k && point < size; point++ {
		mv := b.signatures[point]
		match := Match{Distance: b.distance(qv, mv), ID: b.ids[point]}
		results.matches = append(results.matches, match)
	}
	if len(results.matches) < k {
		return results.matches
	}
	heap.Init(&results)
	var dist float64
	for i := point; i < size; i++ {
		mv := b.signatures[i]
		dist = b.distance(qv, mv)
		if dist <= results.matches[0].Distance {
			heap.Pop(&results)
			heap.Push(&results, Match{Distance: dist, ID: b.ids[i]})
		}
	}

	return results.matches
}

// Remove removes the vector with the specified id from the index.  If no vector
// is found with the specified id the method will simply do nothing.
func (b *LinearScanIndex) Remove(id interface{}) {
	b.lock.Lock()
	defer b.lock.Unlock()

	for i, v := range b.ids {
		if v == id {
			copy(b.signatures[i:], b.signatures[i+1:])
			b.signatures[len(b.signatures)-1] = nil
			b.signatures = b.signatures[:len(b.signatures)-1]

			copy(b.ids[i:], b.ids[i+1:])
			b.ids[len(b.ids)-1] = nil
			b.ids = b.ids[:len(b.ids)-1]

			return
		}
	}
}

// Hasher interface represents a Locality Sensitive Hashing algorithm whereby
// the proximity of data points is preserved in the hash space i.e. similar data
// points will be hashed to values close together in the hash space.
type Hasher interface {
	// Hash hashes the input vector into a BinaryVector hash representation
	Hash(mat.Vector) *sparse.BinaryVec
}

// LSHScheme interface represents LSH indexing schemes to support Approximate Nearest
// Neighbour (ANN) search.
type LSHScheme interface {
	// Put stores the specified LSH signature and associated ID in the LSH index
	Put(id interface{}, signature *sparse.BinaryVec)

	// GetCandidates returns the IDs of candidate nearest neighbours.  It is up to
	// the calling code to further filter these candidates based on distance to arrive
	// at the top-k approximate nearest neighbours.  The number of candidates returned
	// may be smaller or larger than k.
	GetCandidates(query *sparse.BinaryVec, k int) []interface{}

	// Remove removes the specified item from the LSH index
	Remove(id interface{})
}

// LSHIndex is an LSH (Locality Sensitive Hashing) based index supporting Approximate
// Nearest Neighbour (ANN) search in O(log n).  The storage required by the index will
// depend upon the underlying LSH index store algorithm used but will typically be
// higher than O(n).  In use cases where accurate Nearest Neighbour search is required
// other types of index should be considered like LinearScanIndex.
type LSHIndex struct {
	lock       sync.RWMutex
	IsApprox   bool
	hasher     Hasher
	scheme     LSHScheme
	signatures map[interface{}]mat.Vector
	distance   pairwise.Comparer
}

// NewLSHIndex creates a new LSHIndex.  When queried, the initial candidate
// nearest neighbours returned by the underlying LSH indexing algorithm
// are further filtered by comparing distances to the query vector using the supplied
// distance metric.  If approx is true, the filtering comparison is performed on the
// hashes and if approx is false, then the comparison is performed on the original
// vectors instead. This will have time and storage implications as comparing the
// original vectors will be more accurate but slower and require the original vectors
// be stored for the comparison.  The LSH algorithm and underlying LSH indexing
// algorithm may both be specified as hasher and store parameters respectively.
func NewLSHIndex(approx bool, hasher Hasher, store LSHScheme, distance pairwise.Comparer) *LSHIndex {
	index := LSHIndex{
		IsApprox:   approx,
		hasher:     hasher,
		scheme:     store,
		signatures: make(map[interface{}]mat.Vector),
		distance:   distance,
	}

	return &index
}

// Index indexes the supplied vector along with its associated ID.
func (l *LSHIndex) Index(v mat.Vector, id interface{}) {
	h := l.hasher.Hash(v)

	l.lock.Lock()
	defer l.lock.Unlock()

	l.scheme.Put(id, h)
	if l.IsApprox {
		l.signatures[id] = h
	} else {
		l.signatures[id] = v
	}
}

// Search searches for the top-k approximate nearest neighbours in the index.  The
// method returns up to the top-k most similar items in unsorted order.  The method may
// return fewer than k items if less than k neighbours are found.
func (l *LSHIndex) Search(q mat.Vector, k int) []Match {
	hv := l.hasher.Hash(q)

	l.lock.RLock()
	defer l.lock.RUnlock()

	candidateIDs := l.scheme.GetCandidates(hv, k)

	var qv mat.Vector
	if l.IsApprox {
		qv = hv
	} else {
		qv = q
	}

	size := len(candidateIDs)
	var results resultHeap
	var point int
	for point = 0; point < k && point < size; point++ {
		mv := l.signatures[candidateIDs[point]]
		match := Match{Distance: l.distance(qv, mv), ID: candidateIDs[point]}
		results.matches = append(results.matches, match)
	}
	if len(results.matches) < k {
		return results.matches
	}
	heap.Init(&results)
	var dist float64
	for i := point; i < size; i++ {
		mv := l.signatures[candidateIDs[i]]
		dist = l.distance(qv, mv)
		if dist <= results.matches[0].Distance {
			heap.Pop(&results)
			heap.Push(&results, Match{Distance: dist, ID: candidateIDs[i]})
		}
	}

	return results.matches
}

// Remove removes the vector with the specified id from the index.  If no vector
// is found with the specified id the method will simply do nothing.
func (l *LSHIndex) Remove(id interface{}) {
	l.lock.Lock()
	defer l.lock.Unlock()

	delete(l.signatures, id)
	l.scheme.Remove(id)
}

// lshTableBucket represents a hash table bucket used for ClassicLSH.  The bucket
// is a slice of IDs relating to items whose hash maps to the bucket.
type lshTableBucket []interface{}

// lshTable is an hash table used for ClassicLSH.  It is simply a map of hashcodes
// to lshTableBuckets
type lshTable map[uint64]lshTableBucket

// remove removes the specified item from the LSH table
func (t lshTable) remove(id interface{}) {
	for key, bucketContents := range t {
		for j, indexedID := range bucketContents {
			if id == indexedID {
				bucketContents[j] = bucketContents[len(bucketContents)-1]
				t[key] = bucketContents[:len(bucketContents)-1]
				if len(t[key]) == 0 {
					delete(t, key)
				}
				return
			}
		}
	}
}

// ClassicLSH supports finding top-k Approximate Nearest Neighbours (ANN) using Locality
// Sensitive Hashing (LSH).  Classic LSH scheme is based on using hash tables to store
// items by their locality sensitive hash code.  Items that map to the same bucket
// (their hash codes collide) are similar.  Multiple hash tables are used to
// improve recall where some similar items would otherwise hash to separate,
// neighbouring buckets in only a single table.
type ClassicLSH struct {
	numHashtables    int
	numHashfunctions int
	reqLen           int
	hashTables       []lshTable
}

// NewClassicLSH creates a new ClassicLSH with the configured number of hash tables
// and hash functions per table.  The length of hash signatures used in this type's
// methods (Put() and GetCandidates()) should be exactly equal to functions * tables.
// The Classic LSH algorithm uses multiple hash tables to improve recall for similar
// items that hash to nearby buckets within a specific hash table.
func NewClassicLSH(functions, tables int) *ClassicLSH {
	hashtables := make([]lshTable, tables)
	for i := range hashtables {
		hashtables[i] = make(map[uint64]lshTableBucket)
	}

	return &ClassicLSH{
		reqLen:           tables * functions,
		numHashtables:    tables,
		numHashfunctions: functions,
		hashTables:       hashtables,
	}
}

// Put stores the specified LSH signature and associated ID in the LSH index.
// The method panics if the signature is not the same length as tables * functions.
func (l *ClassicLSH) Put(id interface{}, signature *sparse.BinaryVec) {
	keys := l.hashKeysForSignature(signature)
	for i := range l.hashTables {
		l.hashTables[i][keys[i]] = append(l.hashTables[i][keys[i]], id)
	}
}

// GetCandidates returns the IDs of candidate nearest neighbours.  It is up to
// the calling code to further filter these candidates based on distance to arrive
// at the top-k approximate nearest neighbours.  The number of candidates returned
// may be smaller or larger than k.  The method panics if the signature is not the
// same length as tables * functions.
func (l *ClassicLSH) GetCandidates(query *sparse.BinaryVec, k int) []interface{} {
	keys := l.hashKeysForSignature(query)

	seen := make(map[interface{}]struct{})
	for i, table := range l.hashTables {
		if bucketEntries, exist := table[keys[i]]; exist {
			for _, id := range bucketEntries {
				if _, exist := seen[id]; exist {
					continue
				}
				seen[id] = struct{}{}
			}
		}
	}

	// Collect results
	ids := make([]interface{}, len(seen))
	var i int
	for index := range seen {
		ids[i] = index
		i++
	}
	return ids
}

// Remove removes the specified item from the LSH index
func (l *ClassicLSH) Remove(id interface{}) {
	for _, table := range l.hashTables {
		table.remove(id)
	}
}

// hashKeysForSignature chunks the hash into a number of smaller hash codes (one per
// table) each the length of the configured number of hash functions per table.
// The method panics if the signature is not the same length as tables * functions.
func (l *ClassicLSH) hashKeysForSignature(signature *sparse.BinaryVec) []uint64 {
	if signature.Len() != l.reqLen {
		panic(fmt.Sprintf("nlp: Specified signature is not the correct length.  Needed %d but received %d", signature.Len(), l.reqLen))
	}
	keys := make([]uint64, l.numHashtables)
	for i := 0; i < l.numHashtables; i++ {
		keys[i] = signature.SliceToUint64(i*l.numHashfunctions, ((i+1)*l.numHashfunctions)-1)
	}
	return keys
}
