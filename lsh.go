package nlp

import (
	"fmt"
	"strings"

	radix "github.com/armon/go-radix"
	"github.com/james-bowman/sparse"
)

// lshTableBucket represents a hash table bucket used for ClassicLSH.  The bucket
// is a slice of IDs relating to items whose hash maps to the bucket.
type lshTableBucket []interface{}

// lshTable is an hash table used for ClassicLSH.  It is simply a map of hashcodes
// to lshTableBuckets
//type lshTable map[uint64]lshTableBucket
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
// items by their locality sensitive hash code based on the work of A. Gionis et al.
// Items that map to the same bucket (their hash codes collide) are similar.  Multiple
// hash tables are used to improve recall where some similar items would otherwise
// hash to separate, neighbouring buckets in only a single table.
//
// A. Gionis, P. Indyk, and R. Motwani, “Similarity Search in High Dimensions via
// Hashing,” VLDB ’99 Proc. 25th Int. Conf. Very Large Data Bases, vol. 99, no. 1,
// pp. 518–529, 1999.
// http://www.cs.princeton.edu/courses/archive/spring13/cos598C/Gionis.pdf%5Cnhttp://portal.acm.org/citation.cfm?id=671516
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
	// TODO: rather than simply chunking up the hash signature into k/l chunks
	// possibly select hash functions (digits) uniformly at random (with replacement?)
	if signature.Len() != l.reqLen {
		panic(fmt.Sprintf("nlp: Specified signature is not the correct length.  Needed %d but received %d", l.reqLen, signature.Len()))
	}
	keys := make([]uint64, l.numHashtables)
	for i := range keys {
		//keys[i] = signature.SliceToUint64(i*l.numHashfunctions, ((i+1)*l.numHashfunctions)-1)
		keys[i] = signature.SliceToUint64(i*l.numHashfunctions, ((i + 1) * l.numHashfunctions))
	}
	return keys
}

// hashKeysForSignature chunks the hash into a number of smaller hash codes (one per
// table) each the length of the configured number of hash functions per table.
// The method panics if the signature is not the same length as tables * functions.
// func (l *ClassicLSH) hashKeysForSignature(signature *sparse.BinaryVec) []string {
// 	// TODO: rather than simply chunking up the hash signature into k/l chunks
// 	// possibly select hash functions (digits) uniformly at random (with replacement?)
// 	if signature.Len() != l.reqLen {
// 		panic(fmt.Sprintf("nlp: Specified signature is not the correct length.  Needed %d but received %d", l.reqLen, signature.Len()))
// 	}
// 	keys := make([]string, l.numHashtables)
// 	key := signature.String()
// 	for i := range keys {
// 		keys[i] = key[i*l.numHashfunctions : (i+1)*l.numHashfunctions]
// 	}
// 	return keys
// }

// LSHForest is an implementation of the LSH Forest Locality Sensitive Hashing scheme
// based on the work of M. Bawa et al.
//
// M. Bawa, T. Condie, and P. Ganesan, “LSH forest: self-tuning indexes for
// similarity search,” Proc. 14th Int. Conf. World Wide Web - WWW ’05, p. 651, 2005.
// http://dl.acm.org/citation.cfm?id=1060745.1060840
type LSHForest struct {
	trees            []*radix.Tree
	numHashfunctions int
	reqLen           int
}

// NewLSHForest creates a new LSHForest Locality Sensitive Hashing scheme with the
// specified number of hash tables and hash functions per table.
func NewLSHForest(functions int, tables int) *LSHForest {
	trees := make([]*radix.Tree, tables)
	for i := range trees {
		trees[i] = radix.New()
	}
	return &LSHForest{
		trees:            trees,
		numHashfunctions: functions,
		reqLen:           functions * tables,
	}
}

// Put stores the specified LSH signature and associated ID in the LSH index
func (l *LSHForest) Put(id interface{}, signature *sparse.BinaryVec) {
	keys := l.hashKeysForSignature(signature)
	for i, tree := range l.trees {
		//bucket, _ := tree.Get(keys[i])
		bucket, ok := tree.Get(keys[i])
		if !ok {
			bucket = make([]interface{}, 0)
		}
		tree.Insert(keys[i], append(bucket.([]interface{}), id))
	}
}

// GetCandidates returns the IDs of candidate nearest neighbours.  It is up to
// the calling code to further filter these candidates based on distance to arrive
// at the top-k approximate nearest neighbours.  The number of candidates returned
// may be smaller or larger than k.
func (l *LSHForest) GetCandidates(query *sparse.BinaryVec, k int) []interface{} {
	keys := l.hashKeysForSignature(query)

	m := k
	seen := make(map[interface{}]struct{})

	for i, tree := range l.trees {
		if bucketEntries, exist := tree.Get(keys[i]); exist {
			for _, id := range bucketEntries.([]interface{}) {
				seen[id] = struct{}{}
			}
		}
	}

	// if we have not found enough candidates then walk back up the trees for
	// similar items in neighbouring buckets with shared prefixes
	x := l.numHashfunctions
	for len(seen) < m && x > 0 {
		for i, tree := range l.trees {
			var k string
			if keys[i][x-1] == '1' {
				k = "0"
			} else {
				k = "1"
			}

			altKey := strings.Join([]string{keys[i][0 : x-1], k}, "")
			tree.WalkPrefix(altKey, func(s string, v interface{}) bool {
				for _, id := range v.([]interface{}) {
					seen[id] = struct{}{}
				}
				return false
			})
		}
		x--
	}

	// Collect results
	candidates := make([]interface{}, len(seen))
	var i int
	for index := range seen {
		candidates[i] = index
		i++
	}

	return candidates
}

// Remove removes the specified item from the LSH index
func (l *LSHForest) Remove(id interface{}) {
	for _, tree := range l.trees {
		tree.Walk(func(s string, v interface{}) bool {
			bucketContents := v.([]interface{})
			for i, indexedID := range bucketContents {
				if id == indexedID {
					bucketContents[i] = bucketContents[len(bucketContents)-1]
					bucketContents = bucketContents[:len(bucketContents)-1]
					if len(bucketContents) == 0 {
						tree.Delete(s)
					} else {
						tree.Insert(s, bucketContents)
					}
					return true
				}
			}
			return false
		})
	}
}

// hashKeysForSignature chunks the hash into a number of smaller hash codes (one per
// table) each the length of the configured number of hash functions per table.
// The method panics if the signature is not the same length as tables * functions.
func (l *LSHForest) hashKeysForSignature(signature *sparse.BinaryVec) []string {
	// TODO: rather than simply chunking up the hash signature into k/l chunks
	// possibly select hash functions (digits) uniformly at random (with replacement?)
	if signature.Len() != l.reqLen {
		panic(fmt.Sprintf("nlp: Specified signature is not the correct length.  Needed %d but received %d", l.reqLen, signature.Len()))
	}
	keys := make([]string, len(l.trees))
	key := signature.String()
	for i := range keys {
		keys[i] = key[i*l.numHashfunctions : (i+1)*l.numHashfunctions]
	}
	return keys
}
