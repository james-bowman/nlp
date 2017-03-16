/*
Package nlp provides implementations of selected machine learning algorithms for natural language processing of text corpora.  The initial primary focus being on the implementation of algorithms supporting LSA (Latent Semantic Analysis), often referred to as Latent Semantic Indexing in the context of information retrieval.

Overview

The algorithms in the package typically support document input as text strings which are then encoded as a matrix of numerical feature vectors called a `term document matrix`.  Columns in this matrix represent the documents in the corpus and the rows represent terms occuring in the documents.  The individual elements within the matrix contains counts of the number of occurances of each term in the associated document.

This matrix can be manipulated through the application of additional transformations for the purpose of weighting features to remove natural biases which would skew results e.g. commonly occuring words like `the`, `of`, `and`, etc. should carry lower weight than unusual words.  Finally, the matrix is decomposed and truncated using an algorithm called SVD (`Singular Value Decomposition`) which approximates the original term document matrix with a new matrix of much lower rank (typically around 100 rather than 1000s).  This is a fundamental part of LSA and serves a number of purposes:

1. The reduced dimensionality of the data theoretically requires less memory.

2. As less significant dimensions are removed, there is less `noise` in the data which could have artificially skewed results.

3. Perhaps most importantly, the SVD effectively encodes the co-occurance of terms within the documents to capture semantic meaning rather than simply the presence (or lack of presence) of words.  This combats the problem of synonymy (a common challenge in NLP) where different words in the English language can be used to mean the same thing (synonyms).  In LSA, documents can have a high degree of semantic similarity with very few words in common.

The post SVD matrix (with each column being a feature vector representing a document within the corpus) can be compared for similarity with each other (for clustering) or with a query (also represented as a feature vector projected into the same dimensional space).  Similarity is measured by the angle between the two feature vectors being considered.
*/
package nlp
