# Natural Language Processing [![GoDoc](https://godoc.org/github.com/james-bowman/nlp?status.svg)](https://godoc.org/github.com/james-bowman/nlp)

<img src="https://github.com/james-bowman/nlp/raw/master/Gophers.008.crop.png" alt="nlp" align="left" hspace="20" vspace="0" />

An implementation of selected machine learning algorithms for basic natural language processing in golang.  Built upon gonum/mat64.  Some inspiration was taken from Python's SciKit Learn.

Check out [the go documentation page](https://godoc.org/github.com/james-bowman/nlp) for full usage and examples.

<br clear="all"/>

## Features

* Convert plain text strings into numerical feature vectors for analysis
* Term document matrix construction and manipulation
* (LSA) Latent Semantic Analysis (aka Latent Semantic Indexing (LSI)) implementation
* TF-IDF weighting to account for frequently occuring words
* SVD (Singular Value Decomposition) implementation for reduced memory usage, noise reduction amd encoding term co-occurance and semantic meaning.

## Planned

* Stop word removal to remove frequently occuring words e.g. "the", "and"
* Stemming to treat words with common root as the same e.g. "go" and "going"
* Feature hashing implementation ('the hashing trick's) for reduced reliance on "completeness" of training data set
* Querying based on cosine similarity of centroid of query sample
* Sparse matrix implementation for more effective memory usage
* Clustering algorithms e.g. K-means
* Classification algorithms e.g. SVM, random forest, etc.

