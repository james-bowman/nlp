# Natural Language Processing 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
[![GoDoc](https://godoc.org/github.com/james-bowman/nlp?status.svg)](https://godoc.org/github.com/james-bowman/nlp) 
[![Build Status](https://travis-ci.org/james-bowman/nlp.svg?branch=master)](https://travis-ci.org/james-bowman/nlp)
[![Go Report Card](https://goreportcard.com/badge/github.com/james-bowman/nlp)](https://goreportcard.com/report/github.com/james-bowman/nlp) 
[![GoCover](https://gocover.io/_badge/github.com/james-bowman/nlp)](https://gocover.io/github.com/james-bowman/nlp) 
<!--[![Sourcegraph Badge](https://sourcegraph.com/github.com/james-bowman/nlp/-/badge.svg)](https://sourcegraph.com/github.com/james-bowman/nlp?badge)-->

<img src="https://github.com/james-bowman/nlp/raw/master/Gophers.008.crop.png" alt="nlp" align="left" />

An implementation of selected machine learning algorithms for basic natural language processing in golang.  The initial focus for this project is [Latent Semantic Analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis) to allow retrieval/searching, clustering and classification of text documents based upon semantic content.

Built upon the [gonum/gonum matrix library](https://github.com/gonum/gonum) with some inspiration taken from Python's [scikit-learn](http://scikit-learn.org/stable/).

Check out [the companion blog post](http://www.jamesbowman.me/post/semantic-analysis-of-webpages-with-machine-learning-in-go/) or [the go documentation page](https://godoc.org/github.com/james-bowman/nlp) for full usage and examples.

<br clear="all"/>

## Features

* [Sparse matrix](http://github.com/james-bowman/sparse) implementations for more effective memory usage
* Convert plain text strings into numerical feature vectors for analysis
* Stop word removal to remove frequently occuring English words e.g. "the", "and"
* Feature hashing implementation ('the hashing trick') (using [MurmurHash3](http://github.com/spaolacci/murmur3)) for reduced memory requirements and reduced reliance on training data
* TF-IDF weighting to account for frequently occuring words
* LSA (Latent Semantic Analysis aka Latent Semantic Indexing (LSI)) implementation using truncated SVD (Singular Value Decomposition) for dimensionality reduction.
* Simhash Locality Sensitive Hashing implementation using sign random projections for dimensionality reduction and efficient information retrieval, enabling approximate cosine similarity using significantly less memory and processing time.
* Cosine, Angular and Hamming similarity/distance measures to calculate the similarity/distance between feature vectors.
* Persistence for trained models (persistence for Vectorisers coming soon)

## Planned

* Ability to persist trained vectorisers
* LDA (Latent Dirichlet Allocation) implementation for topic extraction
* Stemming to treat words with common root as the same e.g. "go" and "going"
* Querying based on multiple query strings (using their centroid) rather than just a single query string.
* Clustering algorithms e.g. Heirachical, K-means, etc.
* Classification algorithms e.g. SVM, random forest, etc.

## References

1. [Wikipedia](https://en.wikipedia.org/wiki/Latent_semantic_analysis)
1. [Rosario, Barbara. Latent Semantic Indexing: An overview. INFOSYS 240 Spring 2000](http://people.ischool.berkeley.edu/~rosario/projects/LSI.pdf)
1. [Latent Semantic Analysis, a scholarpedia article on LSA written by Tom Landauer, one of the creators of LSA.](http://www.scholarpedia.org/article/Latent_semantic_analysis)
1. [Thomo, Alex. Latent Semantic Analysis (Tutorial).](http://webhome.cs.uvic.ca/~thomo/svd.pdf)
1. [Latent Semantic Indexing. Standford NLP Course](http://nlp.stanford.edu/IR-book/html/htmledition/latent-semantic-indexing-1.html)