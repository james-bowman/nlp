# Natural Language Processing 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
[![GoDoc](https://godoc.org/github.com/james-bowman/nlp?status.svg)](https://godoc.org/github.com/james-bowman/nlp) 
[![wercker status](https://app.wercker.com/status/33d6c1400cca054635f46a8f44c14c42/s/master "wercker status")](https://app.wercker.com/project/byKey/33d6c1400cca054635f46a8f44c14c42) 
[![Go Report Card](https://goreportcard.com/badge/github.com/james-bowman/nlp)](https://goreportcard.com/report/github.com/james-bowman/nlp) <!--[![Sourcegraph Badge](https://sourcegraph.com/github.com/james-bowman/nlp/-/badge.svg)](https://sourcegraph.com/github.com/james-bowman/nlp?badge)-->

<img src="https://github.com/james-bowman/nlp/raw/master/Gophers.008.crop.png" alt="nlp" align="left" />

An implementation of selected machine learning algorithms for basic natural language processing in golang.  The initial focus for this project is [Latent Semantic Analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis) to allow retrieval/searching, clustering and classification of text documents based upon semantic content.

Built upon [gonum/matrix](https://github.com/gonum/matrix) with some inspiration taken from Python's [scikit-learn](http://scikit-learn.org/stable/).

Check out [the companion blog post](http://www.jamesbowman.me/post/semantic-analysis-of-webpages-with-machine-learning-in-go/) or [the go documentation page](https://godoc.org/github.com/james-bowman/nlp) for full usage and examples.

<br clear="all"/>

## Features

* Sparse matrix implementations for more effective memory usage
* Convert plain text strings into numerical feature vectors for analysis
* Stop word removal to remove frequently occuring English words e.g. "the", "and"
* LSA (Latent Semantic Analysis aka Latent Semantic Indexing (LSI)) implementation using truncated SVD (Singular Value Decomposition) for dimensionality reduction.
* TF-IDF weighting to account for frequently occuring words
* Cosine similarity implementation to calculate the similarity (measured in terms of difference in angles) between feature vectors.

## Planned

* Pipelining of transformations to simplify usage e.g. vectorisation -> tf-idf weighting -> truncated SVD
* Ability to persist trained models
* Feature hashing implementation ('the hashing trick') for improved performance and reduced reliance on training data
* LDA (Latent Dirichlet Allocation) implementation for topic extraction
* Stemming to treat words with common root as the same e.g. "go" and "going"
* Querying based on multiple query strings (using their centroid) rather than just a single query string.
* Support partitioning for the Latent Semantic Index (LSI)
* Clustering algorithms e.g. Heirachical, K-means, etc.
* Classification algorithms e.g. SVM, random forest, etc.

## References

1. [Wikipedia](https://en.wikipedia.org/wiki/Latent_semantic_analysis)
1. [Rosario, Barbara. Latent Semantic Indexing: An overview. INFOSYS 240 Spring 2000](http://people.ischool.berkeley.edu/~rosario/projects/LSI.pdf)
1. [Latent Semantic Analysis, a scholarpedia article on LSA written by Tom Landauer, one of the creators of LSA.](http://www.scholarpedia.org/article/Latent_semantic_analysis)
1. [Thomo, Alex. Latent Semantic Analysis (Tutorial).](http://webhome.cs.uvic.ca/~thomo/svd.pdf)
1. [Latent Semantic Indexing. Standford NLP Course](http://nlp.stanford.edu/IR-book/html/htmledition/latent-semantic-indexing-1.html)