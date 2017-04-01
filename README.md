# Natural Language Processing [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GoDoc](https://godoc.org/github.com/james-bowman/nlp?status.svg)](https://godoc.org/github.com/james-bowman/nlp) [![wercker status](https://app.wercker.com/status/33d6c1400cca054635f46a8f44c14c42/s/master "wercker status")](https://app.wercker.com/project/byKey/33d6c1400cca054635f46a8f44c14c42) 
<img src="https://github.com/james-bowman/nlp/raw/master/Gophers.008.crop.png" alt="nlp" align="left" />

An implementation of selected machine learning algorithms for basic natural language processing in golang.  The initial focus for this project is [Latent Semantic Analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis) to allow retrieval/searching, clustering and classification of text documents based upon semantic content.

Built upon [gonum/matrix](https://github.com/gonum/matrix) with some inspiration taken from Python's [scikit-learn](http://scikit-learn.org/stable/).

Check out [the companion blog post](http://www.jamesbowman.me/post/semantic-analysis-of-webpages-with-machine-learning-in-go/) or [the go documentation page](https://godoc.org/github.com/james-bowman/nlp) for full usage and examples.

<br clear="all"/>

## Features

* Convert plain text strings into numerical feature vectors for analysis
* Stop word removal to remove frequently occuring English words e.g. "the", "and"
* Term document matrix construction and manipulation
* LSA (Latent Semantic Analysis aka Latent Semantic Indexing (LSI)) implementation
* TF-IDF weighting to account for frequently occuring words
* Truncated SVD (Singular Value Decomposition) implementation for reduced memory usage, noise reduction and encoding term co-occurance and semantic meaning.
* Cosine similarity implementation to calculate the similarity (measured in terms of difference in angles) between 2 feature vectors.

## Planned

* Pipelining of transformations to simplify usage e.g. vectorisation -> tf-idf weighting -> truncated SVD
* Stemming to treat words with common root as the same e.g. "go" and "going"
* Feature hashing implementation ('the hashing trick') for reduced reliance on "completeness" of training dataset
* Querying based on centroid of queries rather than just a single query.
* Sparse matrix implementation for more effective memory usage
* LDA (Latent Dirichlet Allocation) implementation for topic extraction
* Clustering algorithms e.g. K-means
* Classification algorithms e.g. SVM, random forest, etc.

## References

1. [Wikipedia](https://en.wikipedia.org/wiki/Latent_semantic_analysis)
1. [Rosario, Barbara. Latent Semantic Indexing: An overview. INFOSYS 240 Spring 2000](http://people.ischool.berkeley.edu/~rosario/projects/LSI.pdf)
1. [Latent Semantic Analysis, a scholarpedia article on LSA written by Tom Landauer, one of the creators of LSA.](http://www.scholarpedia.org/article/Latent_semantic_analysis)
1. [Thomo, Alex. Latent Semantic Analysis (Tutorial).](http://webhome.cs.uvic.ca/~thomo/svd.pdf)
1. [Latent Semantic Indexing. Standford NLP Course](http://nlp.stanford.edu/IR-book/html/htmledition/latent-semantic-indexing-1.html)