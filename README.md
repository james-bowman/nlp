# Natural Language Processing 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
[![GoDoc](https://godoc.org/github.com/james-bowman/nlp?status.svg)](https://godoc.org/github.com/james-bowman/nlp) 
[![Build Status](https://travis-ci.org/james-bowman/nlp.svg?branch=master)](https://travis-ci.org/james-bowman/nlp)
[![Go Report Card](https://goreportcard.com/badge/github.com/james-bowman/nlp)](https://goreportcard.com/report/github.com/james-bowman/nlp) 
[![GoCover](https://gocover.io/_badge/github.com/james-bowman/nlp)](https://gocover.io/github.com/james-bowman/nlp) 
<!--[![Sourcegraph Badge](https://sourcegraph.com/github.com/james-bowman/nlp/-/badge.svg)](https://sourcegraph.com/github.com/james-bowman/nlp?badge)-->

<img src="https://github.com/james-bowman/nlp/raw/master/Gophers.008.crop.png" alt="nlp" align="left" />

An implementation of selected machine learning algorithms for basic natural language processing in golang.  The initial focus for this project is [Latent Semantic Analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis) to allow retrieval/searching, clustering and classification of text documents based upon semantic content.

Built upon the [Gonum library](http://http://www.gonum.org/) for linear algebra and scientific computing with some inspiration taken from Python's [scikit-learn](http://scikit-learn.org/stable/).

Check out [the companion blog post](http://www.jamesbowman.me/post/semantic-analysis-of-webpages-with-machine-learning-in-go/) or [the go documentation page](https://godoc.org/github.com/james-bowman/nlp) for full usage and examples.

<br clear="all"/>

## Features

* [Sparse matrix](http://github.com/james-bowman/sparse) implementations for more effective memory usage
* Convert plain text strings into numerical feature vectors for analysis
* Stop word removal to remove frequently occuring English words e.g. "the", "and"
* [Feature hashing](https://en.wikipedia.org/wiki/Feature_hashing)('the hashing trick') implementation (using [MurmurHash3](http://github.com/spaolacci/murmur3)) for reduced memory requirements and reduced reliance on training data
* [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) weighting to account for frequently occuring words
* [LSA (Latent Semantic Analysis aka Latent Semantic Indexing (LSI))](https://en.wikipedia.org/wiki/Latent_semantic_analysis) implementation using truncated [SVD (Singular Value Decomposition)](https://en.wikipedia.org/wiki/Singular-value_decomposition) for dimensionality reduction.
* [PCA (Principal Component Analysis)](https://en.wikipedia.org/wiki/Principal_component_analysis)
* [SimHash](https://en.wikipedia.org/wiki/SimHash) implementation of [LSH (Locality Sensitive Hashing)](https://en.wikipedia.org/wiki/Locality-sensitive_hashing) using [sign random projection](https://en.wikipedia.org/wiki/Locality-sensitive_hashing#Random_projection) to support approximate cosine similarity using significantly less memory and processing time.
* [Random Indexing (RI)](https://en.wikipedia.org/wiki/Random_indexing) and Reflective Random Indexing (RRI) (which extends RI to support indirect inference) for scalable [Latent Semantic Analysis (LSA)]((https://en.wikipedia.org/wiki/Latent_semantic_analysis) with semantic vector space models.
* Cosine, Angular and Hamming similarity/distance measures to calculate the similarity/distance between feature vectors.
* Persistence for trained models (persistence for Vectorisers coming soon)

## Planned

* Ability to persist trained vectorisers
* [LDA (Latent Dirichlet Allocation)](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) implementation for topic extraction
* Stemming to treat words with common root as the same e.g. "go" and "going"
* Clustering algorithms e.g. Heirachical, K-means, etc.
* Classification algorithms e.g. SVM, random forest, etc.

## References

1. [Rosario, Barbara. Latent Semantic Indexing: An overview. INFOSYS 240 Spring 2000](http://people.ischool.berkeley.edu/~rosario/projects/LSI.pdf)
1. [Latent Semantic Analysis, a scholarpedia article on LSA written by Tom Landauer, one of the creators of LSA.](http://www.scholarpedia.org/article/Latent_semantic_analysis)
1. [Thomo, Alex. Latent Semantic Analysis (Tutorial).](http://webhome.cs.uvic.ca/~thomo/svd.pdf)
1. [Latent Semantic Indexing. Standford NLP Course](http://nlp.stanford.edu/IR-book/html/htmledition/latent-semantic-indexing-1.html)
1. [Charikar, Moses S. Similarity Estimation Techniques from Rounding Algorithms](https://www.cs.princeton.edu/courses/archive/spr04/cos598B/bib/CharikarEstim.pdf)
1. [Kanerva, Pentti, Kristoferson, Jan and Holst, Anders (2000). Random Indexing of Text Samples for Latent Semantic Analysis](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.4.6523&rep=rep1&type=pdf)
1. [Rangan, Venkat. Discovery of Related Terms in a corpus using Reflective Random Indexing](https://www.umiacs.umd.edu/~oard/desi4/papers/rangan.pdf)
1. [Vasuki, Vidya and Cohen, Trevor. Reflective random indexing for semi-automatic indexing of the biomedical literature](https://ac.els-cdn.com/S1532046410000481/1-s2.0-S1532046410000481-main.pdf?_tid=f31f92e8-028a-11e8-8c31-00000aab0f6c&acdnat=1516965824_e24a804445fff1744281ca6f5898a3a4)