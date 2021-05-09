# Natural Language Processing 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
[![GoDoc](https://godoc.org/github.com/james-bowman/nlp?status.svg)](https://godoc.org/github.com/james-bowman/nlp) 
[![Build Status](https://travis-ci.org/james-bowman/nlp.svg?branch=master)](https://travis-ci.org/james-bowman/nlp)
[![Go Report Card](https://goreportcard.com/badge/github.com/james-bowman/nlp)](https://goreportcard.com/report/github.com/james-bowman/nlp)
[![codecov](https://codecov.io/gh/james-bowman/nlp/branch/master/graph/badge.svg)](https://codecov.io/gh/james-bowman/nlp)
[![Mentioned in Awesome Go](https://awesome.re/mentioned-badge-flat.svg)](https://github.com/avelino/awesome-go)
[![Sourcegraph](https://sourcegraph.com/github.com/james-bowman/nlp/-/badge.svg)](https://sourcegraph.com/github.com/james-bowman/nlp?badge)


<img src="https://github.com/james-bowman/nlp/raw/master/Gophers.008.crop.png" alt="nlp" align="left" />

Implementations of selected machine learning algorithms for natural language processing in golang.  The primary focus for the package is the statistical semantics of plain-text documents supporting semantic analysis and retrieval of semantically similar documents.

Built upon the [Gonum](https://www.gonum.org/) package for linear algebra and scientific computing with some inspiration taken from Python's [scikit-learn](http://scikit-learn.org/stable/) and [Gensim](https://radimrehurek.com/gensim/).

Check out [the companion blog post](http://www.jamesbowman.me/post/semantic-analysis-of-webpages-with-machine-learning-in-go/) or [the Go documentation page](https://godoc.org/github.com/james-bowman/nlp) for full usage and examples.

<br clear="all"/>

## Features

* [LSA (Latent Semantic Analysis aka Latent Semantic Indexing (LSI))][LSA] implementation using truncated [SVD (Singular Value Decomposition)](https://en.wikipedia.org/wiki/Singular-value_decomposition) for dimensionality reduction.
* Fast comparison and retrieval of semantically similar documents using [SimHash](https://en.wikipedia.org/wiki/SimHash)(random hyperplanes/[sign random projection](https://en.wikipedia.org/wiki/Locality-sensitive_hashing#Random_projection)) algorithm with multi-index and Forest schemes for [LSH (Locality Sensitive Hashing)](https://en.wikipedia.org/wiki/Locality-sensitive_hashing) to support fast, approximate cosine similarity/angular distance comparisons and approximate nearest neighbour search using significantly less memory and processing time.
* [Random Indexing (RI)](https://en.wikipedia.org/wiki/Random_indexing) and Reflective Random Indexing (RRI) (which extends RI to support indirect inference) for scalable [Latent Semantic Analysis (LSA)][LSA] over large, web-scale corpora.
* [Latent Dirichlet Allocation (LDA)](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) using a parallelised implementation of the fast [SCVB0 (Stochastic Collapsed Variational Bayesian inference)][SCVB0] algorithm for unsupervised topic extraction. 
* [PCA (Principal Component Analysis)](https://en.wikipedia.org/wiki/Principal_component_analysis)
* [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) weighting to account for frequently occuring words
* [Sparse matrix](http://github.com/james-bowman/sparse) implementations used for more efficient memory usage and processing over large document corpora.
* Stop word removal to remove frequently occuring English words e.g. "the", "and"
* [Feature hashing](https://en.wikipedia.org/wiki/Feature_hashing) ('the hashing trick') implementation (using [MurmurHash3](http://github.com/spaolacci/murmur3)) for reduced memory requirements and reduced reliance on training data
* Similarity/distance measures to calculate the similarity/distance between feature vectors.

## Planned

* Expanded persistence support
* Stemming to treat words with common root as the same e.g. "go" and "going"
* Clustering algorithms e.g. Heirachical, K-means, etc.
* Classification algorithms e.g. SVM, KNN, random forest, etc.

## References

1. [Rosario, Barbara. Latent Semantic Indexing: An overview. INFOSYS 240 Spring 2000](http://people.ischool.berkeley.edu/~rosario/projects/LSI.pdf)
1. [Latent Semantic Analysis, a scholarpedia article on LSA written by Tom Landauer, one of the creators of LSA.](http://www.scholarpedia.org/article/Latent_semantic_analysis)
1. [Thomo, Alex. Latent Semantic Analysis (Tutorial).](http://webhome.cs.uvic.ca/~thomo/svd.pdf)
1. [Latent Semantic Indexing. Standford NLP Course](http://nlp.stanford.edu/IR-book/html/htmledition/latent-semantic-indexing-1.html)
1. [Charikar, Moses S. "Similarity Estimation Techniques from Rounding Algorithms" in Proceedings of the thiry-fourth annual ACM symposium on Theory of computing - STOC ’02, 2002, p. 380.](https://www.cs.princeton.edu/courses/archive/spr04/cos598B/bib/CharikarEstim.pdf)
1. [M. Bawa, T. Condie, and P. Ganesan, “LSH forest: self-tuning indexes for similarity search,” Proc. 14th Int. Conf. World Wide Web - WWW ’05, p. 651, 2005.](http://dl.acm.org/citation.cfm?id=1060745.1060840)
1. [A. Gionis, P. Indyk, and R. Motwani, “Similarity Search in High Dimensions via Hashing,” VLDB ’99 Proc. 25th Int. Conf. Very Large Data Bases, vol. 99, no. 1, pp. 518–529, 1999.](http://www.cs.princeton.edu/courses/archive/spring13/cos598C/Gionis.pdf%5Cnhttp://portal.acm.org/citation.cfm?id=671516)
1. [Kanerva, Pentti, Kristoferson, Jan and Holst, Anders (2000). Random Indexing of Text Samples for Latent Semantic Analysis](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.4.6523&rep=rep1&type=pdf)
1. [Rangan, Venkat. Discovery of Related Terms in a corpus using Reflective Random Indexing](https://www.umiacs.umd.edu/~oard/desi4/papers/rangan.pdf)
1. [Vasuki, Vidya and Cohen, Trevor. Reflective random indexing for semi-automatic indexing of the biomedical literature](https://ac.els-cdn.com/S1532046410000481/1-s2.0-S1532046410000481-main.pdf?_tid=f31f92e8-028a-11e8-8c31-00000aab0f6c&acdnat=1516965824_e24a804445fff1744281ca6f5898a3a4)
1. [QasemiZadeh, Behrang and Handschuh, Siegfried. Random Indexing Explained with High Probability](http://pars.ie/publications/papers/pre-prints/random-indexing-dr-explained.pdf)
1. [Foulds, James; Boyles, Levi; Dubois, Christopher; Smyth, Padhraic; Welling, Max (2013). Stochastic Collapsed Variational Bayesian Inference for Latent Dirichlet Allocation][SCVB0]

<!--
1. [Geva, Shlomo & De Vries, Christopher M (2011). TOPSIG : Topology Preserving Document Signatures.](https://eprints.qut.edu.au/43451/4/43451.pdf)
-->

[LSA]: https://en.wikipedia.org/wiki/Latent_semantic_analysis
[SCVB0]: https://arxiv.org/pdf/1305.2452
