/*
Package nlp provides implementations of selected machine learning algorithms for natural language processing of text corpora.  The primary focus is the statistical semantics of plain-text documents supporting semantic analysis and retrieval of semantically similar documents.

The package makes use of the Gonum (http://http//www.gonum.org/) library for linear algebra and scientific computing with some inspiration taken from Python's scikit-learn (http://scikit-learn.org/stable/) and Gensim(https://radimrehurek.com/gensim/)

Overview

The primary intended use case is to support document input as text strings encoded as a matrix of numerical feature vectors called a `term document matrix`.  Each column in the matrix corresponds to a document in the corpus and each row corresponds to a unique term occurring in the corpus.  The individual elements within the matrix contain the frequency with which each term occurs within each document (referred to as `term frequency`).  Whilst textual data from document corpora are the primary intended use case, the algorithms can be used with other types of data from other sources once encoded (vectorised) into a suitable matrix e.g. image data, sound data, users/products, etc.

These matrices can be processed and manipulated through the application of additional transformations for weighting features, identifying relationships or optimising the data for analysis, information retrieval and/or predictions.

Typically the algorithms in this package implement one of two interfaces:

	Vectoriser - Taking document input as strings and outputting matrices of numerical features.
	Transformer - Takes matrices of numerical features and applies logic/transformation to output a new matrix.

One of the implementations of Vectoriser is Pipeline which can be used to wire together pipelines composed of a Vectoriser and one or more Transformers arranged in serial so that the output from each stage forma the input of the next.  This can be used to construct a classic LSI (Latent Semantic Indexing) pipeline (vectoriser -> TF.IDF weighting -> Truncated SVD):

	pipeline := nlp.NewPipeline(
		nlp.NewCountVectoriser(true),
		nlp.NewTFIDFTransformer(),
		nlp.NewTruncatedSVD(100),
	)

A common transformation is `TF.IDF`` for the purpose of weighting features to remove natural biases which would skew results e.g. commonly occurring words like `the`, `of`, `and`, etc. which should carry lower weight than unusual words.

Term Document matrices typically have a very large number of dimensions which can cause issues with memory and performance but also skew distance/similarity measures in vector space.  Transformations are often applied to reduce the dimensionality using techniques such as Random Projection, Principal Component Analysis and Singular Value Decomposition.  These approximate the original term document matrix with a new matrix of much lower rank (typically 100s of dimensions rather than 10s or 100s of thousands).

Dimensionality reduction is also an important aspect of NLP techniques like Locality Sensitive Hashing and Latent Semantic Analysis/Indexing (typically performed using matrix SVD - `Singular Value Decomposition` or Random Indexing).  The dimensionality reduction is used to exchange a high-number of features for a much smaller number of 'better' features that represent the latent semantic variables within the document inferred through term co-occurance.

As an obvious conclusion, processed and transformed matrices can be compared for similarity with each other (e.g. for cluster analysis or training a classifier) or with a query (also represented as a feature vector projected into the same dimensional space).  Various pairwise similarity and distance measures are provided within the package to support various use cases.
*/
package nlp
