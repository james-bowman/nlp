/*
Package nlp provides implementations of selected machine learning algorithms for natural language processing of text corpora.  The primary focus is the statistical semantics of plain-text documents supporting semantic analysis and retrieval of semantically similar documents.

The package makes use of the Gonum (http://http//www.gonum.org/) library for linear algebra and scientific computing with some inspiration taken from Python's scikit-learn (http://scikit-learn.org/stable/) and Gensim(https://radimrehurek.com/gensim/)

Overview

The primary intended use case is to support document input as text strings encoded as a matrix of numerical feature vectors called a `term document matrix`.  Each column in the matrix corresponds to a document in the corpus and each row corresponds to a unique term occurring in the corpus.  The individual elements within the matrix contain the frequency with which each term occurs within each document (referred to as `term frequency`).  Whilst textual data from document corpora are the primary intended use case, the algorithms can be used with other types of data from other sources once encoded (vectorised) into a suitable matrix e.g. image data, sound data, users/products, etc.

These matrices can be processed and manipulated through the application of additional transformations for weighting features, identifying relationships or optimising the data for analysis, information retrieval and/or predictions.

Typically the algorithms in this package implement one of three primary interfaces:

	Vectoriser - Taking document input as strings and outputting matrices of numerical features e.g. term frequency.
	Transformer - Takes matrices of numerical features and applies some logic/transformation to output a new matrix.
	Comparer - Functions taking two vectors (columns from a matrix) and outputting a distance/similarity measure.

One of the implementations of Vectoriser is Pipeline which can be used to wire together pipelines composed of a Vectoriser and one or more Transformers arranged in serial so that the output from each stage forms the input of the next.  This can be used to construct a classic LSI (Latent Semantic Indexing) pipeline (vectoriser -> TF.IDF weighting -> Truncated SVD):

	pipeline := nlp.NewPipeline(
		nlp.NewCountVectoriser(true),
		nlp.NewTFIDFTransformer(),
		nlp.NewTruncatedSVD(100),
	)

Whilst they take different inputs, both Vectorisers and Transformers have 3 primary methods:

	Fit() - Trains the model based upon the supplied, input training data.
	Transform() - Transforms the input into the output matrix (requires the model to be already fitted by a previous call to Fit() or FitTransform()).
	FitTransform() - Convenience method combining Fit() and Transform() methods to transform input data, fitting the model to the input data in the process.
*/
package nlp
