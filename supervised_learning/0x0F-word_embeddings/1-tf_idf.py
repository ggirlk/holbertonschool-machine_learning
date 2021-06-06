#!/usr/bin/env python3
""" TF-IDF """
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
bag_of_words = __import__('0-bag_of_words').bag_of_words


def tf_idf(sentences, vocab=None):
    """
    ***********************************************
    ********** creates a TF-IDF embedding *********
    ***********************************************
    @sentences: is a list of sentences to analyze
    @vocab: is a list of the vocabulary words to use
            for the analysis
            **If None: all words within sentences
                       should be used
    Returns: embeddings, features
             embeddings: is a numpy.ndarray of shape (s, f)
             containing the embeddings
                 s is the number of sentences in sentences
                 f is the number of features analyzed
             features: is a list of the features used for embeddings
    """
    E, vocab = bag_of_words(sentences, vocab)
    return TfidfTransformer().fit_transform(E).toarray(), vocab
