#!/usr/bin/env python3
""" Bag Of Words """
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """
    ***********************************************
    *** creates a bag of words embedding matrix ***
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
    vect = CountVectorizer(vocabulary=vocab)
    data = vect.fit_transform(sentences)
    return data.toarray(), vect.get_feature_names()
