#!/usr/bin/env python3
""" Extract Word2Vec """
from tensorflow import keras


def word2vec_model(sentences, size=100, min_count=5, window=5,
                   negative=5, cbow=True, iterations=5, seed=0, workers=1):
    """
    *************************************************
    ** converts a gensim word2vec model to a keras **
    **************** Embedding layer ****************
    *************************************************
    @model: is a trained gensim word2vec models

    Returns: the trainable keras Embedding
    """
    return model.wv.get_keras_embedding(train_embeddings=True)
