#!/usr/bin/env python3
""" Dataset """
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset():
    """ loads and preps a dataset for machine translation """

    def __init__(self):
        """
        *********************************************
        *****************Constructor*****************
        *********************************************
        """
        # 👇 contains the ted_hrlr_translate/pt_to_en tf.data.Dataset
        # train split, loaded as_supervided
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True)
        # 👇 contains the ted_hrlr_translate/pt_to_en tf.data.Dataset
        # validate split, loaded as_supervided
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)
        # the Portuguese tokenizer created from the training set
        self.tokenizer_pt = self.tokenize_dataset(self.data_train)[0]
        # the English tokenizer created from the training set
        self.tokenizer_en = self.tokenize_dataset(self.data_train)[1]

    def tokenize_dataset(self, data):
        """
        ***************************************************
        ****Creates sub-word tokenizers for our dataset****
        ***************************************************
        @data: is a tf.data.Dataset whose examples are formatted as a tuple (pt, en)
               pt: is the tf.Tensor containing the Portuguese sentence
               en: is the tf.Tensor containing the corresponding English sentence
        *** The maximum vocab size should be set to 2**15
        Returns:
                tokenizer_pt: is the Portuguese tokenizer
                tokenizer_en: is the English tokenizer
        """
        tf.compat.v1.enable_eager_execution()
        builder = tfds.features.text.SubwordTextEncoder.build_from_corpus
        pt = builder((pt.numpy() for pt, _ in data.repeat(1)), 2**15)
        en = builder((en.numpy() for _, en in data.repeat(1)), 2**15)
        return pt, en
