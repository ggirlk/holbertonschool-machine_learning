#!/usr/bin/env python3
""" BLEU: bilingual evaluation understudy """
import numpy as np


def uni_bleu(references, sentence):
    """
    ******************************************************
    ** Calculates the unigram BLEU score for a sentence **
    ******************************************************
    @references: is a list of reference translations
                 each reference translation is a list
                 of the words in the translation
    @sentence: is a list containing the model proposed sentence
    Returns:
            the unigram BLEU score
    """
    wordsDict = {}
    for word in sentence:
        wordsDict[word] = wordsDict.get(word, 0) + 1
    maxs = {}
    for reference in references:
        ref = {}
        for word in reference:
            ref[word] = ref.get(word, 0) + 1
        for word in ref:
            maxs[word] = max(maxs.get(word, 0), ref[word])
    in_ref = 0
    for word in wordsDict:
        in_ref += min(maxs.get(word, 0), wordsDict[word])
    closest = np.argmin(np.abs([len(ref) - len(sentence)
                                for ref in references]))
    closest = len(references[closest])
    if len(sentence) >= closest:
        brevity = 1
    else:
        brevity = np.exp(1 - closest / len(sentence))
    return brevity * in_ref / len(sentence)
