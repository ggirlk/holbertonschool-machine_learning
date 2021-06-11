#!/usr/bin/env python3
import numpy as np


def ngramify(wordsList, n):
    """
    ***************************************************
    *** Converts a words list of 1-grams to n-grams ***
    ***************************************************
    @wordsList: list of words of sentences
    @n: number of grams
    Return:
            n-grams word list
    """
    unlist = 0
    if type(wordsList[0]) is not list:
        wordsList = [wordsList]
        unlist = 1
    nWordsList = []
    for line in wordsList:
        new_line = []
        for gram in range(len(line) - n + 1):
            new_gram = ""
            for i in range(n):
                if i != 0:
                    new_gram += " "
                new_gram += line[gram + i]
            new_line.append(new_gram)
        nWordsList.append(new_line)
    if unlist:
        return nWordsList[0]
    return nWordsList


def ngram_bleu(references, sentence, n):
    """
    ******************************************************
    *** calculates the n-gram BLEU score for a sentence **
    ******************************************************
    @references: is a list of reference translations
                 each reference translation is a list
                 of the words in the translation
    @sentence: is a list containing the model proposed sentence
    @n: is the size of the n-gram to use for evaluation
    Returns:
            the unigram BLEU score
    """
    references = ngramify(references, n)
    sentence = ngramify(sentence, n)
    wordsDict = {}
    for gram in sentence:
        wordsDict[gram] = wordsDict.get(gram, 0) + 1
    max_dict = {}
    for reference in references:
        ref = {}
        for gram in reference:
            ref[gram] = ref.get(gram, 0) + 1
        for gram in ref:
            max_dict[gram] = max(max_dict.get(gram, 0), ref[gram])
    in_ref = 0
    for gram in wordsDict:
        in_ref += min(max_dict.get(gram, 0), wordsDict[gram])
    closest = np.argmin(np.abs([len(ref) - len(sentence)
                        for ref in references]))
    closest = len(references[closest])
    if len(sentence) >= closest:
        brevity = 1
    else:
        brevity = np.exp(1 - (closest + n - 1) / (len(sentence) + n - 1))
    return brevity * in_ref / len(sentence)
