#!/usr/bin/env python3
""" BLEU: bilingual evaluation understudy """
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


def ngram_modscore(references, sentence, n):
    """
    **************************************************
    ********** Calculate unigram bleu score **********
    **************************************************
    @references: is a list of reference translations
                each reference translation is a list
                of the words in the translation
    @sentence: is a list containing the model proposed sentence
    @n: is the size of the largest n-gram to use for evaluation
    *** All n-gram scores should be weighted evenly
    Returns:
            the unigram n-gram BLEU score
    """
    references = ngramify(references, n)
    sentence = ngramify(sentence, n)
    sent_dict = {}
    for gram in sentence:
        sent_dict[gram] = sent_dict.get(gram, 0) + 1
    max_dict = {}
    for reference in references:
        this_ref = {}
        for gram in reference:
            this_ref[gram] = this_ref.get(gram, 0) + 1
        for gram in this_ref:
            max_dict[gram] = max(max_dict.get(gram, 0), this_ref[gram])
    in_ref = 0
    for gram in sent_dict:
        in_ref += min(max_dict.get(gram, 0), sent_dict[gram])
    return np.log(in_ref / len(sentence))


def cumulative_bleu(references, sentence, n):
    """
    ***************************************************
    *** calculates the cumulative n-gram BLEU score ***
    ***************************************************
    @references: is a list of reference translations
                each reference translation is a list
                of the words in the translation
    @sentence: is a list containing the model proposed sentence
    @n: is the size of the largest n-gram to use for evaluation
    *** All n-gram scores should be weighted evenly
    Returns:
            the cumulative n-gram BLEU score

    """
    weight = 1 / n
    scores = [ngram_modscore(references, sentence, i) * weight
              for i in range(1, n + 1)]
    closest = np.argmin(np.abs([len(ref) - len(sentence)
                        for ref in references]))
    closest = len(references[closest])
    if len(sentence) >= closest:
        brevity = 1
    else:
        brevity = np.exp(1 - closest / len(sentence))
    return brevity * np.exp(sum(scores))
