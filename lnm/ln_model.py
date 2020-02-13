import sys
from lnm.parser import parse
from spacy.lang.en import English
import numpy as np

class ln_model():
    def __init__(self, corpus):
        self._unigram_counts = {}
        self._bigram_counts = {}
        self._trigram_counts = {}
        self._number_of_unigrams = 0
        self._number_of_bigrams = 0
        self._number_of_trigrams = 0

        self._nlp = English()
        self.__corpus = corpus
        self.__parse()
        self.__ngrams_size()
    
    def __ngrams_size(self):
        self._number_of_unigrams = len(list(self._unigram_counts.keys()))
        for context, tokens in self._bigram_counts.items():
            self._number_of_bigrams+=len(list(tokens.keys()))
        for context, tokens in self._trigram_counts.items():
            self._number_of_trigrams+=len(list(tokens.keys()))

    def _return_count(self, n):
        if n==1:
            return self._unigram_counts
        elif n==2:
            return self._bigram_counts
        elif n==3:
            return self._trigram_counts
        
    def __parse(self):
        self._unigram_counts, self._bigram_counts, self._trigram_counts = parse(self.__corpus)

    def _mle_unigram_probability(self, head):
        p = 0
        try:
            p = self._unigram_counts[head]/np.sum(list(self._unigram_counts.values()))
        except:
            return 0
        return p

    def _mle_bigram_probability(self, head, context):
        p = 0
        try:
            p = self._bigram_counts[context][head]/np.sum(list(self._bigram_counts[context].values()))
        except:
            return 0
        return p
    
    def _mle_trigram_probability(self, head, context):
        p = 0
        try:
            p = self._trigram_counts[context][head]/np.sum(list(self._trigram_counts[context].values()))
        except:
            return 0
        return p
    
    def _preprocess_input(self, sentence):
        return 'start_symbol '+sentence.rstrip().lstrip()+' end_symbol'
    