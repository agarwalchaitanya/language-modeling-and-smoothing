from lnm.ln_model import ln_model
import numpy as np

class kneser_ney(ln_model):
    def __init__(self, corpus):
        super().__init__(corpus)
        self.__discount = 0.75
        self.__unigram_continuation_weight = self.__discount/(self._number_of_unigrams**2)
    
    def __calc_norm_constant(self, context, n):
        ngram = self._return_count(n)
        ngram_count = 0
        weight = 0
        try:
            ngram_count = np.sum(list(ngram[context].values()))
            weight = len(list(ngram[context].keys()))
        except:
            if n==3:
                ngram_count = self._number_of_trigrams
            elif n==2:
                ngram_count = self._number_of_bigrams
        return (self.__discount*weight)/ngram_count

    def __kn_unigram_continuation_probability(self, head):
        continuation_count = 0
        for con, continuation_tokens in self._bigram_counts.items():
            if head in list(continuation_tokens.keys()):
                continuation_count+=1
        return max(continuation_count-self.__discount, 0)/self._number_of_bigrams + self.__unigram_continuation_weight
    
    def __kn_unigram_count_probability(self, head):
        count = 0
        try:
            count = self._unigram_counts[head]
        except:
            pass
        sum_of_freq_of_unigrams = np.sum(list(self._unigram_counts.values()))
        return max(count-self.__discount, 0)/sum_of_freq_of_unigrams + self.__unigram_continuation_weight
            
    def __kn_bigram_continuation_probability(self, head, context):
        continuation_count = 0
        for con, continuation_tokens in self._trigram_counts.items():
            if head in list(continuation_tokens.keys()):
                continuation_count+=1
        return max(continuation_count-self.__discount, 0)/self._number_of_trigrams + self.__calc_norm_constant(context, 2)*self.__kn_unigram_continuation_probability(head)
    
    def __kn_bigram_count_probability(self, head, context):
        count = 0
        sum_of_freq_of_bigrams = 0
        try:
            count = self._bigram_counts[context][head]
        except:
            pass
        for con, tokens in self._bigram_counts.items():
            sum_of_freq_of_bigrams+=np.sum(list(tokens.values()))
        return max(count-self.__discount, 0)/sum_of_freq_of_bigrams + self.__calc_norm_constant(context, 2)*self.__kn_unigram_continuation_probability(head)

    def __kn_trigram_count_probability(self, head, context):
        count = 0
        sum_of_freq_of_trigrams = 0
        try:
            count = self._trigram_counts[context][head]
        except:
            pass
        for con, tokens in self._trigram_counts.items():
            sum_of_freq_of_trigrams+=np.sum(list(tokens.values()))
        return max(count-self.__discount, 0)/sum_of_freq_of_trigrams + self.__calc_norm_constant(context, 3)*self.__kn_bigram_continuation_probability(head, context.split()[1])
    
    def estimate(self, sentence, highest_order):
        probability = 1
        tokens = [token.text for token in self._nlp(sentence)]
        if highest_order==3:
            for i, token in enumerate(tokens):
                try:
                    probability*=self.__kn_trigram_count_probability(tokens[i+2], token+' '+token[i+1])
                except:
                    pass
            return probability
        elif highest_order==2:
            for i, token in enumerate(tokens):
                try:
                    probability*=self.__kn_bigram_count_probability(tokens[i+1], token)
                except:
                    pass
            return probability
        elif highest_order==1:
            for i, token in enumerate(tokens):
                probability*=self.__kn_unigram_count_probability(token)
            return probability
        return 0
