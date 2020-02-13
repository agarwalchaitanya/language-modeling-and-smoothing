from lnm.ln_model import ln_model

class witten_bell(ln_model):
    def __init(self, corpus):
        super().__init__(self, corpus)

    def __calc_back_off_weight(self, head, context, n):
        ngram = self._return_count(n)
        context_count = 0
        seq_count = 0
        try:
            context_count = len(list(ngram[context].keys()))
        except:
            pass

        try:
            seq_count = self._return_count(n)[context][head]
        except:
            pass
        if context_count==0 and seq_count==0:
            return 0
        else:
            return (context_count)/(context_count+seq_count)

    def __wb_unigram_probability(self, head):
        '''
        source: https://www.ee.columbia.edu/~stanchen/e6884/labs/lab3/x207.html
        '''
        return 1/self._number_of_unigrams

    def __wb_bigram_probability(self, head, context):
        back_off_weight = self.__calc_back_off_weight(head, context, 2)
        return (1-back_off_weight)*self._mle_bigram_probability(head, context) + back_off_weight*self.__wb_unigram_probability(head)

    def __wb_trigram_probability(self, head, context):
        back_off_weight = self.__calc_back_off_weight(head, context, 3)
        return (1-back_off_weight)*self._mle_trigram_probability(head, context) + back_off_weight*self.__wb_bigram_probability(head, context.split()[1])

    def estimate(self, sentence, n):
        sentence = self._preprocess_input(sentence)
        probability = 1
        tokens = [token.text for token in self._nlp(sentence)]
        if n==3:
            for i, token in enumerate(tokens):
                try:
                    probability*=self.__wb_trigram_probability(tokens[i+2], token+' '+tokens[i+1])
                except:
                    pass
            return probability
        elif n==2:
            for i, token in enumerate(tokens):
                try:
                    probability*=self.__wb_bigram_probability(tokens[i+1], token)
                except:
                    pass
            return probability
        elif n==1:
            for i, token in enumerate(tokens):
                probability*=self.__wb_unigram_probability(token)
            return probability
        return 0
            
