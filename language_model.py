#!/bin/python3
from lnm.ln_model import ln_model
from lnm.witten_bell import witten_bell
from lnm.kneser_ney import kneser_ney
import sys

def _take_input():
    n = int(sys.argv[1])
    smoothing_alg = sys.argv[2]
    corpus_path = sys.argv[3]
    sentence = input('input sentence: ')
    return n, smoothing_alg, corpus_path, sentence

def _calculate_probability(n, smoothing_alg, corpus_path, sentence):
    if smoothing_alg == 'k':
        model = kneser_ney(open(corpus_path).read())
        print(model.estimate(sentence, n))
    elif smoothing_alg == 'w':
        model = witten_bell(open(corpus_path).read())
        print(model.estimate(sentence, n))
    return 
    
if __name__ == "__main__":
   n, smoothing_alg, corpus_path, sentence = _take_input()
   _calculate_probability(n, smoothing_alg, corpus_path, sentence)
