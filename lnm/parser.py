#!/bin/python3
import os
from nltk import sent_tokenize
from spacy.lang.en import English
import numpy as np

def parse(corpus):
    corpus = corpus.lower().splitlines()
    plc = '' 
    for i, line in enumerate(corpus):
        line = line.lstrip(' \t\n').rstrip(' \t\n')
        if line!='':
            if line[0]=='[' or line[0]=='}':
                pass
            else:
                plc+=line+' '
    corpus = sent_tokenize(plc)
    corpus = ['start_symbol '+line+' end_symbol' for line in corpus] 

    nlp = English()
    unigram_count= {}
    bigram_count = {}
    trigram_count = {}

    for sent in corpus:
        doc = nlp(sent)
        tokens = [token.text for token in doc]
        for i, token in enumerate(tokens):
            try:
                unigram_count[token]+=1
            except:
                unigram_count[token]=1
            if i+1<len(tokens):
                try:
                    bigram_count[token]
                except:
                    bigram_count[token] = {}
                try:
                    bigram_count[token][tokens[i+1]]+=1
                except:
                    bigram_count[token][tokens[i+1]]=1
            
            if i+2<len(tokens):
                try:
                    trigram_count[token+' '+tokens[i+1]]
                except:
                    trigram_count[token+' '+tokens[i+1]] = {}
                try:
                    trigram_count[token+' '+tokens[i+1]][tokens[i+2]]+=1
                except:
                    trigram_count[token+' '+tokens[i+1]][tokens[i+2]]=1
    
    return unigram_count, bigram_count, trigram_count
