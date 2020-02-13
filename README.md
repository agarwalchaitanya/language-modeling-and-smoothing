# Language Modeling and Smoothing   

## Getting Started
`cd` into the repository and install dependencies mentioned in [`requirements.txt`](requirements.txt) in a virtual environment (preferably):
```
virtualenv .env
source .env/bin/activate
pip3 install -r requirements.txt
```
run the test script to estimate probabilities for any sentence
```
$ python3 language_model.py <value of n> <smoothing type> <path to txt corpus>
>input sentence: <input sentence>
```
where `<value of n>` can be `1`, `2` or `3`, and `<smoothing type>` can be `k` for Kneyser-Ney or `w` for Witten-Bell.

## Analysis
### Preprocessing Pipeline
> the following discussion is with respect to the given [`resources/corpus.txt`](resources/corpus.txt) file which is cleaned off of the names of the book and the poet.
#### Sentence Segmentation and Tokenization
`lnm/parser.py` uses nltk's sent_tokenize to tokenize sentences which then uses spacy's nlp() pipeline to tokenize the sentences. Sentence boundaries are marked by `start_symbol` and `end_symbol` out of which the ngram counts are calculated.

### Comparison

## Author(s)
[Chaitanya Agarwal](htts://www.github.com/agarwalchaitanya)
