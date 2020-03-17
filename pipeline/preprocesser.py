import pandas as pd
import matplotlib.pyplot as plt

import re
import os

import nltk
from nltk.tokenize import word_tokenize, string_span_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.probability import FreqDist

nltk.download('averaged_perceptron_tagger')

from textblob import TextBlob
tags = ['FW','JJ','JJR','JJS','NN','NNS','NNP','NNPS','RBR','RBS','UH','VB','VBD','VBG','VBN','VBP','VBZ']

def spans(text):
    tokens=word_tokenize(text)
    offset = 0
    i=0
    for token in tokens:
        # !!!!! Because the word_tokenize mistakes the " for `` or '' !!!!!
        if token == "``":
            token = '"'
        if token == "''":
            token = '"'
        offset = text.find(token, offset)
        yield token, offset, offset+len(token)
        offset += len(token)

def tokenize(text):
    token_words = []
    token_words_pos = []
    for s in spans(text):
        token, offset, offset_len = s
        token_words.append(token)
        token_words_pos.append([offset, offset_len])
    return token_words, token_words_pos