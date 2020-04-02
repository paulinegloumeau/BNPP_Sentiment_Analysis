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

############# Tokenizing functions used for Sentiment Analysis and Aspect Detection #############

def spans(text):
    words=word_tokenize(text)
    offset = 0
    i=0
    for word in words:
        # !!!!! Because the word_tokenize mistakes the " for `` or '' !!!!!
        if word == "``":
            word = '"'
        if word == "''":
            word = '"'
        offset = text.find(word, offset)
        yield word, offset, offset+len(word)
        offset += len(word)

def tokenize(row):
    text = row['text']
    words = []
    words_index = []
    for s in spans(text):
        word, offset, offset_len = s
        words.append(word)
        words_index.append([offset, offset_len])
    return words, words_index

############# Preprocessing functions used for Aspect Detection #############

def split_to_sentences(row, many_delimiters=False):
    words = row['words']
    words_index = row['words_index']

    delimiters = ['?','!','.']
    if many_delimiters:
        delimiters = ['?','!','.',',',';',':','\n','"','(',')']

    sentences = []
    sentences_words_index = []

    sentence = ''
    sentence_words_index = []
    for i in range(len(words)):
        if words[i] not in delimiters:
            if words[i].isalpha():
                sentence = sentence + ' ' + words[i] if sentence != '' else sentence + words[i]
                sentence_words_index.append(words_index[i])
        else:
            if sentence != '':
                sentences.append(sentence)
                sentences_words_index.append(sentence_words_index)
            sentence = ''
            sentence_words_index = []
        
    return sentences, sentences_words_index

def select_tagged(row):
    sentences = row['sentences']
    sentences_words_index = row['sentences_words_index']

    useful_words = []
    useful_words_index = []

    for i in range(len(sentences)):
        list_words = sentences[i].split(' ')
        assert len(list_words) == len(sentences_words_index[i])
        tagged_words = nltk.pos_tag(list_words)
        useful_sentence_words = []
        useful_sentence_words_index = []
        for j in range(len(tagged_words)):
            if tagged_words[j][1] in tags and tagged_words[j][0] != 'i' and tagged_words[j][1] != 'u':
                useful_sentence_words.append(tagged_words[j][0])
                useful_sentence_words_index.append(sentences_words_index[i][j])
        useful_words.append(useful_sentence_words)
        useful_words_index.append(useful_sentence_words_index)

    return useful_words, useful_words_index

stop_words = stopwords.words('english')
stops_empiric = ['i','u',"it's","don't","they're",'wa',"didn't",'was','done','were','went','had','got','has','am',"i'm","i've"]
stops_empiric.extend(stop_words)

def remove_stops(row):
    words_useful = row['words_useful']
    words_useful_index = row['words_useful_index']

    useful_words = []
    useful_words_index = []

    # For each sentence 
    for i in range(len(words_useful)):
        sentence_remove_stops = []
        sentence_remove_stops_index = []
        # For each word in this sentence
        for j in range(len(words_useful[i])):
            if words_useful[i][j] not in stops_empiric:
                sentence_remove_stops.append(words_useful[i][j])
                sentence_remove_stops_index.append(words_useful_index[i][j])
        useful_words.append(sentence_remove_stops)
        useful_words_index.append(sentence_remove_stops_index)

    return useful_words, useful_words_index

def unpolarized(row):
    words_useful = row['words_useful']
    words_useful_index = row['words_useful_index']

    unpolarized_words = []
    unpolarized_words_index = []

    # For each sentence
    for i in range(len(words_useful)):
        sentence_unpolarized = []
        sentence_unpolarized_index = []
        # For each word in this sentence
        for j in range(len(words_useful[i])):
            if (abs(TextBlob(words_useful[i][j]).sentiment.polarity)<0.2 and abs(TextBlob(words_useful[i][j]).sentiment.polarity<0.2)):
                sentence_unpolarized.append(words_useful[i][j])
                sentence_unpolarized_index.append(words_useful_index[i][j])
        unpolarized_words.append(sentence_unpolarized)
        unpolarized_words_index.append(sentence_unpolarized_index)
    
    return unpolarized_words, unpolarized_words_index

lemming = WordNetLemmatizer()

def lem_list(row):
    words_meaningful = row['words_meaningful']
    words_meaningful_index = row['words_meaningful_index']

    lemmed_words = []
    lemmed_words_index = []

    # For each sentence
    for i in range(len(words_meaningful)):
        sentence_lemmed = []
        sentence_lemmed_index = []
        # For each word in this sentence
        for j in range(len(words_meaningful[i])):
            word_lemmed = lemming.lemmatize(words_meaningful[i][j])
            if word_lemmed != '':
                sentence_lemmed.append(word_lemmed)
                sentence_lemmed_index.append(words_meaningful_index[i][j])
        lemmed_words.append(sentence_lemmed)
        lemmed_words_index.append(sentence_lemmed_index)

    return lemmed_words, lemmed_words_index

def rejoin_words(row):
    sentences = row['words_meaningful']
    joined_words = []
    for sentence in sentences :
        joined_words.extend(sentence)
    return joined_words