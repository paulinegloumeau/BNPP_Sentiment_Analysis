import pandas as pd
import matplotlib.pyplot as plt

import re
import os

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.probability import FreqDist

nltk.download('averaged_perceptron_tagger')

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from data_preprocesser import preprocessed_data_path, create_file, split_raw_csv_review_file

from textblob import TextBlob
tags = ['FW','JJ','JJR','JJS','NN','NNS','NNP','NNPS','RBR','RBS','UH','VB','VBD','VBG','VBN','VBP','VBZ']

raw_data_path = './data_yelp/raw/'
preprocessed_data_path = './data_yelp/preprocessed/'

# Processing Functions

def identify_tokens(row):
    sentences = row['sentences']
    token_words = []
    for sentence in sentences :
        tokens = nltk.word_tokenize(sentence)
        # taken only words (not punctuation)
        # token_words = [re.sub("[^\w\s]", " ", w) for w in tokens]
        token_words.append([w for w in tokens if w.isalpha()])
    return token_words


def split_to_sentences(row):
    review = row['text']
    delimiters = ['?','!','.',',',';',':','\n','"','(',')']
    regexPattern = '|'.join(map(re.escape, delimiters))
    sentences = re.split(regexPattern,review)
    sentences = [sentence for sentence in sentences if sentence and sentence != ' ']
    return sentences

### Garder juste les noms, verbes, adjectifs et interjections

def select_tagged(row):
    sentences = row['sentences']
    useful_words = []
    for sentence in sentences :
        list_words = sentence.split(' ')
        list_words = [w for w in list_words if w]
        tagged_words = nltk.pos_tag(list_words)
        useful_words.append([w[0] for w in tagged_words if w[1] in tags and w[0] != 'i' and w[0] != 'u'])
    return useful_words

stop_words = stopwords.words('english')
stops_empiric = ['i','u',"it's","don't","they're",'wa',"didn't",'was','done','were','went','had','got','has','am',"i'm","i've"]
stops_empiric.extend(stop_words)


def remove_stops(row):
    sentences = row['words_useful']
    useful_words = []

    for sentence in sentences :
        new_sentence = []
        for w in sentence :
            if w not in stops_empiric :
                new_sentence.append(w)
        useful_words.append(new_sentence)
    return useful_words

### Enlever les mots trops polarisés

def unpolarized(row):
    sentences = row['words_useful']
    unpolarized_words = []
    for sentence in sentences :
        unpolarized_words.append([w for w in sentence if (abs(TextBlob(w).sentiment.polarity)<0.2 
            and abs(TextBlob(w).sentiment.polarity<0.2))])
    return unpolarized_words

stemming = PorterStemmer()

def stem_list(row):
    sentences = row['words_meaningful']
    stemmed_sentences = []
    for sentence in sentences :
        stemmed_sentences.append([stemming.stem(word) for word in sentence])
    return (stemmed_sentences)

lemming = WordNetLemmatizer()

def lem_list(row):
    sentences = row['words_meaningful']
    lemmed_sentences = []
    for sentence in sentences :
        lemmed_sentences.append([lemming.lemmatize(word) for word in sentence])
    return (lemmed_sentences)

def rejoin_words(row):
    sentences = row['words_meaningful']
    joined_words = []
    for sentence in sentences :
        joined_words.extend(sentence)
    return joined_words

def index_words(row):
    sentences = row['words_meaningful']
    all_words = row['sentences']
    indexes = []
    for j,sentence in enumerate(sentences) :
        indexes.append([(w,all_words[j].index(w)) for w in sentence])
    return indexes

def process(data_df):
    data_df['text'] = data_df['text'].str.lower()
    print(1)
    data_df['sentences'] = data_df.apply(split_to_sentences,axis=1)
    print(2)
    data_df['words'] = data_df.apply(identify_tokens,axis=1)
    print(3)
    data_df['words_useful'] = data_df.apply(select_tagged,axis=1)
    print(4)
    data_df['words_useful'] = data_df.apply(remove_stops,axis=1)
    print('4bis')
    data_df['words_meaningful'] = data_df.apply(unpolarized, axis=1)
    print(5)
    data_df['index_meaningful'] = data_df.apply(index_words,axis=1)
    print(6)
    data_df['words_lemmatized'] = data_df.apply(lem_list, axis=1)
    print(7)
    data_df['joined_words'] = data_df.apply(rejoin_words,axis=1)

    return data_df

create_file('data_yelp/clustering/')

for file in os.listdir('data_yelp/preprocessed/categories_30000'):
    print(file)
    df = pd.read_csv('data_yelp/preprocessed/categories_30000/' + file).head(10)
    df_processed = process(df)
    df_processed.to_csv('data_yelp/clustering/' + file)