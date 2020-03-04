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

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from data_preprocesser import preprocessed_data_path, create_file, split_raw_csv_review_file

from textblob import TextBlob
tags = ['FW','JJ','JJR','JJS','NN','NNS','NNP','NNPS','RBR','RBS','UH','VB','VBD','VBG','VBN','VBP','VBZ']

raw_data_path = './data_yelp/raw/'
preprocessed_data_path = './data_yelp/preprocessed/'
reviews_type_path = 'categories_30000/yelp_academic_dataset_review_Auto Repair.csv'

# Processing Functions

def split_to_sentences(row):
    review = row['text'] 
    sentences = review.split(r'[.,;:]')
    return sentences


def identify_tokens(row):
    review = row['sentences']
    tokens = nltk.word_tokenize(review)
    # taken only words (not punctuation)
    # token_words = [re.sub("[^\w\s]", " ", w) for w in tokens]
    token_words = [w for w in tokens if w.isalpha()]
    return token_words

stops = set(stopwords.words("english"))

def remove_stops(row):
    my_list = row['words']
    meaningful_words = [w for w in my_list if not w in stops and w != " "]
    return (meaningful_words)

### Garder juste les noms, verbes, adjectifs et interjections

def select_tagged(row):
    my_list = row['words_non_stop']
    tagged_words = nltk.pos_tag(my_list)
    useful_words = [w[0] for w in tagged_words if w[1] in tags]
    return useful_words


### Enlever les mots trops polarisés

def unpolarized(row):
    my_list = row['words_useful']
    unpolarized_words = [w for w in my_list if (abs(TextBlob(w).sentiment.polarity)<0.2 and abs(TextBlob(w).sentiment.polarity<0.2))]
    return unpolarized_words

stemming = PorterStemmer()

def stem_list(row):
    my_list = row['words_meaningful']
    stemmed_list = [stemming.stem(word) for word in my_list]
    return (stemmed_list)

lemming = WordNetLemmatizer()

def lem_list(row):
    my_list = row['words_non_stop']
    lemmed_list = [lemming.lemmatize(word) for word in my_list]
    return (lemmed_list)

def rejoin_words(row):
    my_list = row['words_meaningful']
    joined_words = ( " ".join(my_list))
    return joined_words

def index_words(row):
    my_list = row['words_meaningful']
    all_words = row['words']
    indexes = []
    for w in my_list :
        indexes.append((w,all_words.index(w)))
    return indexes

def process(data_df):
    data_df['text'] = data_df['text'].str.lower()
    print(1)
    data_df['words'] = data_df.apply(identify_tokens, axis=1)
    print(2)
    data_df['words_non_stop'] = data_df.apply(remove_stops, axis=1)
    print(3)
    data_df['words_useful'] = data_df.apply(select_tagged,axis=1)
    print(4)
    data_df['words_meaningful'] = data_df.apply(unpolarized, axis=1)
    print(5)
    data_df['index_meaningful'] = data_df.apply(index_words,axis=1)
    print(6)
    data_df['words_lemmatized'] = data_df.apply(lem_list, axis=1)

    return data_df



for file in os.listdir('data_yelp/preprocessed/categories_30000'):
    print(file)
    df = pd.read_csv('data_yelp/preprocessed/categories_30000/' + file)
    df_processed = process(df)
    df_processed.to_csv('data_yelp/clustering/' + file)




