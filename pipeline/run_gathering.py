import os
import pandas as pd
import argparse
from tqdm import tqdm
import glob

import model
import constants

os.chdir("./pipeline")

df_sa = pd.read_pickle("data/processed/input_example_sa.pkl")
print(df_sa.columns)
df_ad = pd.read_pickle("data/processed/input_example_ad.pkl")

df = pd.merge(df_ad, df_sa, on="review_id")

print(df.iloc[0])


df["sentences_cluster"] = None
df["sentences_sentiment"] = None

for index, row in df.iterrows():

    document_sentences_cluster = []
    document_sentences_words_sentiment = []
    # For each sentence in the document
    for sentence_index in range(len(row['sentences_words_index'])):
        sentence_words_sentiment = []
        # For clustering
        sentence_clusters_dict = row['clustered'][sentence_index]
        try:
            sentence_cluster = max(sentence_clusters_dict, key=sentence_clusters_dict.get)
        except ValueError as e:
            sentence_cluster = None
        document_sentences_cluster.append(sentence_cluster)
        # For sentiment
        # words_start = 
        # words_end = len(row["sentences_words_index"][sentence_index])
        # sentence_words_sentiment = row["words_sentiment"][words_index_start:words_index_end]
        # document_sentences_words_sentiment.append(sentence_words_sentiment)

        # words_index_start = words_index_end 

    df.at[index, "sentences_cluster"] = document_sentences_cluster
    # df.at[index, "sentences_sentiment"] = document_sentences_words_sentiment


df = df[["review_id", "text", "words_index", "words_sentiment", "sentences_words_index", "sentences_cluster"]]

df.to_pickle(constants.PROCESSED_DATA_PATH + "/output.pkl")