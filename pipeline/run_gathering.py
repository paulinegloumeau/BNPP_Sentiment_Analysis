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


df["sentences_cluster"] = None
df["sentences_sentiment"] = None
df["sentences_index"] = None

for index, row in df.iterrows():
    sentence_start = 0

    document_sentences_cluster = []
    document_sentences_words_sentiment = []
    document_sentences_words_index = []

    # For each sentence in the document, we rebuild its index (preprocessing actually messed them up) and retrieve cluster and words sentiment
    for sentence_index in range(len(row['sentences_words_index'])):
        sentence_words_sentiment = []

        # For clustering
        sentence_clusters_dict = row['clustered'][sentence_index]
        try:
            sentence_cluster = max(sentence_clusters_dict, key=sentence_clusters_dict.get)
        except ValueError as e:
            sentence_cluster = None
        document_sentences_cluster.append(sentence_cluster)

        # For words sentiment
        sentence_end = row["words_index"].index(row["delimiters_index"][sentence_index]) + 1
        sentence_words_sentiment = row["words_sentiment"][sentence_start:sentence_end]
        document_sentences_words_sentiment.append(sentence_words_sentiment)

        # For words index
        sentences_words_index = row["words_index"][sentence_start:sentence_end]
        document_sentences_words_index.append(sentences_words_index) 

        sentence_start = sentence_end


    df.at[index, "sentences_cluster"] = document_sentences_cluster
    df.at[index, "sentences_sentiment"] = document_sentences_words_sentiment
    df.at[index, "sentences_index"] = document_sentences_words_index


test = sum(df.iloc[0]["sentences_sentiment"], [])
print(df.iloc[0]["delimiters_index"])
print(df.iloc[0]["words_index"])

print(len(test))
print(test)
print(df.iloc[0]['words_sentiment'])

print(df.iloc[0])


df = df[["review_id", "text", "sentences_sentiment", "sentences_cluster", "sentences_index"]]


df.to_pickle(constants.PROCESSED_DATA_PATH + "/output.pkl")