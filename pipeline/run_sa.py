import os
import pandas as pd
import argparse
from tqdm import tqdm
import glob

from sa import Sentiment
import model
import constants

os.chdir("./pipeline")

def process(df):
    SentimentAnalyser = Sentiment(load_model='./imdb_clf.pt', visualize=True)
    SentimentAnalyser.initialize()

    df['words_sentiment'] = None

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        text = row['text']
        vals = SentimentAnalyser.process(text)
        tokens_sentiment = []
        # test first word
        for i in range(len(row['words'])):
            token = row['words'][i]
            token_index = row['words_index'][i]
            token_sentiments = vals[token_index[0]:token_index[1]]
            # Compute the mean sentiment of the word by averaging the sentimens of all its characters
            token_mean_sentiment = sum(token_sentiments)/len(token_sentiments)
            tokens_sentiment.append(round(token_mean_sentiment, 3))
        df.at[index, 'words_sentiment'] = tokens_sentiment
        
for file in glob.glob("{}*.pkl".format(constants.TOKENIZED_DATA_PATH)):
    df = pd.read_pickle(file).head(1)
    process(df)
    # You may want to change this if not on Windows OS
    output_path = constants.PROCESSED_DATA_PATH + '_'.join((file.split('.')[-2]).split('\\')[-1].split('_')[:-1]) + '_sa.pkl'
    df.to_pickle(output_path)