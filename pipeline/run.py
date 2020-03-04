import os
import pandas as pd
import argparse
from tqdm import tqdm

from analyse import Sentiment
import model


parser = argparse.ArgumentParser(description='Sentiment Analysis and Aspects Detection Pipeline')

# Model parameters.
parser.add_argument('--input', type=str, default = 'pipeline/input_example.csv',
                    help='Path to input data, .csv file. Should have the following columns : review_id, text')

args = parser.parse_args()

df = pd.read_csv(args.input)[['review_id', 'text']].head(1)

df_short = df.set_index('review_id').T.to_dict('list')

Sentiment = Sentiment(load_model='pipeline/imdb_clf.pt', visualize=True)
Sentiment.initialize()

for index, row in tqdm(df.iterrows()):
    text = row['text']
    vals = Sentiment.process(text)
    print(len(' '))