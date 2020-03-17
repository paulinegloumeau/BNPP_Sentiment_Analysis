import os
import pandas as pd
import glob

from utils import create_file
import constants
import preprocesser

os.chdir("./pipeline")

# Create the folders if they did not exist
create_file(constants.RAW_DATA_PATH)
create_file(constants.PREPROCESSED_DATA_PATH)
create_file(constants.TOKENIZED_DATA_PATH)
create_file(constants.PROCESSED_DATA_PATH)

# Run the first preprocessing step, i.e. tokenizing and getting the index of the tokenized words

def process(data_df):
    # Only keep the id of the review, and its text
    data_df = data_df[['review_id', 'text']]
    # Lower case only
    data_df['text'] = data_df['text'].str.lower()
    # Get the tokenized words and their index
    tokens, tokens_index = zip(*data_df['text'].map(preprocesser.tokenize))
    data_df['tokens'] = tokens
    data_df['tokens_index'] = tokens_index
    return data_df

for file in glob.glob("{}*.csv".format(constants.RAW_DATA_PATH)):
    df = pd.read_csv(file)
    data_df = process(df)
    output_path = constants.TOKENIZED_DATA_PATH + (file.split('.')[-2]).split('/')[-1] + '_tokenized.pkl'
    data_df.to_pickle(output_path)