import os
import pandas as pd
import glob

from utils import create_file
import constants
import preprocesser

os.chdir("./pipeline")

def run_preprocess(data_df):
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

for file in glob.glob("{}*.csv".format(constants.RAW_DATA_PATH)):
    df = pd.read_csv(file)
    data_df = process(df)
    output_path = constants.TOKENIZED_DATA_PATH + (file.split('.')[-2]).split('/')[-1] + '_tokenized.pkl'
    data_df.to_pickle(output_path)