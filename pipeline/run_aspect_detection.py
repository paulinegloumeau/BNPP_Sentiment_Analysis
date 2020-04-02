import os
import pandas as pd
import argparse
from tqdm import tqdm
import glob

from aspect_detection import data_preprocessing, LDA
import model
import constants

os.chdir("./pipeline")

def process(df):
    df, texts, id2word, corpus = data_preprocessing(df, no_below=2, no_above=0.8, split_sentences=False)
    df = LDA(df, texts, corpus, id2word, 5)
        
for file in glob.glob("{}*.pkl".format(constants.PREPROCESSED_DATA_PATH)):
    df = pd.read_pickle(file)
    process(df)
    output_path = constants.PROCESSED_DATA_PATH + '_'.join((file.split('.')[-2]).split('/')[-1].split('_')[:-1]) + '_ad.pkl'
    print(df)
    df.to_pickle(output_path)
