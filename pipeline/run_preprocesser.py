import os
import pandas as pd
import glob

from utils import create_file
import constants
import preprocesser

os.chdir("./pipeline")

#  Create the folders if they did not exist
create_file(constants.RAW_DATA_PATH)
create_file(constants.PREPROCESSED_DATA_PATH)
create_file(constants.TOKENIZED_DATA_PATH)
create_file(constants.PROCESSED_DATA_PATH)

#  Run the first preprocessing step, i.e. tokenizing and getting the index of the tokenized words


def tokenize(df):
    # Only keep the id of the review, and its text
    print("Tokenizing ...")
    try:
        df = df[["review_id", "text"]]
    except KeyError:
        print("Input dataframe should contain a column text and a column review_id")
    else:
        #  Lower case only
        df["text"] = df["text"].str.lower()
        # Get the tokenized words and their index
        df[["words", "words_index"]] = df.apply(
            preprocesser.tokenize, axis=1, result_type="expand"
        )
        return df


def preprocess(df, light=False):
    print("Splitting to sentences and removing non alpha words ...")
    df[["sentences", "sentences_words_index"]] = df.apply(
        preprocesser.split_to_sentences, axis=1, result_type="expand"
    )
    print("Keeping only nouns ...")
    df[["words_useful", "words_useful_index"]] = df.apply(
        preprocesser.select_tagged, axis=1, result_type="expand"
    )
    print("Removing some common stop words ...")
    df[["words_useful", "words_useful_index"]] = df.apply(
        preprocesser.remove_stops, axis=1, result_type="expand"
    )
    print("Lemmatizing ...")
    df[["words_lemmatized", "words_lemmatized_index"]] = df.apply(
        preprocesser.lem_list, axis=1, result_type="expand"
    )
    print("Keeping only unpolarized words ...")
    df[["words_meaningful", "words_meaningful_index"]] = df.apply(
        preprocesser.unpolarized, axis=1, result_type="expand"
    )
    print("Rejoining the words to recreate sentences")
    df["joined_words"] = df.apply(preprocesser.rejoin_words, axis=1)
    if light:
        df = df[
            ["review_id", "words_lemmatized", "words_lemmatized_index", "joined_words"]
        ]
    return df


for file in glob.glob("{}*.csv".format(constants.RAW_DATA_PATH)):
    print("Test :        ", file)
    # We load the input data
    df = pd.read_csv(file)
    # Run the tokenizer and save the output, used for sa
    df_tokenized = tokenize(df)
    output_path_tokenized = (
        constants.TOKENIZED_DATA_PATH
        + (file.split("\\")[-1]).split(".")[0]
        + "_tokenized.pkl"
    )
    df_tokenized.to_pickle(output_path_tokenized)
    # Run the remaining preprocessing functions needed for aspect detection
    df_preprocessed = preprocess(df_tokenized, light=True)
    output_path_preprocessed = (
        constants.PREPROCESSED_DATA_PATH
        + (file.split("\\")[-1]).split(".")[0]
        + "_preprocessed.pkl"
    )
    print(df_preprocessed.columns)
    df_preprocessed.to_pickle(output_path_preprocessed)
