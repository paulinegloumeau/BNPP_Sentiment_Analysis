import os
import pandas as pd
import argparse
from tqdm import tqdm
import glob
from nltk.probability import FreqDist


from aspect_detection_louvain import (
    get_matrix,
    louvain,
    critere3,
    classes_labels,
    attribute_cluster,
)
import model
import constants

os.chdir("./pipeline")


def get_matrix(n, processed_df):
    reviews_with_sentences = processed_df["words_lemmatized"]
    reviews = processed_df["joined_words"]
    all_words = processed_df["joined_words"].tolist()
    all_words = [inner for outer in all_words for inner in outer]
    fdist = FreqDist(all_words)
    frequent_words = fdist.most_common(n)
    A = np.zeros((n, n))
    words = [word[0] for word in frequent_words]
    frequencies = [word[1] for word in frequent_words]
    for i, review in enumerate(reviews_with_sentences):
        for sentence in review:
            for w1 in sentence:
                if w1 in words:
                    index1 = words.index(w1)
                    for w2 in sentence:
                        if w2 in words:
                            index2 = words.index(w2)
                            A[index1, index2] += 1
    for i in range(len(A)):
        A[i][i] = 1
    return (A, words)


for file in glob.glob("{}*.pkl".format(constants.PREPROCESSED_DATA_PATH)):
    df = pd.read_pickle(file)
    process(df)
    A, words = get_matrix(100, df)
    clusters = classes_labels(louvain(critere3, A), 100)
    print("Il y a " + len(clusters) + " clusters : ", clusters)
    df["clustered"] = df.apply(attribute_cluster, axis=1)
    output_path = (
        constants.PROCESSED_DATA_PATH
        + "_".join((file.split(".")[-2]).split("/")[-1].split("_")[:-1])
        + "_ad.pkl"
    )
    print(df)
    df.to_pickle(output_path)
