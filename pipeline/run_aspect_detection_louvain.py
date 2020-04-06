import os
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import glob
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from aspect_detection_louvain import process_aspect_detection
import model
import constants

# os.chdir("./pipeline")


for file in glob.glob("{}*.pkl".format(constants.PREPROCESSED_DATA_PATH)):
    df = pd.read_pickle(file)
    clusters, new_df = process_aspect_detection(df, 50)
    print("Il y a " + str(len(clusters)) + " clusters")
    for i, cluster in enumerate(clusters):
        words_in_cluster = " ".join(cluster)
        word_cloud = WordCloud().generate(words_in_cluster)
        plt.figure()
        plt.imshow(word_cloud, interpolation="bilinear")
        plt.savefig(
            "Clusters Louvain/"
            + (file.split("\\")[1]).split(".")[0]
            + "_cluster_"
            + str(i)
        )
        plt.axis("off")
        plt.show()
    output_path = (
        constants.PROCESSED_DATA_PATH
        + (file.split("\\")[1]).split(".")[0]
        + "_ad_louvain.pkl"
    )

    df.to_pickle(output_path)
