import os
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import glob
import argparse

from aspect_detection_louvain import process_aspect_detection
import model
import constants

os.chdir("./pipeline")

for file in glob.glob("{}*.pkl".format(constants.PREPROCESSED_DATA_PATH)):
    df = pd.read_pickle(file)
    clusters, new_df = process_aspect_detection(df, 50)
    print("There are " + str(len(clusters)) + " clusters")
    print(clusters)
    output_path = (
        constants.PROCESSED_DATA_PATH +  '_'.join((file.split("\\")[1]).split(".")[0].split('_')[:-1]) + "_ad.pkl"
    )
    df.to_pickle(output_path)

    df_clusters = pd.concat([pd.Series(x) for x in clusters], axis=1)
    output_path = (
        constants.OUTPUT_DATA_PATH +  '_'.join((file.split("\\")[1]).split(".")[0].split('_')[:-1]) + "_clusters.pkl"
    )
    df_clusters.to_pickle(output_path)
