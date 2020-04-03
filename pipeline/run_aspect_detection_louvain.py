import os
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import glob


from aspect_detection_louvain import process_aspect_detection
import model
import constants

# os.chdir("./pipeline")


for file in glob.glob("{}*.pkl".format(constants.PREPROCESSED_DATA_PATH)):
    df = pd.read_pickle(file)
    clusters, new_df = process_aspect_detection(df, 50)
    print("Il y a " + str(len(clusters)) + " clusters")
    print(clusters)
    output_path = (
        constants.PROCESSED_DATA_PATH + (file.split("\\")[1]).split(".")[0] + "_ad.pkl"
    )
    df.to_pickle(output_path)
