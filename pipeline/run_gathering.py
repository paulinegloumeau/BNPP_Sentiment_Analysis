import os
import pandas as pd
import argparse
from tqdm import tqdm
import glob

import model
import constants

os.chdir("./pipeline")

df_sa = pd.read_pickle("data/processed/input_example_sa.pkl")
print(df_sa.columns)
df_ad = pd.read_pickle("data/processed/input_example_ad.pkl")

df = pd.merge(df_ad, df_sa, on="review_id")


print(df.iloc[0])

df.to_csv(constants.PROCESSED_DATA_PATH + "/output.csv")