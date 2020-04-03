import pandas as pd
import constants
import os

os.chdir('pipeline/')

df = pd.read_pickle(constants.PROCESSED_DATA_PATH + 'input_example_preprocessed_ad.pkl')
print(df)