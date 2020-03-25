import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
import ast
import os
import constants
sns.set_style({'font.family': 'monospace'})

os.chdir("./pipeline")

def make_heatmap(text, tokens_values, tokens_index, save=None, polarity=1):
    values = [0]*len(text)
    for i in range(len(tokens_index)):
        values[tokens_index[i][0]:tokens_index[i][1]] = [tokens_values[i]]*(tokens_index[i][1]-tokens_index[i][0])

    cell_height=.325
    cell_width=.15
    n_limit = 74
    text = list(map(lambda x: x.replace('\n', '\\n'), text))
    num_chars = len(text)
    total_chars = math.ceil(num_chars/float(n_limit))*n_limit
    mask = np.array([0]*num_chars + [1]*(total_chars-num_chars))
    text = np.array(text+[' ']*(total_chars-num_chars))
    values = np.array(values+[0]*(total_chars-num_chars))
    values *= polarity

    # error again
    values = np.array([value.item() if type(value) != int else value for value in values])

    values = values.reshape(-1, n_limit)
    text = text.reshape(-1, n_limit)
    mask = mask.reshape(-1, n_limit)
    num_rows = len(values)
    plt.figure(figsize=(cell_width*n_limit, cell_height*num_rows))
    hmap=sns.heatmap(values, annot=text, mask=mask, fmt='', vmin=-1, vmax=1, cmap='RdYlGn',
                     xticklabels=False, yticklabels=False, cbar=False)
    plt.tight_layout()
    if save is not None:
        plt.savefig(save)
    # clear plot for next graph since we returned `hmap`
    plt.clf()
    return hmap

n_to_plot = 10

df = pd.read_pickle(constants.PROCESSED_DATA_PATH + 'input_example_tokenized_sa.pkl')
df_sample = df.sample(10)

i = 0
for index, row in df_sample.iterrows():
    hmap = make_heatmap(row['text'], row['tokens_sentiment'], row['tokens_index'], './heatmap_examples/example_{}'.format(i))
    i+=1