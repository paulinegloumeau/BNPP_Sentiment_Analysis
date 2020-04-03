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

def make_heatmap(text, words_values, words_index, save=None, polarity=1):
    print(words_index[2], words_values[2])


    # values = [0]*len(text)
    # for i in range(len(words_index)):
    #     values[words_index[i][0]:words_index[i][1]] = [words_values[i]]*(words_index[i][1]-words_index[i][0])

    # cell_height=.325
    # cell_width=.15
    # n_limit = 74
    # text = list(map(lambda x: x.replace('\n', '\\n'), text))
    # num_chars = len(text)
    # total_chars = math.ceil(num_chars/float(n_limit))*n_limit
    # mask = np.array([0]*num_chars + [1]*(total_chars-num_chars))
    # text = np.array(text+[' ']*(total_chars-num_chars))
    # values = np.array(values+[0]*(total_chars-num_chars))
    # values *= polarity

    # # error again
    # values = np.array([value.item() if type(value) != int else value for value in values])

    # values = values.reshape(-1, n_limit)
    # text = text.reshape(-1, n_limit)
    # mask = mask.reshape(-1, n_limit)
    # num_rows = len(values)
    # plt.figure(figsize=(cell_width*n_limit, cell_height*num_rows))
    # hmap=sns.heatmap(values, annot=text, mask=mask, fmt='', vmin=-1, vmax=1, cmap='RdYlGn',
    #                  xticklabels=False, yticklabels=False, cbar=False)
    # plt.tight_layout()
    # if save is not None:
    #     plt.savefig(save)
    # # clear plot for next graph since we returned `hmap`
    # plt.clf()
    # return hmap

n_to_plot = 1

df = pd.read_pickle(constants.PROCESSED_DATA_PATH + 'output.pkl')
df_sample = df.sample(1)

i = 0
for index, row in df_sample.iterrows():
    hmap = make_heatmap(row['text'], row['sentences_sentiment'], row['sentences_words_index'], './heatmap_examples/example_{}'.format(i))
    i+=1