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

def make_document_heatmap(text, sentences_words_index, sentences_words_sentiment, sentences_cluster, save=None, polarity=1):
    n_sentences = len(sentences_cluster)
    f, axes = plt.subplots(n_sentences, 1)


    for i in range(n_sentences):
        cluster = sentences_cluster[i]

        # ax1 = plt.subplot(n_sentences,1,i+1)

        words_sentiment = sentences_words_sentiment[i]
        words_index = sentences_words_index[i]

        sentence_text = text[words_index[0][0]:words_index[-1][1]+1]

        # The index of the sentence first character
        sentence_offset = words_index[0][0]

        # We need to have a value per character
        sentiment_values = [0]*len(sentence_text)
        for j in range(len(words_index)):
            sentiment_values[(words_index[j][0]-sentence_offset):(words_index[j][1]-sentence_offset)] = [words_sentiment[j]]*(words_index[j][1]-words_index[j][0])

        # We compute the average sentiment for the sentence, we delete 0 values because they represent spaces
        average_sentiment = round(sum([i for i in sentiment_values if i!=0])/len([i for i in sentiment_values if i!=0]), 2)

        cell_height=.325
        cell_width=.15
        n_limit = 74
        sentence_text = list(map(lambda x: x.replace('\n', '\\n'), sentence_text))
        num_chars = len(sentence_text)
        total_chars = math.ceil(num_chars/float(n_limit))*n_limit
        mask = np.array([0]*num_chars + [1]*(total_chars-num_chars))
        sentence_text = np.array(sentence_text+[' ']*(total_chars-num_chars))
        sentiment_values = np.array(sentiment_values+[0]*(total_chars-num_chars))
        sentiment_values *= polarity

        # error again
        sentiment_values = np.array([value.item() if type(value) != int else value for value in sentiment_values])
        sentiment_values = sentiment_values.reshape(-1, n_limit)
        sentence_text = sentence_text.reshape(-1, n_limit)
        mask = mask.reshape(-1, n_limit)
        num_rows = len(sentiment_values)
        # plt.figure(figsize=(cell_width*n_limit, cell_height*num_rows))
        hmap=sns.heatmap(sentiment_values, annot=sentence_text, mask=mask, fmt='', vmin=-1, vmax=1, cmap='RdYlGn',
                         xticklabels=False, yticklabels=False, cbar=False, ax=axes[i])
                        
        axes[i].set_title('Sentence {sentence_index} : cluster {cluster_index} & average sentiment {average_sentiment}'.format(
            sentence_index=i, 
            cluster_index=cluster,
            average_sentiment=average_sentiment
            ))

    plt.tight_layout()
    if save is not None:
        plt.savefig(save)
    # clear plot for next graph since we returned `hmap`
    plt.clf()
    return hmap
    

n_to_plot = 1

df = pd.read_pickle(constants.PROCESSED_DATA_PATH + 'input_example_output.pkl')
df_clusters = pd.read_pickle(constants.PROCESSED_DATA_PATH + 'input_example_clusters.pkl')

df_sample = df.sample(1)

i = 0
for index, row in df_sample.iterrows():
    hmap = make_document_heatmap(row['text'], row['sentences_index'], row['sentences_sentiment'], row['sentences_cluster'], './heatmap_examples/example_{}'.format(i))
    i+=1