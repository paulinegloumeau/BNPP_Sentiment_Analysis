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
import argparse
sns.set_style({'font.family': 'monospace'})

from utils import create_file, empty_directory, Radar, radar_factory

os.chdir("./pipeline")

def make_document_heatmap(text, sentences_words_index, sentences_words_sentiment, sentences_cluster, save=None, polarity=1):
    cell_height=.35
    cell_width=.15
    n_limit = 74

    n_sentences = len(sentences_cluster)

    ### First we retrieve the "height" each sentence will need in the plot

    sentences_height = []

    for i in range(n_sentences):
        words_index = sentences_words_index[i]
        sentence_text = text[words_index[0][0]:words_index[-1][1]+1]

        # We need to have a value per character
        sentiment_values = [0]*len(sentence_text)

        sentence_text = list(map(lambda x: x.replace('\n', '\\n'), sentence_text))
        num_chars = len(sentence_text)
        total_chars = math.ceil(num_chars/float(n_limit))*n_limit
        sentiment_values = np.array(sentiment_values+[0]*(total_chars-num_chars))
        sentiment_values *= polarity

        sentiment_values = np.array([value.item() if type(value) != int else value for value in sentiment_values])
        sentiment_values = sentiment_values.reshape(-1, n_limit)
        num_rows = len(sentiment_values)

        sentences_height.append(cell_height*num_rows)

    # We can instantiate the plot
    f, axes = plt.subplots(n_sentences, ncols=1, gridspec_kw={'height_ratios':sentences_height}, figsize=(cell_width*n_limit, sum(sentences_height)*num_rows))

    ### Then we can feed the plot
    ### NB : doing twice the loot may not be optimized, this could be improved  
    for i in range(n_sentences):
        cluster = sentences_cluster[i]

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

        sentence_text = list(map(lambda x: x.replace('\n', '\\n'), sentence_text))
        num_chars = len(sentence_text)
        total_chars = math.ceil(num_chars/float(n_limit))*n_limit
        mask = np.array([0]*num_chars + [1]*(total_chars-num_chars))
        sentence_text = np.array(sentence_text+[' ']*(total_chars-num_chars))
        sentiment_values = np.array(sentiment_values+[0]*(total_chars-num_chars))
        sentiment_values *= polarity

        sentiment_values = np.array([value.item() if type(value) != int else value for value in sentiment_values])
        sentiment_values = sentiment_values.reshape(-1, n_limit)
        sentence_text = sentence_text.reshape(-1, n_limit)
        mask = mask.reshape(-1, n_limit)
        num_rows = len(sentiment_values)

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

def make_heatmaps(df, n, output_dir):
    df_sample = df.sample(n)
    for index, row in df_sample.iterrows():
        hmap = make_document_heatmap(row['text'], row['sentences_index'], row['sentences_sentiment'], row['sentences_cluster'], output_dir + '/heatmap_{}'.format(row['review_id']))

def make_global_analysis(df, df_clusters, output_dir):
    clusters_list = list(df_clusters.columns)
    clusters_list.append(None)
    clusters_mean_sentiment = dict.fromkeys(clusters_list, 0)
    clusters_sentiment = dict.fromkeys(clusters_list, [])

    for index, row in df.iterrows():
        n_sentences = len(row['sentences_cluster'])
        text = row['text']
        for i in range(n_sentences):
            cluster = row['sentences_cluster'][i]

            words_sentiment = row['sentences_sentiment'][i]
            words_index = row['sentences_index'][i]

            sentence_text = text[words_index[0][0]:words_index[-1][1]+1]

            # The index of the sentence first character
            sentence_offset = words_index[0][0]

            # We need to have a value per character
            sentiment_values = [0]*len(sentence_text)
            for j in range(len(words_index)):
                sentiment_values[(words_index[j][0]-sentence_offset):(words_index[j][1]-sentence_offset)] = [words_sentiment[j]]*(words_index[j][1]-words_index[j][0])

            # We compute the average sentiment for the sentence, we delete 0 values because they represent spaces
            average_sentiment = round(sum([i for i in sentiment_values if i!=0])/len([i for i in sentiment_values if i!=0]), 2)

            clusters_sentiment[cluster] = clusters_sentiment.get(cluster, []) + [average_sentiment]

    for key, value in clusters_sentiment.items():
        clusters_mean_sentiment[key] = (sum(value)/len(value))

    fig = plt.figure()
    labels = [[-1, -0.5, 0, 0.5, 1] for i in range(len(clusters_list))]
    radar = Radar(fig, [cluster if cluster is not None else "No Cluster" for cluster in clusters_list], labels)
    radar.plot(list(clusters_mean_sentiment.values()), '-', lw=2, color='b', alpha=0.4)
    plt.tight_layout()

    plt.savefig(output_dir + '/clusters_sentiment')
    plt.clf()

    fig, ax = plt.subplots()
    df_clusters = df_clusters.head(5)
    ax.table(cellText=df_clusters.values, colLabels=df_clusters.columns, loc='center')
    fig.tight_layout()
    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    plt.savefig(output_dir + '/clusters_top_words')


if __name__ == "__main__":
    create_file(constants.VISUALIZATION_PATH)

    parser = argparse.ArgumentParser()

    parser.add_argument("--processed-file", type=str, default='data/output/input_example_output.pkl',
                        help="Input pkl file, must be an output of run_gathering.py script")
    
    parser.add_argument("--clusters-file", type=str, default='data/output/input_example_clusters.pkl',
                    help="Input clusters pkl file, must be an output of run_aspect_detection_louvain.py script")

    parser.add_argument("--output-dir", type=str, default='data/visualization/input_example',
                        help="Output directory in the visualization directory")

    parser.add_argument('--heatmap', action='store_true', default=False,
                        help='Should random heatmaps be created')
    
    parser.add_argument('--heatmap-n', type=int, default=1,
                    help='How many heatmap to create')

    parser.add_argument('--global-analysis', action='store_true', default=False,
                    help='Should the global analysis be created')

    parser.add_argument('--empty-dir', type=bool, default=True,
                    help='Should the output directory be emptied before running the script')

    args = parser.parse_args()

    df = pd.read_pickle(args.processed_file)
    create_file(args.output_dir)

    if args.empty_dir:
        empty_directory(args.output_dir)

    if args.heatmap:
        make_heatmaps(df, args.heatmap_n, args.output_dir)

    if args.global_analysis:
        df_clusters = pd.read_pickle(args.clusters_file)
        make_global_analysis(df, df_clusters, args.output_dir)