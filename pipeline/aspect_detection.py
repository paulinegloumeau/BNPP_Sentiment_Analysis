import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import operator
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel, TfidfModel

def concat_lists(lists):
    return sum(lists, [])

def data_preprocessing(df, no_below, no_above, split_sentences=False):

    if split_sentences:
        df = df.set_index(['review_id']).apply(pd.Series.explode).reset_index()
        df = df[~df['words_lemmatized'].isnull()]

    else:
        df['words_lemmatized'] = df['words_lemmatized'].apply(concat_lists)

    texts = df['words_lemmatized']

    # Create dictionnary
    id2word = corpora.Dictionary(texts)
    print('Unique words :', len(id2word))

    # On filter out les mots qui appraraissent dans moins de 2 textes ou dans plus de 50% des textes
    # Filter out words in less than no_below documents or in more than no_above % documents
    id2word.filter_extremes(no_below=no_below, no_above=no_above)
    print('Unique words after filtering out the most and least frequent ones :', len(id2word))

    # Create Corpus
    corpus = [id2word.doc2bow(text) for text in texts]

    # tfidf = TfidfModel(corpus)
    # corpus_tfidf = tfidf[corpus]

    return df, texts, id2word, corpus

def LDA(df, texts, corpus, id2word, num_topics):
    # Compute the LDA model
    print("Running Latent Dirichlet Allocation ...")
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, 
                                                id2word=id2word, 
                                                num_topics=num_topics, 
                                                update_every=1, 
                                                chunksize=100, 
                                                passes=10, 
                                                alpha='auto', 
                                                per_word_topics=True)
    # Compute coherence score
    coherencemodel = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence='c_v')
    print("Coherence score : " + str(coherencemodel.get_coherence()))   
    # Get all the topics probabilities for each word
    words_topics = lda_model.get_topics()
    # Keep the highest one
    words_best_topic = np.argmax(words_topics, axis=0)
    words_best_topic_p = np.max(words_topics, axis=0)

    df['words_topic'] = None
    #np.empty((len(df), 0)).tolist()

    # Add the list of the best topics (for each word) for each document
    for index, row in df.iterrows():
        words_idx = id2word.doc2idx(row['words_lemmatized'])
        words_topic = [words_best_topic[i] if i!=-1 else -1 for i in words_idx]
        df.at[index, 'words_topic'] = words_topic

    return df


def ret_top_model(texts, corpus, id2word):
    """
    Since LDAmodel is a probabilistic model, it comes up different topics each time we run it. To control the
    quality of the topic model we produce, we can see what the interpretability of the best topic is and keep
    evaluating the topic model until this threshold is crossed. 
    
    Returns:
    -------
    lm: Final evaluated topic model
    top_topics: ranked topics in decreasing order. List of tuples
    """
    top_topics = [(0, 0)]
    for i in range(10):
        lm = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word)
        coherence_values = {}
        for n, topic in lm.show_topics(num_topics=-1, formatted=False):
            topic = [word for word, _ in topic]
            cm = CoherenceModel(topics=[topic], texts=texts, dictionary=id2word, window_size=10)
            coherence_values[n] = cm.get_coherence()
        top_topics = sorted(coherence_values.items(), key=operator.itemgetter(1), reverse=True)
    return lm, top_topics