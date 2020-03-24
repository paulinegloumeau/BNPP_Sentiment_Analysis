import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import operator
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

def concat_lists(lists):
    return sum(lists, [])

def data_preprocessing(serie, no_below, no_above):
    # Create dictionnary
    id2word = corpora.Dictionary(serie)
    print('Unique words :', len(id2word))

    # On filter out les mots qui appraraissent dans moins de 2 textes ou dans plus de 50% des textes
    # Filter out words in less than no_below documents or in more than no_above % documents
    id2word.filter_extremes(no_below=no_below, no_above=no_above)
    print('Unique words after filtering out the most and least frequent ones :', len(id2word))

    # Create Corpus
    texts = serie
    corpus = [id2word.doc2bow(text) for text in texts]

    return texts, id2word, corpus

def LDA(texts, corpus, id2word, num_topics):
    print("Running Latent Dirichlet Allocation ...")
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, 
                                                id2word=id2word, 
                                                num_topics=num_topics, 
                                                update_every=1, 
                                                chunksize=100, 
                                                passes=10, 
                                                alpha='auto', 
                                                per_word_topics=True)
    coherencemodel = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence='c_v')
    print("Coherence score : " + str(coherencemodel.get_coherence()))    
    return lda_model


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

def get_topics(lda_model, threshold):
    all_topics = lda_model.get_document_topics()




df = pd.read_pickle('pipeline/data/preprocessed/yelp_academic_dataset_review_Auto Repair_preprocessed.pkl')
df['words_lemmatized'] = df['words_lemmatized'].apply(concat_lists)
print(df)
texts, id2word, corpus = data_preprocessing(df['words_lemmatized'], no_below=2, no_above=0.5)

lda_model = LDA(texts, corpus, id2word, 5)
print(lda_model.get_document_topics(corpus[0], per_word_topics=True))
print(lda_model.get_document_topics(corpus[0], minimum_probability=0.2, per_word_topics=True))
print(df.iloc[0])
topics = lda_model.get_topics()

first_topic = topics[0]
first_topic.sort()
print(first_topic[-10:])

# lm, top_topics = ret_top_model(texts, corpus, id2word)
# print(top_topics[:5])
# pprint([lm.show_topic(topicid) for topicid, c_v in top_topics[:5]])

