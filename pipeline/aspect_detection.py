import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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



df = pd.read_pickle('pipeline/data/preprocessed/input_example_preprocessed.pkl')
df['words_lemmatized'] = df['words_lemmatized'].apply(concat_lists)
print(df)
texts, id2word, corpus = data_preprocessing(df['words_lemmatized'], no_below=2, no_above=0.5)
lda_model = LDA(texts, corpus, id2word, 5)
