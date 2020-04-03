import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import operator
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel, TfidfModel
from nltk.probability import FreqDist


def concat_lists(lists):
    return sum(lists, [])


def get_matrix(n, processed_df):
    reviews_with_sentences = processed_df["words_lemmatized"]
    reviews = processed_df["joined_words"]
    all_words = processed_df["joined_words"].tolist()
    all_words = [inner for outer in all_words for inner in outer]
    fdist = FreqDist(all_words)
    frequent_words = fdist.most_common(n)
    A = np.zeros((n, n))
    words = [word[0] for word in frequent_words]
    frequencies = [word[1] for word in frequent_words]
    for i, review in enumerate(reviews_with_sentences):
        for sentence in review:
            for w1 in sentence:
                if w1 in words:
                    index1 = words.index(w1)
                    for w2 in sentence:
                        if w2 in words:
                            index2 = words.index(w2)
                            A[index1, index2] += 1
    for i in range(len(A)):
        A[i][i] = 1
    return (A, words)


def data_preprocessing(df, no_below, no_above, split_sentences=False):

    if split_sentences:
        df = df.set_index(["review_id"]).apply(pd.Series.explode).reset_index()
        df = df[~df["words_lemmatized"].isnull()]

    else:
        df["words_lemmatized"] = df["words_lemmatized"].apply(concat_lists)

    texts = df["words_lemmatized"]

    # Create dictionnary
    id2word = corpora.Dictionary(texts)
    print("Unique words :", len(id2word))

    # On filter out les mots qui appraraissent dans moins de 2 textes ou dans plus de 50% des textes
    # Filter out words in less than no_below documents or in more than no_above % documents
    id2word.filter_extremes(no_below=no_below, no_above=no_above)
    print(
        "Unique words after filtering out the most and least frequent ones :",
        len(id2word),
    )

    # Create Corpus
    corpus = [id2word.doc2bow(text) for text in texts]

    # tfidf = TfidfModel(corpus)
    # corpus_tfidf = tfidf[corpus]

    return df, texts, id2word, corpus


def get_classe(i, classes):
    for c in classes:
        if i in c:
            return c


def louvain_1(critere, a, classes):
    n = np.shape(a)[0]
    dico_fusion = {}
    for i in range(n):
        classe_i = get_classe(i, classes)
        moins_classe_i = np.sum([critere(i, j, a) for j in classe_i])
        delta = 0
        classe_found = None
        for classe in classes:
            plus_classe = np.sum([critere(i, j, a) for j in classe])
            if i not in classe:
                plus_classe += critere(i, i, a)
            current_delta = -moins_classe_i + plus_classe
            if current_delta > delta:
                delta = current_delta
                classe_found = classe
        if classe_found:
            classe_found.update({i})
            classe_i.remove(i)
    return [s for s in classes if s]


def merge(classes, a):
    n = len(classes)
    new_a = np.zeros((n, n))
    new_i = 0
    dico = {}
    for classe in classes:
        for i in classe:
            dico[i] = new_i
        new_i += 1
    for i in range(np.shape(a)[0]):
        for j in range(np.shape(a)[1]):
            new_a[dico[i]][dico[j]] = new_a[dico[i]][dico[j]] + a[i][j]
    return new_a


def louvain(critere, a, classes_finales={}):
    n = np.shape(a)[0]
    classes = [{i} for i in range(n)]
    if not classes_finales:
        classes_finales = classes.copy()
    classes = louvain_1(critere, a, classes)
    a = merge(classes, a)
    classes_finales = [
        set.union(*[classes_finales[i] for i in classe]) for classe in classes
    ]
    if len(classes) == n:
        return classes_finales
    else:
        return louvain(critere, a, classes_finales)


def classes_labels(classes_finales, n, words):
    clusters = []
    for i, classe in enumerate(classes_finales):
        cluster = []
        for i in classe:
            cluster.append(words[i])
        clusters.append(cluster)
    return clusters


def critere1(i, j, a):
    s = a[i][j] - np.sum(a[i]) * np.sum(a[j]) / (2 * len(a))
    return s


def critere2(i, j, a):
    s = (
        a[i][j]
        - np.sum(a[i]) / len(a)
        - np.sum(a[j]) / len(a)
        + np.sum(np.sum(a)) / len(a) ** 2
    )
    return s


def critere3(i, j, a):
    s = a[i][j] - np.sum(a[i]) * np.sum(a[j]) / (np.sum(np.sum(a)))
    return s


def is_in_clusters(word, clusters):
    res = False
    index = 0
    for i, classe in enumerate(clusters):
        for w in classe:
            if word == w:
                res = True
                index = i
                break
    return res, index


def process_aspect_detection(df, n):
    A, words = get_matrix(n, df)
    clusters = classes_labels(louvain(critere3, A), n, words)
    sentences = df["words_lemmatized"]
    cs = []
    for sentence in sentences:
        c = []
        for word in sentence:
            res, i = is_in_clusters(word, clusters)
            if res:
                c.append(i)
            else:
                c.append("None")
        cs.append(c)
    df["clustered"] = cs
    return clusters, df
