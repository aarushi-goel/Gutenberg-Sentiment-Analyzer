from __future__ import print_function
import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
import mpld3
import pickle
from sklearn.externals import joblib
from nltk.corpus import wordnet as wn

def load_data(file):
    with open(file, "rb") as f:
        books = pickle.load(f)

    titles = map(lambda x: x._title, books)
    synopses = map(lambda x: x._content, books)

    return list(titles), list(synopses)


def tokenize_data(synopses):
    # load nltk's English stopwords as variable called 'stopwords'
    stopwords = nltk.corpus.stopwords.words('english')
    # print (stopwords[:10])

    # load nltk's SnowballStemmer as variabled 'stemmer'
    from nltk.stem.snowball import SnowballStemmer
    stemmer = SnowballStemmer("english")

    # next, we will stem and tokenize. We define a function stem_tokenize that will stem and tokenize depending on parameters
    vocab_stemmed = []
    vocab_tokenized = []
    for i in synopses:
        vocab_stemmed.extend(tokenize_and_stem(i, stem=True, stemmer=stemmer))
        vocab_tokenized.extend(tokenize_and_stem(i))

    return vocab_tokenized, vocab_stemmed


def tokenize_and_stem(text, stem=False, stemmer=None):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    if stem:
        stems = [stemmer.stem(t) for t in filtered_tokens]
        return stems
    return filtered_tokens


def tf_idf_calc(synopses):
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                       min_df=0.2, stop_words='english',
                                       use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1, 3))

    tfidf_matrix = tfidf_vectorizer.fit_transform(synopses)  # fit the vectorizer to synopses

    terms = tfidf_vectorizer.get_feature_names()

    from sklearn.metrics.pairwise import cosine_similarity
    dist = 1 - cosine_similarity(tfidf_matrix)

    return tfidf_vectorizer, tfidf_matrix, dist, terms


def k_means(tfidf_matrix):
    from sklearn.cluster import KMeans
    num_clusters = 5
    km = KMeans(n_clusters=num_clusters)
    km.fit(tfidf_matrix)

    from sklearn.externals import joblib
    joblib.dump(km, 'doc_cluster.pkl')

    return km


def create_cluster():
    titles, synopses = load_data('data')

    # perform stop-words, stemming and tokenizing
    vocab_tokenized, vocab_stemmed = tokenize_data(synopses)
    # create pandas dataframe using the above data
    vocab_frame = pd.DataFrame({'words': vocab_tokenized}, index=vocab_stemmed)
    # print ('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame') 231 tokens currently

    print(vocab_frame.head())

    # tf-idf document similarity
    tfidf_vectorizer, tfidf_matrix, dist, terms = tf_idf_calc(synopses)
    joblib.dump(dist,"dist_vals")
    # perform k-means
   # km = k_means(tfidf_matrix)
    return None, vocab_frame, terms


def use_clusters(km):
    clusters = km.labels_.tolist()
    titles, synopses = load_data('data')
    films = {'title': titles, 'synopsis': synopses, 'cluster': clusters}

    frame = pd.DataFrame(films, index=[clusters], columns=['title', 'cluster'])
    print (frame['cluster'].value_counts()) # number of films per cluster (clusters from 0 to 4))
    return titles, synopses, frame


def identify_top_words(km, num_clusters, vocab_frame, frame, terms):

    print("Top terms per cluster:")
    print()
    # sort cluster centers by proximity to centroid
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    cluster_word_dict = {}

    for i in range(num_clusters):
        # print("Cluster {} words:".format(i), end='')

        wordlist = []
        for ind in order_centroids[i, :1000]:  # replace 6 with n words per cluster
            words = vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0]
            if words is not np.nan:
                wordlist.append(words)

        print (set(wordlist))

        cluster_word_dict[i] = set(wordlist)


    joblib.dump(cluster_word_dict, "cluster_words.pkl")


def mds():
    import os  # for os.path.basename

    import matplotlib.pyplot as plt
    import matplotlib as mpl

    from sklearn.manifold import MDS


    # convert two components as we're plotting points in a two-dimensional plane
    # "precomputed" because we provide a distance matrix
    # we will also specify `random_state` so the plot is reproducible.
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

    pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

    xs, ys = pos[:, 0], pos[:, 1]
    print()
    print()


def main():
    km, vocab_frame, terms = create_cluster()
    # joblib.dump(vocab_frame, "vocab_frame.pkl")
    # joblib.dump(terms, "terms.pkl")

    # km = joblib.load("doc_cluster.pkl")
    # vocab_frame = joblib.load("vocab_frame.pkl")
    # terms = joblib.load("terms.pkl")
    titles, synopses, frame = use_clusters(km)
    identify_top_words(km, 5, vocab_frame, frame, terms)

    k = joblib.load("cluster_words.pkl")
    # for k,v in cluster_word_dict.items():
    #     print (k, v)
    #     print()
    #     print()

    c = {}
    c[0] = k[0] - k[1] - k[2] - k[3] - k[4]
    c[1] = k[1] - k[0] - k[2] - k[3] - k[4]
    c[2] = k[2] - k[1] - k[0] - k[3] - k[4]
    c[3] = k[3] - k[1] - k[2] - k[0] - k[4]
    c[4] = k[4] - k[1] - k[2] - k[3] - k[0]
    #
    for i in range(5): #new top words for clusters
        k[i] = c[i]
    #     print (k[i], '\n')
    #     print("Cluster %d titles:" % i, end='')
    #     for title in frame.ix[i]['title'].values.tolist():
    #         print(' {},'.format(title), end='')
    #     print()  # add whitespace
    #     print()  # add whitespace
    #
    # joblib.dump(k, "cluster_disjoint.pkl")
    # mds()

if __name__ == '__main__':
    main()
