import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import plot_confusion_matrix


def read_corpus(fake_corpus, truth_corpus):
    target = []
    with open(fake_corpus, 'r', encoding='utf-8') as f, open(truth_corpus, 'r', encoding='utf-8') as t:
        fake = f.readlines()
        truth = t.readlines()

    corpus = fake + truth
    target = target + len(fake) * [0]
    target = target + len(truth) * [1]

    return corpus, target


def feature_extraction(trainX, vectorizer_type, max_features=10000):
    if vectorizer_type == "tf":
        vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=max_features, min_df=1)
    else:
        vectorizer = TfidfVectorizer(lowercase=False, ngram_range=(1, 3), max_features=max_features, min_df=3)

    trainX = vectorizer.fit_transform(trainX)
    return trainX, vectorizer


def plot_clusters(X):
    NUMBER_OF_CLUSTERS = 2
    km = KMeans(
        n_clusters=NUMBER_OF_CLUSTERS,
        init='k-means++',
        max_iter=500)
    km.fit(X)

    # First: for every document we get its corresponding cluster
    clusters = km.predict(X)

    # We train the PCA on the dense version of the tf-idf.
    pca = PCA(n_components=2)
    two_dim = pca.fit_transform(X.todense())

    scatter_x = two_dim[:, 0]  # first principle component
    scatter_y = two_dim[:, 1]  # second principle component

    plt.style.use('ggplot')

    fig, ax = plt.subplots()
    fig.set_size_inches(20, 10)

    # color map for NUMBER_OF_CLUSTERS we have
    cmap = {0: 'green', 1: 'blue'}

    # group by clusters and scatter plot every cluster
    # with a colour and a label
    for group in np.unique(clusters):
        ix = np.where(clusters == group)
        ax.scatter(scatter_x[ix], scatter_y[ix], c=cmap[group], label=group)

    ax.legend()
    plt.xlabel("PCA 0")
    plt.ylabel("PCA 1")
    plt.show()


def plot_confusion(clf, X_test, y_test):
    np.set_printoptions(precision=2)
    class_names = ["fake", "truth"]
    # Plot non-normalized confusion matrix
    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(clf, X_test, y_test,
                                     display_labels=class_names,
                                     cmap=plt.cm.Blues)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

    plt.show()
