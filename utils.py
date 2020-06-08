from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def read_corpus(fake_corpus,truth_corpus):
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
        vectorizer = CountVectorizer(ngram_range=(2, 2), max_features=max_features)
    else:
        vectorizer = TfidfVectorizer(lowercase=False, ngram_range=(2, 2), max_features=max_features)

    trainX = vectorizer.fit_transform(trainX)
    return trainX,vectorizer
