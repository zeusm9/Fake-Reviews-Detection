import config as c
import utils
import evaluation
import numpy as np
from sklearn import naive_bayes
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    fbeta_score,
    make_scorer,
)


def train_svm(X_train, y_train,vectorizer, c: float = 1.0, max_iter: int = 1000):
    X_train,vectorizer = utils.feature_extraction(X_train, max_features=10000, vectorizer_type=vectorizer)
    svm_model = LinearSVC(C=c, max_iter=max_iter, dual=False)
    print("Cross Validation ...")
    train_linear(svm_model, X_train, y_train)
    print("\nFitting ...")
    svm_model.fit(X_train, y_train)
    return svm_model,vectorizer


def train_NB(X_train, y_train,vectorizer):
    X_train,vectorizer = utils.feature_extraction(X_train, max_features=10000, vectorizer_type=vectorizer)
    naive_model = naive_bayes.MultinomialNB()
    print("\nCross validation ...")
    train_linear(naive_model, X_train, y_train)
    print("\nFitting")
    naive_model.fit(X_train, y_train)
    return naive_model,vectorizer


def train_knn(X_train, y_train, vectorizer_type):
    X_train,vectorizer = utils.feature_extraction(X_train, max_features=10000, vectorizer_type=vectorizer_type)
    model = KNeighborsClassifier(n_neighbors=10)
    train_linear(model, X_train, y_train)
    model.fit(X_train, y_train)
    return model,vectorizer


def train_linear(model, X_train, y_train):
    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score, pos_label=1),
        "recall": make_scorer(recall_score, pos_label=1),
        "f1": make_scorer(fbeta_score, beta=1, pos_label=1)
    }

    kfold = StratifiedKFold(5, True, 1)
    accuracy, precision, recall, fscore = validation(
        X_train, y_train, model, kfold, scoring
    )
    print("Accuracy: {0:.2f}".format(accuracy))
    print("Precision: {0:.2f}".format(precision))
    print("Recall: {0:.2f}".format(recall))
    print("F1 score: {0:.2f}".format(fscore))


def validation(train_x, train_y, estimator, cv, scoring):
    scores = cross_validate(
        estimator, train_x, train_y, cv=cv, scoring=scoring, n_jobs=-1
    )
    return (
        np.mean(scores["test_accuracy"]),
        np.mean(scores["test_precision"]),
        np.mean(scores["test_recall"]),
        np.mean(scores["test_f1"]),
    )


def main():
    VECTORIZER_TYPE = "tf-idf"

    X, y = utils.read_corpus(c.FAKE_CORPUS, c.TRUTH_CORPUS)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,shuffle=True)
    #model,vectorizer = train_svm(X_train, y_train, VECTORIZER_TYPE)
    model,vectorizer = train_knn(X_train,y_train,VECTORIZER_TYPE)
    evaluation.evaluate_linear(model, vectorizer ,X_test, y_test)


if __name__ == "__main__":
    main()
