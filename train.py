import config as c
import utils
import evaluation
import numpy as np
import random
from sklearn import naive_bayes
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    fbeta_score,
    make_scorer,
)


def train_svm(X_train, y_train, vectorizer, type="normal", c: float = 1.0, max_iter: int = 1000,
              max_features: int = 1000):
    """
        Used to train a svm classifier
        :param X_train: input train
        :param y_train: label train
        :param vectorizer: type of vectorizer
        :param type : type of svm (linear, not linear)
        :param c: regularization parameter
        :param max_iter : maximum iterations
        :param max_features : maximum number of features
        :return: svm model, features vectorizer
    """
    X_train, vectorizer = utils.feature_extraction(X_train, max_features=max_features, vectorizer_type=vectorizer)
    if type == "normal":
        svm_model = SVC(C=c, max_iter=max_iter)
    else:
        svm_model = LinearSVC(C=c, max_iter=max_iter, dual=False)
    print("Cross Validation ...")
    train_linear(svm_model, X_train, y_train)
    print("\nFitting ...")
    svm_model.fit(X_train, y_train)
    return svm_model, vectorizer


def train_NB(X_train, y_train, vectorizer, max_features: int = 1000):
    """
        Used to train a naive bayes classifier
        :param X_train: input train
        :param y_train: label train
        :param vectorizer: type of vectorizer
        :param max_features : maximum number of features
        :return: naive bayes model, features vectorizer
    """
    X_train, vectorizer = utils.feature_extraction(X_train, max_features=max_features, vectorizer_type=vectorizer)
    naive_model = naive_bayes.MultinomialNB()
    print("\nCross validation ...")
    train_linear(naive_model, X_train, y_train)
    print("\nFitting")
    naive_model.fit(X_train, y_train)
    return naive_model, vectorizer


def train_knn(X_train, y_train, vectorizer_type, max_features: int = 1000):
    """
        Used to train a k-nn classifier
        :param X_train: input train
        :param y_train: label train
        :param vectorizer_type: type of vectorizer
        :param max_features : maximum number of features
        :return: naive bayes model, features vectorizer
    """

    X_train, vectorizer = utils.feature_extraction(X_train, max_features=max_features, vectorizer_type=vectorizer_type)
    model = KNeighborsClassifier(n_neighbors=10)
    train_linear(model, X_train, y_train)
    model.fit(X_train, y_train)
    return model, vectorizer


def train_linear(model, X_train, y_train):
    """
        Used to evaluate and compute metrics with k-fold cross validation
        :param model: model to evaluate
        :param X_train: input train
        :param y_train: label train
        :return:
    """
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
    """
        Used to evaluate and compute metrics with k-fold cross validation
        :param train_x: input train
        :param train_y: label train
        :param estimator: trained model
        :param cv : type of cross validation
        :param scoring: metrics to display
        :return: mean of the metrics
    """
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
    MAX_FEATURES = 50000
    SVM_TYPE = "linear"
    C = 1.1
    X, y = utils.read_corpus(c.FAKE_CORPUS, c.TRUTH_CORPUS)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)
    model1, vectorizer = train_svm(X_train, y_train, VECTORIZER_TYPE, max_features=MAX_FEATURES, type=SVM_TYPE, c=C,
                                   max_iter=10000)
    evaluation.evaluate_linear(model1, vectorizer, X_test, y_test)
    utils.plot_clusters(X_test, vectorizer)


if __name__ == "__main__":
    main()
