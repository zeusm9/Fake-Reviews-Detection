import utils
import numpy as np
from sklearn.utils import shuffle
from sklearn import naive_bayes
from sklearn.model_selection import train_test_split,StratifiedKFold, cross_validate
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    fbeta_score,
    make_scorer,
)

def train_svm(X_train, y_train,vectorizer_type, c: float = 1.0, max_iter: int = 1000):
    X_train = utils.feature_extraction(X_train, max_features=10000,vectorizer_type=vectorizer_type)
    svm_model = LinearSVC(C=c, max_iter=max_iter, dual=False)
    print("Cross Validation ...")
    train_linear(svm_model,X_train,y_train)
    print("\nFitting ...")
    svm_model.fit(X_train,y_train)
    return svm_model

def train_NB(X_train, y_train,vectorizer_type):
    X_train = utils.feature_extraction(X_train, max_features=10000,vectorizer_type=vectorizer_type)
    naive_model = naive_bayes.MultinomialNB()
    print("\nCross validation ...")
    train_linear(naive_model,X_train,y_train)
    print("\nFitting")
    naive_model.fit(X_train,y_train)
    return naive_model

def train_linear(model, X_train, y_train ):
    scoring = {
        "accuracy":make_scorer(accuracy_score),
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
    X, y = utils.read_corpus()
    X, y = shuffle(X,y)
    train_svm(X,y,"tf")
    train_NB(X,y,"tf")

if __name__ == "__main__":
    main()
