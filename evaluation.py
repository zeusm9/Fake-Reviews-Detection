from sklearn.metrics import accuracy_score, precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt


def evaluate_linear(model, vectorizer, X_test, y_test):
    X_test = vectorizer.transform(X_test)
    y_pred = model.predict(X_test)
    print("Accuracy Score:", accuracy_score(y_pred, y_test) * 100)
    cr = classification_report(y_test, y_pred, [1, 0])
    print("Classification report : \n", cr)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix : \n", cm)
    ax = sns.heatmap(cm, annot=True, fmt='g')
    plt.show()
    print_roc(y_test, y_pred)


def print_roc(y_test, preds):
    n_classes = 2
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test, preds)
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure()
    plt.plot(fpr[1], tpr[1])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.show()
