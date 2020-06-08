from sklearn.metrics import accuracy_score, precision_recall_curve
from sklearn.metrics import classification_report

def evaluate_linear(model,vectorizer,X_test,y_test):
    X_test = vectorizer.transform(X_test)
    y_pred = model.predict(X_test)
    print("Accuracy Score:", accuracy_score(y_pred, y_test) * 100)
    cr = classification_report(y_test, y_pred, [1, 0])
    print("Classification report : \n", cr)