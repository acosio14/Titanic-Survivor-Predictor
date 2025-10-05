from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

def logistic_regression(X_train, y_train, X_test, y_test):
    
    # Initialize Model, fit, and predict
    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    if accuracy >= .8:
        print(f"Model Accuracy: {round(accuracy * 100,3)}%")
        return model
    else:
        print(f"Model Accuracy: {round(accuracy * 100,3)}%")
        print(f"Not returning model.")

def decision_tree():
    ...

def random_forest():
    ...

def gradient_boosting():
    ...

def svm():
    ...

def evaluate_model():
    ...

def save_model(model, filepath):
    
    if not filepath.endswith(".pkl"):
        print(f"Error: {filepath} not .pkl")
        return
    try:
        with open(filepath,'wb') as file:
            pickle.dump(model, file)

        print(f"Saved {model} in {filepath}")
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
