from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

def logistic_regression(X_train, y_train, X_test, y_test):
    
    # Initialize Model, fit, and predict
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)


    return round(accuracy * 100,3)



def save_model(model, filename):
    
    if not filename.endswith(".pkl"):
        print(f"Error: {filename} not .pkl")
        return
    try:
        with open(filename,'wb') as file:
            pickle.dump(model, file)

        print(f"Saved {model} in {filename}")
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
