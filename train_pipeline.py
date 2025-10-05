from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve, 
    auc,
    RocCurveDisplay,
)
import pickle

class Model:

    def __init__(self, init_model):
        """ init_model = LogisticRegression() or tree.DecisionTreeClassifier()"""
        self.init_model = init_model


    def train(self, X_train, y_train, X_test):
        
        # Initialize Model, fit, and predict
        model = self.init_model
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        
        return model, y_pred

    # To-Do: After keeping all models above 80%. Generate table of each with their respective accuracy.
    def evaluate_model(self, X_test, y_test, y_pred):
        
        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {round(accuracy * 100,3)}%")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)

        # Classification Report
        print(classification_report(y_test, y_pred))
        
        # ROC Curve and AUC
        y_prob = self.init_model.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        # Display ROC Curve
        roc_disp = RocCurveDisplay.from_preditctions(y_test, y_pred)

    def save_model(self, model, filepath):
        
        if not filepath.endswith(".pkl"):
            print(f"Error: {filepath} not .pkl")
            return
        try:
            with open(filepath,'wb') as file:
                pickle.dump(model, file)

            print(f"Saved {model} in {filepath}")
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
