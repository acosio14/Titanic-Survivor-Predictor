import pickle

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)


class Model:
    def __init__(self, model):
        """init_model = LogisticRegression() or tree.DecisionTreeClassifier()"""
        self.model = model

    def train(self, X_train, y_train, X_test):
        # Initialize Model, fit, and predict
        model = self.model
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        return model, y_pred

    def evaluate_model(self, X_test, y_test, y_pred):
        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Confusion Matrix
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Classification Report
        class_report = classification_report(y_test, y_pred)

        # ROC Curve and AUC
        y_prob = self.model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)

        return accuracy, conf_matrix, class_report, fpr, tpr

    def display_metrics(
        self, accuracy, confusion_matrix, classification_report, fpr, tpr
    ):
        print(f"Model: {str(self.model)}\n")

        print(f"Model Accuracy: {round(accuracy * 100, 3)}%\n")

        print(f"Classification Report:\n {classification_report}")

        # Display Confusion Matrix
        ConfusionMatrixDisplay(confusion_matrix=confusion_matrix).plot()
        plt.title("Confusion Matrix")
        plt.show()

        # Display ROC Curve
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.show()

    def save_model(self, filepath):
        if not filepath.endswith(".pkl"):
            print(f"Error: {filepath} not .pkl")
            return
        try:
            with open(filepath, "wb") as file:
                pickle.dump(self.model, file)

            model_name = str(self.model).split("()")[0]
            print(f"Saved model '{model_name}' as {filepath}")
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
