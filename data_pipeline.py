import pandas as pd
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def read_titanic_data(directory, filename):
    """Read csv file and convert to pandas DataFrame (Data Acquisition)

    Args:
        directory: path of file
        filename: name of file

    Returns:
        titanic_df: DataFrame of csv data

    """
    try:
        full_file_path = os.path.join(directory, filename)
        titanic_df = pd.read_csv(full_file_path)
    except Exception as e:
        print(f"Error in read_titanic_data function: {e}")

    return titanic_df


def clean_titanic_data(titanic_df):
    """Clean DataFrame data."""

    # Drop irrelevant columns
    drop_values = ["PassengerId", "Name", "Ticket", "Cabin"]
    train_df = titanic_df.drop(drop_values, axis=1)

    # Handle Missing Data
    train_clean_df = train_df[train_df["Age"].notna()].dropna()

    # Remove Zero Fare
    fare_is_not_zero = train_clean_df["Fare"] != 0
    train_clean_df = train_clean_df[fare_is_not_zero]

    sex_categories = train_clean_df["Sex"]
    embarked_categories = train_clean_df["Embarked"]

    le = LabelEncoder()
    train_clean_df["Sex"] = le.fit_transform(sex_categories)
    train_clean_df["Embarked"] = le.fit_transform(embarked_categories)

    return train_clean_df, (sex_categories, embarked_categories)


def prepare_titanic_data(titanic_df, test_size, random_state):
    """Prepare data by normalizing, scaling, and divide features/target."""

    # standard scaling -> x_new = X - mu / sigma
    age = titanic_df["Age"]
    fare = titanic_df["Fare"]

    titanic_df["Age"] = (age - age.mean()) / age.std()
    titanic_df["Fare"] = (fare - fare.mean()) / fare.std()

    # Split features(X) and target(y)
    X = titanic_df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
    y = titanic_df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return titanic_df, (X_train, X_test, y_train, y_test)
