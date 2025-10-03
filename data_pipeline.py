import pandas as pd
import os

from sklearn.preprocessing import LabelEncoder

def read_titanic_data(directory, filename):
    """ Read csv file and convert to pandas DataFrame (Data Acquisition)

    Ags:
        directory: path of file
        filename: name of file
    
    Returns:
        titanic_df: DataFrame of csv data
    
    """
    try:
        full_file_path = os.join(directory,filename)
        titanic_df = pd.read_csv(full_file_path)
    except Exception as e:
        print(f"Error in read_titanic_data function: {e}")
    
    return titanic_df

def clean_titanic_data(titanic_df):
    """ Clean DataFrame data."""
    
    # Drop irrelevant columns
    drop_values = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    train_df = titanic_df.drop(drop_values, axis=1)   

    # Handle Missing Data
    train_clean_df = train_df[train_df['Age'].notna()].dropna()

    # Change categorical columns to numeric
    # To-Do: 
    # - Figure out if I need to output original categories or if stored in LableEncoder somehow
    # - Try-Catch Error detectioin
    # - Docstring
    train_clean_df['Sex'] = LabelEncoder().fit_transform(train_clean_df['Sex'])
    train_clean_df['Embarked'] = LabelEncoder().fit_transform(train_clean_df['Embarked'])

    return train_clean_df
    
    # To-Do:
    # - Create Feature Enginnering File