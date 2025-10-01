import pandas as pd
import os

def read_titanic_data(directory, filename):
    """ Read csv file and convert to pandas DataFrame

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

