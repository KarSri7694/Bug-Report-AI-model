import pandas as pd
import numpy as np
import logging
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

stop_words = set(stopwords.words('english'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    Parameters:
    file_path (str): The path to the CSV file.
    
    Returns:
    pd.DataFrame: The loaded data as a DataFrame.
    """
    return pd.read_csv(file_path)

def view_data(df: pd.DataFrame) -> None:
    """
    Display the first few rows of the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to view.
    num_rows (int): The number of rows to display (default is 5).
    """
    print(df.head())
 
        
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data by handling missing values and encoding categorical variables.

    Parameters:
    df (pd.DataFrame): The DataFrame to preprocess.

    Returns:
    pd.DataFrame: The preprocessed DataFrame.
    """
    

    # Handle missing values (example: fill with mean for numeric columns)
    features = ['Bug Id', 'Priority Id', 'Priority Name', 'Status', 'Resolution', 'Description', 'Assigned To', 'Fix Version', 'Bug Creation Date', 'Components']
    for column in df.columns:
        if column not in features:
            df.drop(column, axis=1, inplace=True)

    logging.info(f"Selected features: {features}")
    logging.info(f"No. of features: {len(features)}")
    logging.info("Shape before removing rows with empty values: {}".format(df.shape))
    
    df.dropna(inplace=True)

    logging.info(f"Shape after removing rows with empty values: {df.shape}")

    # Remove rows where 'Issue key' doesn't match the pattern SRCTREEWIN-<number>
    pattern = r'^SRCTREEWIN-\d+$'
    # df = df[df['Issue key'].astype(str).str.match(pattern, na=False)]

    logging.info(f"Shape after removing rows not matching 'Issue key' pattern: {df.shape}")

    def convert_priority_to_numeric(priority):
        priority_mapping = {
            'Highest': 4,
            'High': 3,
            'Medium': 2,
            'Low': 1
        }
        return priority_mapping.get(priority, np.nan)
    #Now we will preprocess the text data in 'Summary' and 'Description' columns
    def preprocess_text(text):
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Tokenize the text
        tokens = word_tokenize(text)
        # Remove stop words
        tokens = [word for word in tokens if word not in stop_words]
        return ' '.join(tokens)
    
    df['Description'] = df['Description'].apply(preprocess_text)
    
    #see new data
    print(df.head())
    #writing new data to csv file
    df.to_csv('dataset/preprocessed_data.csv', index=False)
    return df
    


if __name__ == "__main__":
    file_path = 'dataset/apache_LUCENE_1_7474.csv'  
    data = load_data(file_path)
    view_data(data)
    preprocess_data(data)
    