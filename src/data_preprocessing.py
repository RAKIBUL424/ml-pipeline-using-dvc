import os
import logging
import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')



log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_preprocessing')
logger.setLevel("DEBUG")

console_handeler = logging.StreamHandler()
console_handeler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handeler = logging.FileHandler(log_file_path)
file_handeler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handeler.setFormatter(formatter)
file_handeler.setFormatter(formatter)

logger.addHandler(console_handeler)
logger.addHandler(file_handeler)


def transform_text(text):
    ps = PorterStemmer()
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word.isalnum()]
    text = [ word for word in text if word not in stopwords.words('english')]
    text = [ps.stem(word) for word in text]
    return " ".join(text)

def preprocess_df(df, text_column='text', target_column='target'):
    try:
        logger.debug("Starting preprocessing of dataframe")
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug("Label encoding completed")

        df = df.drop_duplicates(keep='first')
        logger.debug("Duplicates removed from dataframe")

        df.loc[:, text_column] = df[text_column].apply(transform_text)
        logger.debug("Text transformation completed")

        return df
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise

def main():
    try:
        train_data = pd.read_csv('data/raw/train.csv')
        test_data = pd.read_csv('data/raw/test.csv')
        logger.debug("Data loaded successfully")

        train_processed_data = preprocess_df(train_data, text_column='text', target_column='target')
        test_processed_data = preprocess_df(test_data, text_column='text', target_column='target')

        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path, 'train_preprocessed.csv'), index=False)
        test_processed_data.to_csv(os.path.join(data_path, 'test_preprocessed.csv'), index=False)
        logger.debug("Preprocessed data saved successfully : %s", data_path)
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    main()