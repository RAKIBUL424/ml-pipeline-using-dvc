import logging
import os
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer


log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('feature_engg')
logger.setLevel("DEBUG")

console_handeler = logging.StreamHandler()
console_handeler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir, 'feature_engg.log')
file_handeler = logging.FileHandler(log_file_path)
file_handeler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handeler.setFormatter(formatter)
file_handeler.setFormatter(formatter)

logger.addHandler(console_handeler)
logger.addHandler(file_handeler)

def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug(f"Parameters loaded successfully from {params_path}")
        return params
    except Exception as e:
        logger.error(f"Error loading parameters from {params_path}: {e}")
        raise

def load_data(file_path: str) -> pd.DataFrame:
    logger.info(f"Loading data from {file_path}")
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug(f"Data loaded successfully with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
    

def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)
        X_train = train_data['text'].values
        y_train = train_data['target'].values
        X_test = test_data['text'].values
        y_test = test_data['target'].values

        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['target'] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['target'] = y_test
        logger.debug(f"TF-IDF applied successfully with max_features={max_features}")
        return train_df, test_df
    except Exception as e:
        logger.error(f"Error applying TF-IDF: {e}")
    
def save_data(df: pd.DataFrame, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug(f"Data saved successfully to {file_path}")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise

def main():
    try:
        max_features = load_params('./params.yaml')['feature_engg']['max_features']
        train_data = load_data('./data/interim/train_preprocessed.csv')
        test_data = load_data('./data/interim/test_preprocessed.csv')

        train_df, test_df = apply_tfidf(train_data, test_data, max_features=max_features)

        save_data(train_df, os.path.join('./data','processed','train_tfidf.csv'))
        save_data(test_df, os.path.join('./data','processed','test_tfidf.csv'))
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise



if __name__ == "__main__":
    main()