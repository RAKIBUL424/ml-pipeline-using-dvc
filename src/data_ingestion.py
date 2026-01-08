import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging


log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_ingestion')
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handeler = logging.FileHandler(log_file_path)
file_handeler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handeler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handeler)

def load_params(params_path:str) -> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters received from %s', params_path)
        return params
    except Exception as e:
        logger.error('Error %s:', e)
        raise



def load_data(data_path: str) -> pd.DataFrame:
    """Load data from csv file"""
    try:
        df = pd.read_csv(data_path)
        logger.debug('Data loaded from %s', data_path)
        return df
    except Exception as e:
        logger.error('Error %s:', e)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """preprocess data"""
    try:
        df.drop(columns = ['Unnamed: 2','Unnamed: 3', 'Unnamed: 4'], inplace=True)
        df.rename(columns={'v1':'target', 'v2': 'text'}, inplace=True)
        logger.debug('Data preprocessing complete')
        return df
    except Exception as e:
        logger.error('Unexcept error during preprocessing %s', e)
        raise

def save_data(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    """Save and train and test datasets"""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(raw_data_path, 'test.csv'), index=True)
        logger.debug('Train and test data save to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise


def main():
    try:
        params = load_params('params.yaml')
        test_size = params['data_ingestion']['test_size']
        data_path = "notebooks\spam.csv"
        df = load_data(data_path)
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        save_data(data_path='data', train_data=train_data, test_data=test_data)
    except Exception as e:
        logger.error('Error in data ingestion process: %s', e)
if __name__ == '__main__':
    main()