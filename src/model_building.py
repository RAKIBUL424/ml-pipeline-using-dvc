import os
import logging
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
import yaml
import numpy as np

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_building')
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir, 'model_building.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter) 
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)
def load_params(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug(f"Parameters loaded successfully from {file_path}")
        return params
    except Exception as e:
        logger.error(f"Error loading parameters: {e}")
        raise

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug(f"Data loaded successfully from {file_path} with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> RandomForestClassifier:
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("Number of samples in X_train and y_train must be the same")
        logger.debug(f"Training model with parameters: {params}")
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])

        logger.debug('Model training started with %d samples', X_train.shape[0])
        clf.fit(X_train, y_train)
        logger.debug('Model training completed successfully')
        return clf
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise


def save_model(model, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
            logger.debug(f"Model saved successfully to {file_path}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

def main():
    try:
        params = load_params('params.yaml')['model_building']
        train_data = load_data('data/processed/train_tfidf.csv')
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        clf = train_model(X_train, y_train, params)

        model_save_path = 'models/model.pkl'
        save_model(clf, model_save_path)
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise


if __name__ == "__main__":
    main()