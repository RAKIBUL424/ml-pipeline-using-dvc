import os
import logging
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import yaml
import json
from dvclive import Live

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_evaluation')
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir, 'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

fomatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(fomatter)
file_handler.setFormatter(fomatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug(f"Parameters loaded successfully from {params_path}")
        return params
    except Exception as e:
        logger.error(f"Error loading parameters from {params_path}: {e}")
        raise

def load_model(model_path: str):
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise

def load_data(data_path: str):
    try:
        data = pd.read_csv(data_path)
        logger.debug(f"Data loaded successfully from {data_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {data_path}: {e}")
        raise

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precission = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            "accuracy": accuracy,
            "precision": precission,
            "recall": recall,
            "roc_auc": roc_auc
        }

        logger.debug(f"Model evaluation completed successfully with metrics: {metrics_dict}")
        return metrics_dict
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise

def save_metrics(metrics: dict, output_path: str):
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug(f"Metrics saved successfully to {output_path}")
    except Exception as e:
        logger.error(f"Error saving metrics to {output_path}: {e}")
        raise
def main():
    try:
        params = load_params(params_path="params.yaml")
        model = load_model('./models/model.pkl')
        test_data = load_data('./data/processed/test_tfidf.csv')
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        metrics = evaluate_model(model, X_test, y_test)

        with Live(save_dvc_exp=True) as live:
            live.log_metric("accuracy", metrics['accuracy'])
            live.log_metric("precision", metrics['precision'])
            live.log_metric("recall", metrics['recall'])
            live.log_metric("roc_auc", metrics['roc_auc'])
            live.log_params(params)

        save_metrics(metrics, output_path="reports/metrics.json")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise



if __name__ == "__main__":
    main()
