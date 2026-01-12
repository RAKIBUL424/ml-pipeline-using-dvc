import logging
import os


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



if __name__ == "__main__":
    pass