from src.utils.logging_config import setup_logging
from src.data_loader import load_data
import yaml

def load_config(path = "config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    setup_logging()
    config = load_config()

    load_data()

