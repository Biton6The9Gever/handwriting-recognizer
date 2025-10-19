import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed")
DATA_CSV = os.path.join(DATA_DIR, "dataset.csv")