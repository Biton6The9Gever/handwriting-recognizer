import sys
import os
import cv2
import time
from tqdm import tqdm
data_creation_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_creation'))
sys.path.append(data_creation_path)
import utils 
import pandas as pd
import numpy as np

def generate_dataset_vectors():
    print('[START] Generating dataset vectors')

    start_time = time.time()

    df = pd.read_csv(utils.DATA_CSV)
    X = df['label'].values.reshape(-1, 1)
    y = []

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # tqdm progress bar
    for path in tqdm(df['path'], desc="[PROGRESS] Processing images", ncols=80):
        full_path = os.path.join(project_root, path)
        img = cv2.imread(full_path)
        if img is None:
            print(f"[WARN] Could not read: {full_path}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64, 64))
        img = img / 255.0
        y.append(img)

    y = np.array(y, dtype=np.float32)

    utils.recreate_data_folder()

    elapsed = time.time() - start_time
    print(f"\n[END] Finished data vectors {len(y)} images in {elapsed:.2f} seconds. \n")

    return X, y






