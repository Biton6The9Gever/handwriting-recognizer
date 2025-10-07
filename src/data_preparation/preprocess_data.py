import sys
import os
import cv2
data_creation_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_creation'))
sys.path.append(data_creation_path)
import utils 
import pandas as pd
import numpy as np

df = pd.read_csv(utils.DATA_CSV)
X = df['label'].values
y = []
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
for path in df['path']:
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

print("X shape:", X.shape)
print("y shape:", y.shape)





