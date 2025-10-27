import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from utils import load_highest_accuracy_model,predict_image
from models.rnn_model import build_model_rnn

model=load_highest_accuracy_model()

folder_path='D:/Biton/VisualStudio/.VSC/Python/ML_Project/data/samples'

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    
    if os.path.isfile(file_path):
        try:
            result = predict_image(model, file_path)
            print(f"{filename}: {result}")
        except Exception as e:
            print(f"[WARN] Error with {filename}: {e}")





