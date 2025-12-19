import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from utils import load_highest_accuracy_model,predict_image ,create_dataset ,recreate_data_folder
from models.rnn_model import build_model_rnn

recreate_data_folder()






