import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from utils import load_highest_accuracy_model,predict_image ,create_dataset ,recreate_data_folder,PROCESSED_DATA_PATH,gradio_ui
from models.rnn_model import build_model_rnn
from data_creation.sentences_generator import generate_sentences


#create_dataset()

gradio_ui()

#recreate_data_folder()






