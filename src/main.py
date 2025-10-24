import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from utils import load_highest_accuracy_model,load_data_vectors
from models.rnn_model import build_model_rnn


# load the X, y vectors for training
X,y = load_data_vectors()
build_model_rnn(X, y)
#load_highest_accuracy_model()



