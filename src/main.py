import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from utils import create_dataset ,load_highest_accuracy_model,load_data_vectors


# create the dataset
create_dataset()

# load the X, y vectors for training
X,y = load_data_vectors()

load_highest_accuracy_model()



