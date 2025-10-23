import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from utils import create_dataset
from data_preparation.preprocess_data import generate_dataset_vectors
from models.architectures.cnn_models.vgg19_model import build_model_vgg19
from models.architectures.cnn_models.mobilenetv2_model import build_model_mobilenetv2

# create the dataset
create_dataset()

# generate the X, y vectors for training
X, y = generate_dataset_vectors()


build_model_mobilenetv2(X, y)
# build_model_vgg19(X, y)



