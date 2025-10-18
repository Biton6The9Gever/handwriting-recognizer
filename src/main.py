from data_creation.utils import create_dataset
from data_preparation.preprocess_data import generate_dataset_vectors
from models.architectures.cnn_models.vgg19_model import build_model
import numpy as np
# create the dataset
create_dataset()

# generate the X, y vectors for training
X, y = generate_dataset_vectors()


build_model(X, y)



