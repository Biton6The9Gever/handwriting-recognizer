from data_creation.utils import create_dataset
from data_preparation.preprocess_data import generate_dataset_vectors

# create the dataset
create_dataset()

# generate the X, y vectors for training
X, y = generate_dataset_vectors()
