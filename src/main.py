from data_creation.utils import create_dataset
from data_preparation.preprocess_data import generate_dataset_vectors
from data_preparation.data_analysis import visualize_all

# create the dataset
create_dataset()

# generate the X, y vectors for training
X, y = generate_dataset_vectors()

visualize_all(X, y, num_samples=9)



