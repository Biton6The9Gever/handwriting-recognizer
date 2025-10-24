# handwriting-recognizer
A machine learning model to recognize and classify my handwriting

## Folder structure
```text
data/
├── raw/                        # original, unprocessed data
│   ├── raw_a.jpg
│   ├── raw_b.jpg
│   ├── ...
│   └── raw_z.jpg
├── processed/                  # data after preprocessing or augmentation
│   └── ...
├── dataset_vectors.npz         # hold the dataset vectors after processing 
└── dataset.csv
src/
├── data_creation/
│   ├── augment_images.py
│   └── crop_images.py
├── data_preparation/
│   ├── data_analysis.py
│   └── preprocess_data.py
├── models/
│   ├── cnn_models/
│   │   ├── vgg19_model.py
│   │   └── mobilenetv2_model.py
│   └── rnn_model.py
├── saved_models/
│   ├── models_info_.json
│   ├── {model_name}_{date}.keras
│   └── ...
├── flush_data.py
├── main.py
└── utils.py
README.md
```


## List of things I've learn in this project
- working with the cv2 library
- data augmentation
- machine learning pipelines
- and more!
