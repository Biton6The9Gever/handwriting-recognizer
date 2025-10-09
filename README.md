# handwriting-recognizer
A machine learning model to recognize and classify my handwriting

## Folder structure
```text
data/
│   ├── raw/
│   │   ├── raw_a.jpg
│   │   ├── raw_b.jpg
│   │   ├── ...
│   │   └── raw_z.jpg
│   ├── processed/              # hold the data after processing
│   └── dataset.csv    
src/
│   ├── data_creation/
│   │   ├── augment_images.py
│   │   ├── crop_images.py
│   │   └── utils.py
│   ├── data_preparation/
│   │   └── preprocess_data.py
│   ├── flush_data.py
│   └── main.py
README.md
```


## List of things I've learn in this project
- working with the cv2 library
- data augmentation
- machine learning pipelines
- and more!
