import os
import cv2
import json
import shutil
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from collections import Counter
import tensorflow.keras as keras  # type: ignore

# ==== CONSTANTS ====
CHAR_AMOUNT = 26
IMAGES_AMOUNT = 300
IMAGE_SIZE = (64, 64)
AUGMENTATIONS_AMOUNT = 4 

# ==== PATH SETUP ====
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed")
DATA_CSV = os.path.join(DATA_DIR, "dataset.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT,"src", "saved_models")



# ==== FUNCTIONS ====

def recreate_data_folder(folder_path=PROCESSED_DATA_PATH):
    try:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path, ignore_errors=True)
        os.makedirs(folder_path, exist_ok=True)
        open(os.path.join(folder_path, ".gitkeep"), "a").close()
        print(f"[WARN] Recreated data folder: {folder_path}")
    except Exception as e:
        print(f"[ERROR] Failed to recreate folder {folder_path}: {e}")


def get_image_size(image_path):
    """Return the dimensions of the image at the given path. for debugging purposes"""
    img = cv2.imread(image_path)
    return img.shape if img is not None else None


def get_images_size_distribution(image_root):
    """Get the distribution of image sizes in the dataset. for debugging purposes"""
    sizes = []
    for i in range(CHAR_AMOUNT):
        letter = chr(ord("A") + i)
        for index in range(1, IMAGES_AMOUNT * (AUGMENTATIONS_AMOUNT + 1) + 1):
            image_path = os.path.join(image_root, f"{letter}_{index:04d}.jpg")
            if os.path.exists(image_path):
                size = get_image_size(image_path)
                if size is not None:
                    sizes.append(size)
                else:
                    print(f"[WARN] Failed to read image: {image_path}")

    size_counts = Counter(sizes)
    total = sum(size_counts.values())
    for size, count in size_counts.items():
        print(f"Size: {size}, Count: {count}")
    print(f"Total images processed: {total}")


def resize_base_images():
    print("\n[START] Resizing base images...")

    total = CHAR_AMOUNT * IMAGES_AMOUNT
    count = 0

    for i, index in tqdm(itertools.product(range(CHAR_AMOUNT), range(1, IMAGES_AMOUNT + 1)),
                         total=total, desc="[PROGRESS] Images", ncols=80):
        letter = chr(ord("A") + i)
        image_path = os.path.join(PROCESSED_DATA_PATH, f"{letter}_{index:04d}.jpg")
        image = cv2.imread(image_path)
        if image is not None:
            resized_image = cv2.resize(image, IMAGE_SIZE)
            cv2.imwrite(image_path, resized_image)
            count += 1
        else:
            print(f"[WARN] Resizing failed for image {image_path}")

    print(f"[END] Resized {count}/{total} images successfully.")


def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')


def create_csv_file():
    letters = [chr(ord('A') + i) for i in range(CHAR_AMOUNT)]
    indices = range(1, IMAGES_AMOUNT * (AUGMENTATIONS_AMOUNT + 1) + 1)

    # build absolute paths
    abs_paths = [os.path.join(PROCESSED_DATA_PATH, f"{letter}_{index:04d}.jpg")
                 for letter in letters for index in indices]

    # convert to relative paths (relative to the project root)
    rel_paths = [os.path.relpath(p, PROJECT_ROOT) for p in abs_paths]

    labels = [letter for letter in letters for _ in indices]

    assert len(labels) == len(rel_paths), "[WARN] Labels and paths must be the same length"

    df = pd.DataFrame({'label': labels, 'path': rel_paths})
    df.to_csv(DATA_CSV, index=False)
    print(f"\n[INFO] Created CSV file at {DATA_CSV} with {len(df)} entries.")


def create_dataset():
    """
    Full dataset creation workflow:
    1. Crop images
    2. Resize images
    3. Augment images
    4. Create CSV
    """
    from data_creation.crop_images import crop_all_letters
    from data_creation.augment_images import augment_images

    print("[START] Starting dataset creation")
    # 1. Crop letters into individual boxes
    crop_all_letters(limit=CHAR_AMOUNT)

    # 2. Resize base images
    resize_base_images()

    # 3. Augment images
    augment_images()

    # 4. Create CSV
    create_csv_file()

    print("\n[END] Dataset creation complete \n")
    
def save_model(model, accuracy, model_name="vgg19"):
    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.path.dirname(__file__), "saved_models")
    os.makedirs(models_dir, exist_ok=True)

    # Timestamp for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Use modern Keras format
    model_filename = f"{model_name}_{timestamp}.keras"
    model_path = os.path.join(models_dir, model_filename)

    # Save the model in the new Keras format
    model.save(model_path)

    # Metadata file
    meta_path = os.path.join(models_dir, "models_info.json")

    # Load existing metadata if available
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            models_info = json.load(f)
    else:
        models_info = []

    # Add new model entry
    models_info.append({
        "name": model_name,
        "path": model_filename,
        "accuracy": float(accuracy),
        "timestamp": timestamp
    })

    # Save metadata to JSON
    with open(meta_path, "w") as f:
        json.dump(models_info, f, indent=4)

    print(f"[SAVED] Model saved at: {model_path}")
    print(f"[INFO] Metadata updated at: {meta_path}")
    
def load_model(file_path):
    # Load the model
    model = keras.models.load_model(os.path.join(MODEL_DIR, file_path))
    return model

def load_highest_accuracy_model():
    meta_path = os.path.join(MODEL_DIR, "models_info.json")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"No metadata file found at {meta_path}")

    with open(meta_path, "r") as f:
        models_info = json.load(f)

    if not models_info:
        raise ValueError("No models found in metadata.")

    # Find the model with the highest accuracy
    best_model_info = max(models_info, key=lambda x: x["accuracy"])
    best_model_path = best_model_info["path"]

    # Load and return the best model
    model = load_model(best_model_path)
    print(f"[LOADED] Loaded model '{best_model_info['name']}' with accuracy {best_model_info['accuracy']:.4f}")
    return model

def save_data_vectors(X, y, file_name='dataset_vectors.npz'):
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, file_name)
    np.savez_compressed(file_path, X=np.asarray(X), y=np.asarray(y))
    print(f"[SAVED] Data vectors saved at: {file_path}")
    return file_path

def load_data_vectors(file_name='dataset_vectors.npz'):
    file_path = os.path.join(DATA_DIR, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[WARN] No data vectors file found at {file_path}")
    data = np.load(file_path, allow_pickle=True)
    X = data['X']
    y = data['y']
    print(f"[LOADED] Data vectors loaded from: {file_path}")
    return X, y
    
    
