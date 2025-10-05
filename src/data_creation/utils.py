import os
import cv2
import shutil
import pandas as pd
from collections import Counter

# ==== CONSTANTS ====
CHAR_AMOUNT = 26
IMAGES_AMOUNT = 300
IMAGE_SIZE = (64, 64)
AUGMENTATIONS_AMOUNT = 4

# ==== PATH SETUP ====
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed")
DATA_CSV = os.path.join(DATA_DIR, "dataset.csv")



# ==== FUNCTIONS ====

def recreate_data_folder():
    folder_path = PROCESSED_DATA_PATH
    try:
        if os.path.exists(folder_path):
            # ignores permission errors
            shutil.rmtree(folder_path, ignore_errors=True)  
        os.makedirs(folder_path, exist_ok=True)
        print(f"Recreated folder: {folder_path}")
    except Exception:
        pass


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
                    print(f"Failed to read image: {image_path}")

    size_counts = Counter(sizes)
    total = sum(size_counts.values())
    for size, count in size_counts.items():
        print(f"Size: {size}, Count: {count}")
    print(f"Total images processed: {total}")


def resize_base_images():
    print('\n --- Resizing base images ---')
    for i in range(CHAR_AMOUNT):
        letter = chr(ord("A") + i)
        for index in range(1, IMAGES_AMOUNT + 1):
            image_path = os.path.join(PROCESSED_DATA_PATH, f"{letter}_{index:04d}.jpg")
            image = cv2.imread(image_path)
            if image is not None:
                resized_image = cv2.resize(image, IMAGE_SIZE)
                cv2.imwrite(image_path, resized_image)
            else:
                print(f"Resizing failed for image {image_path}")
    print("Resizing completed.")


def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')


def create_csv_file():
    letters = [chr(ord('A') + i) for i in range(CHAR_AMOUNT)]
    indices = range(1, IMAGES_AMOUNT * (AUGMENTATIONS_AMOUNT + 1) + 1)

    paths = [os.path.join(PROCESSED_DATA_PATH, f"{letter}_{index:04d}.jpg")
             for letter in letters for index in indices]
    labels = [letter for letter in letters for _ in indices]

    assert len(labels) == len(paths), "Labels and paths must be the same length"

    df = pd.DataFrame({'label': labels, 'path': paths})
    df.to_csv(DATA_CSV, index=False)
    print(f"CSV file created: {DATA_CSV}")


def create_dataset():
    """
    Full dataset creation workflow:
    1. Crop images
    2. Resize images
    3. Augment images
    4. Create CSV
    """
    from data_creation.crop_images import process_all_letters
    from data_creation.augment_images import process_images

    print("Processing images...")
    # 1. Crop letters into individual boxes
    process_all_letters(limit=CHAR_AMOUNT)

    # 2. Resize base images
    resize_base_images()

    # 3. Augment images
    process_images()

    # 4. Create CSV
    print(f'\n--- Creating .csv file named {DATA_CSV} ---')
    create_csv_file()

    input("Data created.\nPress Enter to continue...")
    clear_console()
    
