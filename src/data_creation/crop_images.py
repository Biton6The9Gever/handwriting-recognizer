import cv2
import os
import shutil
from tqdm import tqdm
import utils

# === GLOBAL CONFIG ===
IMAGE_ROOT = utils.RAW_DATA_PATH
BOX_SIZE = 65  # px for 2x2 box
PIXEL_ROWS_GAP = (1, 2, 2, 1, 2, 2, 1, 2, 2)
PIXEL_COLS_GAP = (1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2)


def crop_image_cv2(image_path, output_path, left, upper, right, lower):
    """Crop an image given coordinates."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"[WARN] Image not found at {image_path}")

        cropped_img = img[upper:lower, left:right]
        cv2.imwrite(output_path, cropped_img)
    except Exception as e:
        print(f"[WARN] Error cropping {image_path}: {e}")


def crop_individual_letter(letter):
    """Crop and save each small letter box from a given letter image."""

    categories = ["Capital", "Non_Capital"]

    def get_box_coordinates(col: int, row: int):
        """Compute bounding box for a given column and row."""
        left = sum(PIXEL_COLS_GAP[:col-1]) + (col - 1) * BOX_SIZE
        upper = sum(PIXEL_ROWS_GAP[:row-1]) + (row - 1) * BOX_SIZE
        right = sum(PIXEL_COLS_GAP[:col]) + col * BOX_SIZE
        lower = sum(PIXEL_ROWS_GAP[:row]) + row * BOX_SIZE
        return left, upper, right, lower

    output_index = 1
    for category in categories:
        input_image = os.path.join(IMAGE_ROOT, category, f"{category}_{letter}.jpg")
        for col in range(1, 16):
            for row in range(1, 11):
                left, upper, right, lower = get_box_coordinates(col, row)
                output_image = os.path.join(utils.PROCESSED_DATA_PATH, f"{letter}_{output_index:04d}.jpg")
                crop_image_cv2(input_image, output_image, left, upper, right, lower)
                output_index += 1


def crop_letter(letter):
    """Process one letter (crop border, split, and create boxes)."""
    try:
        input_path = os.path.join(IMAGE_ROOT, f"raw_{letter.lower()}.jpg")
        temp_path = "border_less_image.jpg"

        # Step 1: Crop border
        crop_image_cv2(input_path, temp_path, 33, 33, 1033, 1363)

        img = cv2.imread(temp_path)
        if img is None:
            raise ValueError(f"[WARN] Temporary file {temp_path} is empty for {letter}")

        height, width = img.shape[:2]
        half_height = height // 2

        # Step 2: Split top and bottom halves
        crop_image_cv2(temp_path, os.path.join(IMAGE_ROOT, "Capital", f"Capital_{letter}.jpg"),
                       0, 0, width, half_height)
        crop_image_cv2(temp_path, os.path.join(IMAGE_ROOT, "Non_Capital", f"Non_Capital_{letter}.jpg"),
                       0, half_height + 1, width, height)

        # Step 3: Remove temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

        # Step 4: Process individual boxes
        crop_individual_letter(letter)

    except Exception as e:
        print(f"[ERROR] processing letter {letter}: {e}")


def crop_all_letters(limit=26):
    """Process all letters A-Z."""
    print("\n[START] Cropping all letters...")

    folders = ["Capital", "Non_Capital"]

    # Create folders
    for folder in folders:
        os.makedirs(os.path.join(IMAGE_ROOT, folder), exist_ok=True)

    total = limit
    processed = 0
    failed = 0

    # Process letters with progress bar
    for i in tqdm(range(limit), desc="[PROGRESS] Cropping letters", ncols=80):
        letter = chr(ord("A") + i)
        try:
            crop_letter(letter)
            processed += 1
        except Exception as e:
            print(f"[WARN] Failed to process letter '{letter}': {e}")
            failed += 1

    # Delete temp folders
    for folder in folders:
        folder_path = os.path.join(IMAGE_ROOT, folder)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)

    # Summary
    print(f"[END] Cropping letters ({processed}/{total} processed successfully)")
    if failed:
        print(f"[INFO] {failed} letters failed.")



if __name__ == "__main__":
    crop_all_letters(limit=utils.CHAR_AMOUNT)
