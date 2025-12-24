import os
import re
import cv2
import csv
import random
import numpy as np
import utils

# regex to match filenames like A_123.jpg
FILENAME_RE = re.compile(r"^([A-Za-z])_(\d+)\.(jpg)$")
LETTER_SPACING=-20

def load_sentences(path):
    with open(path, encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip()
        ]
        
def load_existing_sentences(csv_path=utils.SENTENCE_CSV):
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
    
def load_valid_files():
    valid_files = []

    for f in os.listdir(utils.PROCESSED_DATA_PATH):
        m = FILENAME_RE.match(f)
        if not m:
            continue

        idx = int(m.group(2))
        if 301 <= idx <= 600:
            continue

        valid_files.append(f)

    return valid_files


def load_letter(char, valid_files):
    char = char.upper()
    candidates = [f for f in valid_files if f.startswith(char + "_")]

    if not candidates:
        return None

    img_path = os.path.join(
        utils.PROCESSED_DATA_PATH, random.choice(candidates)
    )
    return cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)


def paste_letter(canvas, letter, x, y):
    canvas_h, canvas_w = canvas.shape
    h, w = letter.shape

    # Compute valid paste region
    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + w, canvas_w)
    y2 = min(y + h, canvas_h)

    if x1 >= x2 or y1 >= y2:
        return  # nothing to paste

    # Corresponding region in letter
    lx1 = x1 - x
    ly1 = y1 - y
    lx2 = lx1 + (x2 - x1)
    ly2 = ly1 + (y2 - y1)

    roi = canvas[y1:y2, x1:x2]
    letter_crop = letter[ly1:ly2, lx1:lx2]

    mask = letter_crop < 245
    roi[mask] = letter_crop[mask]


def estimate_sentence_width(sentence, canvas_h, valid_files):
    width = 20

    for char in sentence:
        if char == " ":
            width += random.randint(20, 35)
            continue

        letter = load_letter(char, valid_files)
        if letter is None:
            continue

        h, w = letter.shape
        if h > canvas_h:
            w = int(w * (canvas_h / h))

        width += w + LETTER_SPACING

    return width + 20


def generate_sentences(path):
    # wipe old data
    utils.recreate_data_folder(utils.SENTENCES_DIR)

    if os.path.exists(utils.SENTENCE_CSV):
        os.remove(utils.SENTENCE_CSV)

    sentences = load_sentences(path)
    valid_files = load_valid_files()

    # init CSV
    with open(utils.SENTENCE_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "text"])

        idx = 1  

        for sentence in sentences:
            canvas_h = 64
            canvas_w = estimate_sentence_width(sentence, canvas_h, valid_files)+20
            canvas = np.ones((canvas_h, canvas_w), dtype=np.uint8) * 255

            x_cursor = 10

            for char in sentence:
                if char == " ":
                    x_cursor += random.randint(20, 35)
                    continue

                letter = load_letter(char, valid_files)
                if letter is None:
                    continue

                h, w = letter.shape
                if h > canvas_h:
                    scale = canvas_h / h
                    letter = cv2.resize(letter, (int(w * scale), canvas_h))
                    h, w = letter.shape

                y = (canvas_h - h) // 2
                paste_letter(canvas, letter, x_cursor, y)
                x_cursor += w + LETTER_SPACING

            img_name = f"sentence_{idx:05d}.jpg"
            cv2.imwrite(os.path.join(utils.SENTENCES_DIR, img_name), canvas)
            writer.writerow([img_name, sentence])

            idx += 1