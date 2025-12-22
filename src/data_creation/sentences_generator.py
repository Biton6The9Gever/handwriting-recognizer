import os
import re
import cv2
import csv
import random
import numpy as np
import utils

# regex to match filenames like A_123.jpg
FILENAME_RE = re.compile(r"^([A-Za-z])_(\d+)\.(jpg)$")


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
    h, w = letter.shape
    roi = canvas[y:y+h, x:x+w]

    mask = letter < 245
    roi[mask] = letter[mask]
    canvas[y:y+h, x:x+w] = roi


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

        width += w + random.randint(5, 12)

    return width + 20


def generate_sentences(sentences):
    """
    Generate sentence images and append labels to CSV.

    Args:
        sentences (list[str]): sentences to generate
    """
    start_idx=0
    os.makedirs(utils.SENTENCES_DIR, exist_ok=True)

    # init CSV once
    if not os.path.exists(utils.SENTENCE_CSV):
        with open(utils.SENTENCE_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["image", "text"])

    valid_files = load_valid_files()

    idx = start_idx

    for sentence in sentences:
        canvas_h = 64
        canvas_w = estimate_sentence_width(sentence, canvas_h, valid_files)
        canvas = np.ones((canvas_h, canvas_w), dtype=np.uint8) * 255

        x_cursor = 10

        for char in sentence:
            if char == " ":
                x_cursor += random.randint(10, 20)
                continue

            letter = load_letter(char, valid_files)
            if letter is None:
                continue

            h, w = letter.shape
            if h > canvas_h:
                scale = canvas_h / h
                letter = cv2.resize(
                    letter, (int(w * scale), canvas_h)
                )
                h, w = letter.shape

            y = (canvas_h - h) // 2
            paste_letter(canvas, letter, x_cursor, y)
            x_cursor += w + random.randint(5, 12)

        img_name = f"sentence_{idx:05d}.jpg"
        cv2.imwrite(
            os.path.join(utils.SENTENCES_DIR, img_name), canvas
        )
        print(f"Generated sentence image: {img_name}")
        with open(utils.SENTENCE_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([img_name, sentence])

        idx += 1