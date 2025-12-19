import cv2
import numpy as np
import os
import random
import json
import re

# temporary file to generate synthetic sentence images

LETTERS_PATH = "data/processed"
OUT_IMG = "data/sentences/images"
OUT_LBL = "data/sentences/labels"

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_LBL, exist_ok=True)

# regex: letter + index
FILENAME_RE = re.compile(r"^([A-Za-z])_(\d+)\.(png|jpg|jpeg)$")

# Cache valid files
VALID_FILES = []

for f in os.listdir(LETTERS_PATH):
    m = FILENAME_RE.match(f)
    if not m:
        continue

    idx = int(m.group(2))
    # exclude inverted augmentation range
    if 301 <= idx <= 600:
        continue

    VALID_FILES.append(f)


def load_letter(char):
    char = char.upper()
    candidates = [f for f in VALID_FILES if f.startswith(char + "_")]

    if not candidates:
        return None

    img_path = os.path.join(LETTERS_PATH, random.choice(candidates))
    return cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)


def paste_letter(canvas, letter, x, y):
    h, w = letter.shape
    roi = canvas[y:y+h, x:x+w]

    # Ink mask (tuned for handwriting)
    mask = letter < 245
    roi[mask] = letter[mask]

    canvas[y:y+h, x:x+w] = roi


def generate_sentence(sentence, idx):
    canvas_h = 64 
    canvas_w = 1600
    canvas = np.ones((canvas_h, canvas_w), dtype=np.uint8) * 255

    x_cursor = 10

    for char in sentence:
        if char == " ":
            x_cursor += random.randint(20, 35)
            continue

        letter = load_letter(char)
        if letter is None:
            continue

        h, w = letter.shape

        if h > canvas_h:
            scale = canvas_h / h
            letter = cv2.resize(letter, (int(w * scale), canvas_h))
            h, w = letter.shape

        if x_cursor + w >= canvas_w:
            break

        y = (canvas_h - h) // 2

        paste_letter(canvas, letter, x_cursor, y)
        x_cursor += w + random.randint(5, 12)

    img_name = f"sentence_{idx:05d}.png"
    cv2.imwrite(os.path.join(OUT_IMG, img_name), canvas)

    with open(os.path.join(OUT_LBL, f"sentence_{idx:05d}.json"), "w") as f:
        json.dump({"image": img_name, "text": sentence}, f)


# TEST
sentences = [
    "HELLO WORLD",
    "MACHINE LEARNING",
    "HANDWRITING OCR"
]

for i, s in enumerate(sentences):
    generate_sentence(s, i)
