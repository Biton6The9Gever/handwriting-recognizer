import os
import cv2
from Scripts import utils


# === Image augmentation functions ===

def inverse_colors(image_path):
    """Invert the colors of the given image."""
    if not image_path or not os.path.exists(image_path):
        return None
    image = cv2.imread(image_path)
    return cv2.bitwise_not(image)


def gauss_blur(image_path):
    """Apply Gaussian blur to the given image."""
    if not image_path or not os.path.exists(image_path):
        return None
    image = cv2.imread(image_path)
    return cv2.GaussianBlur(image, (5, 5), 0)


def positive_rotate(image_path, angle=15):
    """Rotate the image clockwise."""
    if not image_path or not os.path.exists(image_path):
        return None
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    return cv2.warpAffine(image, M, (w, h), borderValue=(255, 255, 255))


def negative_rotate(image_path, angle=-15):
    """Rotate the image counterclockwise."""
    if not image_path or not os.path.exists(image_path):
        return None
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    return cv2.warpAffine(image, M, (w, h), borderValue=(255, 255, 255))


def process_images(data_path=utils.DATA_PATH, images_amount=utils.IMAGES_AMOUNT):
    """Apply all augmentations to all base images."""
    augmentations = [
        inverse_colors,
        gauss_blur,
        positive_rotate,
        negative_rotate
    ]

    start_index = images_amount

    for i in range(utils.CHAR_AMOUNT):
        letter = chr(ord("A") + i)

        for method_idx, augment in enumerate(augmentations, start=1):
            for index in range(1, images_amount + 1):
                image_path = os.path.join(data_path, f"{letter}_{index:04d}.jpg")
                augmented_image = augment(image_path)

                if augmented_image is not None:
                    new_index = start_index + (method_idx - 1) * images_amount + index
                    save_path = os.path.join(data_path, f"{letter}_{new_index:04d}.jpg")
                    cv2.imwrite(save_path, augmented_image)
                    # print(f"Saved augmented image: {save_path}")
                else:
                    print(f"Augmentation {augment.__name__} failed for image: {image_path}")


if __name__ == "__main__":
    process_images()
