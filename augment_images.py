import cv2
import os

class ImageAugmenter:
    def __init__(self, image_root, sentence_amount):
        self.image_root = image_root
        self.sentence_amount = sentence_amount
    
    def inverse_colors(self, image_path, i):
        """Invert the colors of the given image."""
        if image_path is None or not os.path.exists(image_path):
            return
        image = cv2.imread(image_path)
        inverted_image = cv2.bitwise_not(image)
        cv2.imwrite(self.image_root + "{:03d}.jpg".format(i+self.sentence_amount), inverted_image)
    
    def gauss_blur(self, image_path, i):
        """Apply Gaussian blur to the given image."""
        if image_path is None or not os.path.exists(image_path):
            return
        image = cv2.imread(image_path)
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
        cv2.imwrite(self.image_root + "{:03d}.jpg".format(i+self.sentence_amount*2), blurred_image)
    
    def process_images(self):
        """Process augmentations for all basic images."""
        for i in range(1, self.sentence_amount + 1):
            image_path = self.image_root + "{:03d}.jpg".format(i)
            self.inverse_colors(image_path, i)
            self.gauss_blur(image_path, i)



if __name__ == "__main__":
    # Path for the images not including the index and extension
    IMAGE_ROOT = "../ML_Project/Dataset/Letters/"
    # Amount of basic images
    LETTER_AMOUNT=150
    
    augmenter = ImageAugmenter(IMAGE_ROOT, LETTER_AMOUNT)
    augmenter.process_images()
