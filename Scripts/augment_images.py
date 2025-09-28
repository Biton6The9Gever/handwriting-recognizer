import os
import cv2
import random
from Scripts.utils import Utils

class ImageAugmenter:
    def __init__(self, data_path, images_amount):
        self.data_path = data_path
        self.images_amount = images_amount
    
    def inverse_colors(self, image_path):
        """Invert the colors of the given image."""
        if image_path is None or not os.path.exists(image_path):
            return
        image = cv2.imread(image_path)
        return cv2.bitwise_not(image)
        
    
    def gauss_blur(self, image_path):
        """Apply Gaussian blur to the given image."""
        if image_path is None or not os.path.exists(image_path):
            return
        image = cv2.imread(image_path)
        return  cv2.GaussianBlur(image, (5, 5), 0)
    
    def postive_rotate(self, image_path,angle=15):
        if image_path is None or not os.path.exists(image_path):
            return
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        return cv2.warpAffine(image, M, (w, h), borderValue=(255, 255, 255))
    
    def negative_rotate(self, image_path,angle=-15):
        if image_path is None or not os.path.exists(image_path):
            return
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        return cv2.warpAffine(image, M, (w, h), borderValue=(255, 255, 255))
    
    def process_images(self):
        """Process augmentations for all basic images."""
        augmentations = [self.inverse_colors, self.gauss_blur,self.postive_rotate,self.negative_rotate]


        # start index for new images after the base ones
        start_index = self.images_amount

        for i in range(Utils.CHAR_AMOUNT):
            letter = chr(ord("A") + i)

            for method_idx, augment in enumerate(augmentations, start=1):  # 1-based index for methods
                for index in range(1, self.images_amount + 1):
                    image_path = os.path.join(self.data_path, f"{letter}_{index:04d}.jpg")
                    augmented_image = augment(image_path)

                    if augmented_image is not None:
                        # Compute new index: existing images + images from previous methods
                        new_index = start_index + (method_idx - 1) * self.images_amount + index
                        save_path = os.path.join(
                            self.data_path,
                            f"{letter}_{new_index:04d}.jpg"
                        )
                        cv2.imwrite(save_path, augmented_image)
                        # print(f"Saved augmented image: {save_path}")
                    else:
                        print(f"Augmentation {augment.__name__} failed for image: {image_path}")
            


if __name__ == "__main__":    
    augmenter = ImageAugmenter(Utils.DATA_PATH,Utils.IMAGES_AMOUNT)
    augmenter.process_images()
