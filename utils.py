import os
import cv2
import shutil
from collections import Counter

class Utils:
    
    CHAR_AMOUNT=2
    IMAGES_AMOUNT=150
    AUGMENTATIONS_AMOUNT=2
    IMAGE_PATH ="../ML_Project/Dataset/"
    DATA_PATH = "../ML_Project/Dataset/Processed/"
    
    
    @staticmethod
    def recreate_data_folder():
        folder_path=rf'../ML_Project/Dataset/Processed'
        
        # Recreate the folder
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            os.makedirs(rf'{folder_path}', exist_ok=True)
            print(f"Recreated folder: {folder_path}")
            
    @staticmethod
    def get_image_size(image_path):
        img=cv2.imread(image_path)
        return img.shape
    
    @staticmethod
    def get_images_size_distribution(image_root):
        sizes = []  # collect all sizes here

        for i in range(Utils.CHAR_AMOUNT):  
            letter = chr(ord("A") + i)
            for index in range(1, Utils.IMAGES_AMOUNT * (Utils.AUGMENTATIONS_AMOUNT + 1) + 1):
                image_path = os.path.join(image_root, f"{letter}_{index:04d}.jpg")

                if os.path.exists(image_path):
                    size = Utils.get_image_size(image_path)
                    if size is not None:
                        sizes.append(size)
                    else:
                        print(f"Failed to read image: {image_path}")
                else:
                    continue  # Skip if the image does not exist

        # Count frequencies of each size
        size_counts = Counter(sizes)

        # Print results
        total = 0
        for size, count in size_counts.items():
            print(f"Size: {size}, Count: {count}")
            total += count
        print(f"Total images processed: {total}")
        
        


