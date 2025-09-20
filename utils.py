import os
import cv2
import shutil
from collections import Counter

class Utils:
    
    DATA_PATH = "../ML_Project/Dataset/Processed/"
    IMAGE_PATH ="../ML_Project/Dataset/"
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
    def get_images_size_distribution(image_root, limit):
        sizes = []  # collect all sizes here

        for i in range(limit):  
            for index in range(1, 151):
                image_path = rf'{image_root}{chr(ord("A") + i)}_{index:04d}.jpg'
                size = Utils.get_image_size(image_path)
                sizes.append(size)
                
        size_counts = Counter(sizes)
        
        # Print results
        sum=0
        for size, count in size_counts.items():
            print(f"Size: {size}, Count: {count}")
            sum+=count
        print(f"Total images processed: {sum}")
        
        


