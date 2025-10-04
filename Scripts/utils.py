import os
import cv2
import shutil
import pandas as pd
from collections import Counter

class Utils:
    
    CHAR_AMOUNT=26
    IMAGES_AMOUNT=300
    IMAGE_SIZE= (64, 64)
    AUGMENTATIONS_AMOUNT=4
    IMAGE_PATH ="../ML_Project/Dataset/"
    DATA_PATH = "../ML_Project/Dataset/Processed/"
    DATA_CSV="test.csv"
    
    
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
        
    def resize_base_images():
         for i in range(Utils.CHAR_AMOUNT):  
            letter = chr(ord("A") + i)
            for index in range(1,Utils.IMAGES_AMOUNT+1):
                image_path = os.path.join(Utils.DATA_PATH, f"{letter}_{index:04d}.jpg")
                image=cv2.imread(image_path)
                if image is not None:
                    resized_image=cv2.resize(image,Utils.IMAGE_SIZE)
                    cv2.imwrite(image_path,resized_image)
                    # print(f"Saved resized image to {image_path}")
                else:
                    print(f"Resizing failed for image {image_path}")
    @staticmethod
    def clear_console():
        # Windows 
        if os.name == 'nt':
            os.system('cls')
        # macOS / Linux
        else:
            os.system('clear')
    
    @staticmethod
    def create_dataset():
        from Scripts.augment_images import ImageAugmenter
        from Scripts.crop_images import LetterImageProcessor
        
        # Initialize processor and augmenter
        processor=LetterImageProcessor()
        augmenter=ImageAugmenter(Utils.DATA_PATH,Utils.IMAGES_AMOUNT)
        
        # 1: Process images
        processor.process_all_letters(limit=Utils.CHAR_AMOUNT)
        
        # 2: Resize images
        Utils.resize_base_images()
        
        # 3: Augment images
        augmenter.process_images()
        
        input("Data created \n Press Enter to continue...")
        Utils.clear_console()
        

    @staticmethod
    def create_csv_file():
        import pandas as pd

        # List of letters
        letters = [chr(ord('A') + i) for i in range(Utils.CHAR_AMOUNT)]

        # List of indices
        indices = range(1, Utils.IMAGES_AMOUNT * (Utils.AUGMENTATIONS_AMOUNT + 1) + 1)

        # List of paths
        paths = [
            Utils.DATA_PATH + f"{letter}_{index:04d}.jpg"
            for letter in letters
            for index in indices
        ]

        # Create the label list to match paths
        labels = [letter for letter in letters for _ in indices]

        # Make sure both lists have the same length
        assert len(labels) == len(paths), "Labels and paths must be the same length"

        data = {'label': labels, 'path': paths}

        # Save Data
        df = pd.DataFrame(data)
        df.to_csv(Utils.DATA_CSV, index=False)

