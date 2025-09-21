from utils import Utils
from collections import Counter
from crop_images import LetterImageProcessor
from augment_images import ImageAugmenter

# Initialize processor and augmenter
processor=LetterImageProcessor()
augmenter=ImageAugmenter(Utils.DATA_PATH,Utils.IMAGES_AMOUNT)

# 1: Process images
processor.process_all_letters(limit=Utils.CHAR_AMOUNT)

# 2: Resize images
Utils.resize_base_images()

# 3: Augment images
augmenter.process_images()

input()
Utils.clear_console()

# Delete the data (Comment to see the data)
Utils.get_images_size_distribution(Utils.DATA_PATH)
Utils.recreate_data_folder()



