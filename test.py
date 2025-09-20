from utils import Utils
from collections import Counter
from crop_images import LetterImageProcessor
from augment_images import ImageAugmenter

# Initialize processor and augmenter
processor=LetterImageProcessor()
augmenter=ImageAugmenter(Utils.DATA_PATH,Utils.IMAGES_AMOUNT)

# Process and augment images
processor.process_all_letters(limit=Utils.CHAR_AMOUNT)
augmenter.process_images()

Utils.get_images_size_distribution(Utils.DATA_PATH)

# Delete the data (Comment to see the data)
Utils.recreate_data_folder()



