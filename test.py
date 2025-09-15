from crop_images import LetterImageProcessor
import os

processor = LetterImageProcessor()
pcg=processor.pixel_cols_gap
prg=processor.pixel_rows_gap
current_directory = os.getcwd()
print(current_directory)
print(f'pcg={sum(pcg)}, prg={sum(prg)}')
processor.crop_image_cv2(
    image_path=f"{current_directory}\Dataset\Letters\Capital\Capital_A.jpg",
    output_path=f"{current_directory}\\test.jpg",
    left=0, upper=0, right=65, lower=65)

processor.crop_image_cv2(
    image_path=f"{current_directory}\Dataset\Letters\Capital\Capital_A.jpg",
    output_path=f"{current_directory}\\test2.jpg",
    left=65+pcg[0], upper=0, right=130, lower=65)