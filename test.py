from crop_images import LetterImageProcessor
import os

processor = LetterImageProcessor()

box_size=processor.box_size
pcg=processor.pixel_cols_gap
prg=processor.pixel_rows_gap
current_directory = os.getcwd()
print(current_directory)

#for i in range(1, 16):  # 15 letters in a row 
#    for j in range(1, 11):  # 10 letters in a column 
#        processor.crop_image_cv2(
#            image_path = rf"{current_directory}\Dataset\Letters\Capital\Capital_A.jpg",
#            output_path = rf"{current_directory}\temp\test_{i}_{j}.jpg",
#            left=sum(pcg[:i-1]) + (i-1) * box_size,
#            upper=sum(prg[:j-1]) + (j-1) * box_size,
#            right=sum(pcg[:i]) + i * box_size,
#           lower=sum(prg[:j]) + j * box_size
#        )
