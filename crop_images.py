import cv2
import os

class LetterImageProcessor:
    def __init__(self, image_root="../ML_Project/Dataset/Letters/"):
        """
        Initialize the image processor with a root directory.
        """
        self.image_root = image_root
        self.box_size=65 #px for 2x2 box 
        # Define pixel gaps measured by eye
        self.pixel_rows_gap = (1, 2, 2, 1, 2, 2, 1, 2, 2)
        self.pixel_cols_gap = (1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2)

    def crop_image_cv2(self, image_path, output_path, left, upper, right, lower):
        """
        Crop an image given the cropping coordinates.

        Args:
            image_path (str): Path to the input image.
            output_path (str): Path to save the cropped image.
            left (int): The x-coordinate of the top-left corner of the crop box.
            upper (int): The y-coordinate of the top-left corner of the crop box.
            right (int): The x-coordinate of the bottom-right corner of the crop box.
            lower (int): The y-coordinate of the bottom-right corner of the crop box.
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise FileNotFoundError(f"Image not found at {image_path}")

            # Crop the image (y:upper:lower, x:left:right)
            cropped_img = img[upper:lower, left:right]
            cv2.imwrite(output_path, cropped_img)
            print(f"Cropped and saved: {output_path}")
        except Exception as e:
            print(f"Error cropping {image_path}: {e}")

    def process_letter(self, letter):
        """
        Process a single letter:
        1. Crop border
        2. Split into Capital and Non-Capital
        3. Save results in respective folders
        """
        try:
            input_path = f"{self.image_root}None_Cropped/None_Cropped_{letter}.jpg"
            temp_path = "border_less_image.jpg"

            # Step 1: Crop the border
            self.crop_image_cv2(input_path, temp_path, 33, 33, 1033, 1363)

            # Step 2: Load cropped image
            img = cv2.imread(temp_path)
            if img is None:
                raise ValueError(f"Temporary file {temp_path} is empty for letter {letter}")

            height, width = img.shape[:2]
            half_height = height // 2

            # Step 3: Save top half -> Capital
            self.crop_image_cv2(
                temp_path,
                f"{self.image_root}Capital/Capital_{letter}.jpg",
                0, 0, width, half_height
            )

            # Step 4: Save bottom half -> Non-Capital
            self.crop_image_cv2(
                temp_path,
                f"{self.image_root}Non_Capital/Non_Capital_{letter}.jpg",
                0, half_height + 1, width, height
            )

            # Step 5: Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)

        except Exception as e:
            print(f"Error processing letter {letter}: {e}")

    def process_all_letters(self, limit=26):
        """
        Process all letters A-Z (default 26).
        """
        for i in range(limit):
            letter = chr(ord('A') + i)
            print(f"\n--- Processing letter {letter} ---")
            self.process_letter(letter)


if __name__ == "__main__":
    processor = LetterImageProcessor()
    processor.process_all_letters(limit=2)  # change to 26 for all letters