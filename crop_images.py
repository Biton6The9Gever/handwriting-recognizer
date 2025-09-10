import cv2
import os

IMAGE_ROOT="../ML_Project/Dataset/Letters/"
def crop_image_cv2(image_path, output_path, left, upper, right, lower):
    """
    Crops an image using OpenCV.

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
        print(f"Image successfully cropped and saved to {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


for i in range(1):  # TODO replace 1 with 26 for all letters
    try:
        letter = chr(ord('A') + i)

        # crop the border
        crop_image_cv2(f"{IMAGE_ROOT}None_Croped/None_Croped_{letter}.jpg", "border_less_image.jpg", 14, 14, 1014, 1344)

        # load the cropped image to get its dimensions
        img = cv2.imread("border_less_image.jpg")
        if img is None:
            raise ValueError(f"border_less_image.jpg is empty for letter {letter}")

        height, width = img.shape[:2]
        half_height = height // 2

        # top half -> Capital
        crop_image_cv2("border_less_image.jpg", f"{IMAGE_ROOT}Capital/Capital_{letter}.jpg", 0, 0, width, half_height)
        # bottom half -> Non-Capital
        # +1 for rounding error
        crop_image_cv2("border_less_image.jpg", f"{IMAGE_ROOT}Non_Capital/Non_Capital_{letter}.jpg", 0, half_height+1, width, height)

        # remove temporary file
        if os.path.exists("border_less_image.jpg"):
            os.remove("border_less_image.jpg")
        else:
            raise FileNotFoundError("The file 'border_less_image.jpg' does not exist")

    except Exception as e:
        print(f"Error processing letter {letter}: {e}")