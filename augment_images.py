from PIL import Image, ImageOps

# path for the images not including the index and extension
IMAGE_ROOT = "../ML_Project/Dataset/SentencePictures/Sentence_"
SENTENCE_AMOUNT=48

def inverse_colors(image_path,i):
    """Invert the colors of the given image."""
    if(image_path==None):
        return
    image=Image.open(image_path)
    image=ImageOps.invert(image)
    image.save(IMAGE_ROOT+"{:03d}.jpg".format(i+SENTENCE_AMOUNT))



for i in range(1,SENTENCE_AMOUNT+1):
    image_path=IMAGE_ROOT+"{:03d}.jpg".format(i)
    inverse_colors(image_path,i)




