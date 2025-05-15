from matplotlib import pyplot as plt
import numpy as np
import pytesseract
from PIL import Image
import cv2

image = cv2.imread('/home/thomas/Digital_Image_Processing/DIP-project-1/DIP-project-1/images-project-1/book-cover.jpeg', cv2.IMREAD_GRAYSCALE)
#text = pytesseract.image_to_string(image)
text = pytesseract.image_to_string('/home/thomas/Digital_Image_Processing/DIP-project-1/DIP-project-1/images-project-1/book-cover.jpeg')
print(text)

plt.imshow(image, cmap='gray')
plt.axis('off')
plt.show()