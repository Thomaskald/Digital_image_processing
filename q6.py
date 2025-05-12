import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, measure

# Φόρτωση εικόνας
image = io.imread('/home/thomas/Digital_Image_Processing/DIP-project-1/DIP-project-1/images-project-1/book-cover.jpeg')

#Μετατροπή εικόνας σε εικόνα αποχρώσεων γκρί
gray_image = color.rgb2gray(image)

#Χρήση Gaussian blur ώστε να είναι καλύτερο το αποτέλεσμα
blured_image = filters.gaussian(gray_image, sigma=1)

plt.imshow(gray_image, cmap='gray')
plt.title('Εικόνα αποχρώσεων γκρί')
plt.axis('off')
plt.show()

#Αλγόριθμος τμηματοποίησης με χρήση Otsu thresholding
binary_mask = blured_image > filters.threshold_otsu(blured_image)

plt.imshow(binary_mask, cmap='gray')
plt.title('Δυαδική μάσκα τμηματοποίησης')
plt.axis('off')
plt.show()