import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, feature, color, measure
from skimage.filters import sobel
from skimage import morphology
from skimage.transform import integral_image

# Ανάγνωση εικόνας
girlface = io.imread('/home/thomas/Digital_Image_Processing/DIP-project-1/DIP-project-1/images-project-1/girlface.jpg', as_gray=True)
fruits = io.imread('/home/thomas/Digital_Image_Processing/DIP-project-1/DIP-project-1/images-project-1/fruits.jpg', as_gray=True)
leaf = io.imread('/home/thomas/Digital_Image_Processing/DIP-project-1/DIP-project-1/images-project-1/leaf.jpg', as_gray=True)

# Ερώτημα Α
# Εφαρμογή thresholding με Otsu
threshold = filters.threshold_otsu(girlface)
binary = girlface > threshold

# Εμφάνιση
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Αρχική Εικόνα")
plt.imshow(girlface, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Κύρια Αντικείμενα με Otsu")
plt.imshow(binary, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

# Ερώτημα Β
# Εφαρμογή Gaussian φίλτρου για μείωση του θορύβου (προαιρετικό, για καθαρότερη ανίχνευση)
blurred_image_fruits = filters.gaussian(fruits, sigma=1.4)

# Εφαρμογή φίλτρου Canny για ανίχνευση περιγραμμάτων
edges_fruits = feature.canny(blurred_image_fruits)

# Εμφάνιση αποτελέσματος
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Αρχική Εικόνα")
plt.imshow(fruits, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Περιγράμματα με Canny")
plt.imshow(edges_fruits, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

# Ερώτημα Γ
# Εφαρμογή φίλτρου Laplacian για τον εντοπισμό των λεπτομερειών
laplacian_image = filters.laplace(leaf)

# Εμφάνιση αποτελέσματος
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Αρχική Εικόνα")
plt.imshow(leaf, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Λεπτομέρειες με Laplacian")
plt.imshow(laplacian_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()