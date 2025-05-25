from skimage import io, filters, feature, color
from skimage.filters import laplace, threshold_otsu
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

# Φόρτωση των εικόνων
leaf = io.imread('leaf.jpg', as_gray=True)
xray = io.imread('X-ray.jpeg', as_gray=True)
parking = io.imread('parking-lot.jpg', as_gray=True)

# Gaussian φίλτρο στις εικόνες για να έχουμε καλύτερα αποτελέσματα
blurred_image_leaf = filters.gaussian(leaf, sigma=1.4)
blurred_image_xray = filters.gaussian(xray, sigma=1.4)
blurred_image_parking = filters.gaussian(parking, sigma=1.4)

# Εικόνα leaf
leaf1 = feature.canny(blurred_image_leaf)

leaf2_sobel = filters.sobel(blurred_image_leaf)
threshold_sobel_leaf = threshold_otsu(leaf2_sobel)
leaf2 = leaf2_sobel > threshold_sobel_leaf

leaf3_laplace = filters.laplace(blurred_image_leaf)
threshold_laplace_leaf = threshold_otsu(leaf3_laplace)
leaf3 = leaf3_laplace > threshold_laplace_leaf

# Εμφάνιση αποτελεσμάτων εικόνας leaf
plt.figure(figsize=(12, 5))

plt.suptitle('Εικόνα leaf')

plt.subplot(2, 2, 1)
plt.imshow(leaf, cmap='gray')
plt.title('Αρχική εικόνα')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(leaf1, cmap='gray')
plt.title('Εικόνα με canny')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(leaf2, cmap='gray')
plt.title('Εικόνα με sobel')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(leaf3, cmap='gray')
plt.title('Εικόνα με laplacian')
plt.axis('off')

plt.show()

# Εικόνα xray
xray1 = feature.canny(blurred_image_xray)

xray2_sobel = filters.sobel(blurred_image_xray)
threshold_sobel_xray = threshold_otsu(xray2_sobel)
xray2 = xray2_sobel > threshold_sobel_xray

xray3_laplace = filters.laplace(blurred_image_xray)
threshold_laplace_xray = threshold_otsu(xray3_laplace)
xray3 = xray3_laplace > threshold_laplace_xray

# Εμφάνιση αποτελεσμάτων εικόνας xray
plt.figure(figsize=(12, 5))

plt.suptitle('Εικόνα xray')

plt.subplot(2, 2, 1)
plt.imshow(xray, cmap='gray')
plt.title('Αρχική εικόνα')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(xray1, cmap='gray')
plt.title('Εικόνα με canny')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(xray2, cmap='gray')
plt.title('Εικόνα με sobel')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(xray3, cmap='gray')
plt.title('Εικόνα με laplacian')
plt.axis('off')

plt.show()

# Εικόνα parking lot
parking1 = feature.canny(blurred_image_parking)

parking2_sobel = filters.sobel(blurred_image_parking)
threshold_sobel_parking = threshold_otsu(parking2_sobel)
parking2 = parking2_sobel > threshold_sobel_parking

parking3_laplace = filters.laplace(blurred_image_parking)
threshold_laplace_parking = threshold_otsu(parking3_laplace)
parking3 = parking3_laplace > threshold_laplace_parking

# Εμφάνιση αποτελεσμάτων εικόνας parking
plt.figure(figsize=(12, 5))

plt.suptitle('Εικόνα parking')

plt.subplot(2, 2, 1)
plt.imshow(parking, cmap='gray')
plt.title('Αρχική εικόνα')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(parking1, cmap='gray')
plt.title('Εικόνα με canny')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(parking2, cmap='gray')
plt.title('Εικόνα με sobel')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(parking3, cmap='gray')
plt.title('Εικόνα με laplacian')
plt.axis('off')

plt.show()