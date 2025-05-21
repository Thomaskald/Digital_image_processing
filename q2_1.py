from skimage import io, filters, feature, color
from skimage.filters import laplace
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
leaf2 = filters.sobel(blurred_image_leaf)
leaf3 = filters.laplace(blurred_image_leaf)

# Εικόνα xray
xray1 = feature.canny(blurred_image_xray)
xray2 = filters.sobel(blurred_image_xray)
xray3 = filters.laplace(blurred_image_xray)

# Εικόνα parking lot
parking1 = feature.canny(blurred_image_parking)
parking2 = filters.sobel(blurred_image_parking)
parking3 = filters.laplace(blurred_image_parking)

# plt.imshow(parking, cmap='gray')
# plt.show()
# plt.imshow(parking1, cmap='gray')
# plt.show()
# plt.imshow(parking2, cmap='gray')
# plt.show()
# plt.imshow(parking3, cmap='gray')
# plt.show()
