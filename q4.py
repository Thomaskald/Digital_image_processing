from matplotlib import pyplot as plt
from skimage import io, filters
from skimage.filters import median
from skimage.metrics import structural_similarity as ssim
from skimage.morphology import disk

image1 = io.imread('/home/thomas/Digital_Image_Processing/DIP-project-1/DIP-project-1/images-project-1/lenna-n1.jpg', as_gray=True)
image2 = io.imread('/home/thomas/Digital_Image_Processing/DIP-project-1/DIP-project-1/images-project-1/lenna-n2.jpg', as_gray=True)
image3 = io.imread('/home/thomas/Digital_Image_Processing/DIP-project-1/DIP-project-1/images-project-1/lenna-n3.jpg', as_gray=True)

#Εικόνα 1
img1_filtered = filters.gaussian(image1, 1.4)
img1_ssim = ssim(image1, img1_filtered, data_range=31)
print(f"SSIM - Εικόνα 1 (Gaussian Blur):       {img1_ssim:.4f}")
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.imshow(image1, cmap='gray')
plt.title('Noisy 1')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(img1_filtered, cmap='gray')
plt.title('Filtered 1 (Gaussian Blur)')
plt.axis('off')
plt.suptitle("Αρχική και Φιλτραρισμένη - Εικόνα 1 (Gaussian noise)")
plt.show()

#Εικόνα 2
img2_filtered = median(image2, footprint=disk(2))
img2_ssim = ssim(image2, img2_filtered)
print(f"SSIM - Εικόνα 2 (Median filter, Salt & Pepper noise):   {img2_ssim:.4f}")
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image2, cmap='gray')
plt.title('Noisy 2')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(img2_filtered, cmap='gray')
plt.title('Filtered 2 (Median)')
plt.axis('off')
plt.suptitle("Αρχική και Φιλτραρισμένη - Εικόνα 2 (Salt & Pepper noise)")
plt.show()