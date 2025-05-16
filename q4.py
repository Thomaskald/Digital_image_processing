from matplotlib import pyplot as plt
from skimage import io
from skimage.filters import median
from skimage.metrics import structural_similarity as ssim
from skimage.morphology import disk
from skimage.restoration import denoise_nl_means

lennaImage = io.imread('lenna.jpg', as_gray=True)
image1 = io.imread('lenna-n1.jpg', as_gray=True) / 255.0
image2 = io.imread('lenna-n2.jpg', as_gray=True)
image3 = io.imread('lenna-n3.jpg', as_gray=True)

#Εικόνα 1
sigma_est = 0.08
patch_kw = dict(patch_size = 5, patch_distance = 6, channel_axis = None)
img1_filtered = denoise_nl_means(image1, h=1.15 * sigma_est, fast_mode = True, **patch_kw)
img1_ssim = ssim(lennaImage, img1_filtered, data_range=31)
print(f"SSIM - Εικόνα 1: {img1_ssim:.4f}")
plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
plt.imshow(image1, cmap='gray')
plt.title('Αρχική εικόνα')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(img1_filtered, cmap='gray')
plt.title('Φιλτραρισμένη εικόνα')
plt.axis('off')
plt.suptitle("Εικόνα 1")
plt.show()

#Εικόνα 2
img2_filtered = median(image2, footprint=disk(2))
img2_ssim = ssim(lennaImage, img2_filtered)
print(f"SSIM - Εικόνα 2: {img2_ssim:.4f}")
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(image2, cmap='gray')
plt.title('Αρχική εικόνα')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(img2_filtered, cmap='gray')
plt.title('Φιλτραρισμένη εικόνα')
plt.axis('off')
plt.suptitle("Εικόνα 2")
plt.show()