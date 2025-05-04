import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image
from sklearn.metrics import mean_squared_error

def quantize_image_kmeans(img_np, n_colors):
    pixels = img_np.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init='auto')
    kmeans.fit(pixels)
    labels = kmeans.predict(pixels)
    new_colors = kmeans.cluster_centers_.astype('uint8')
    quantized_pixels = new_colors[labels]
    return quantized_pixels.reshape(img_np.shape)

def compute_mse(original, quantized):
    return mean_squared_error(original.reshape(-1, 3), quantized.reshape(-1, 3))

# Φόρτωση εικόνας
image_path = "/home/thomas/Digital_Image_Processing/DIP-project-1/DIP-project-1/images-project-1/flowers.jpg"
original_img = Image.open(image_path).convert('RGB')
original_np = np.array(original_img)

# Επίπεδα κβάντισης
levels = [5, 20, 100, 200, 500]

# Δημιουργία plot 2x3
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Εμφάνιση αρχικής εικόνας
axes[0].imshow(original_np)
axes[0].set_title("Αρχική Εικόνα")
axes[0].axis('off')

# Κβάντιση και εμφάνιση κάθε εικόνας
for idx, k in enumerate(levels, start=1):
    quantized = quantize_image_kmeans(original_np, k)
    mse = compute_mse(original_np, quantized)

    axes[idx].imshow(quantized)
    axes[idx].set_title(f"{k} χρώματα\nMSE = {mse:.2f}")
    axes[idx].axis('off')

# Αν περισσεύει κενό subplot, απενεργοποίησέ το
if len(levels) < len(axes) - 1:
    for i in range(len(levels)+1, len(axes)):
        axes[i].axis('off')

plt.tight_layout()
plt.show()
