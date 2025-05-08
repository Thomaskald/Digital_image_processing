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
levels = [5, 20, 100, 1000]

# Εμφάνιση αρχικής εικόνας
plt.figure(figsize=(5, 5))
plt.imshow(original_np)
plt.title("Αρχική Εικόνα")
plt.axis('off')
plt.show()

# Επεξεργασία κάθε επιπέδου κβάντισης
for k in levels:
    quantized = quantize_image_kmeans(original_np, k)
    mse = compute_mse(original_np, quantized)

    # Εμφάνιση κάθε κβαντισμένης εικόνας αμέσως
    plt.figure(figsize=(5, 5))
    plt.imshow(quantized)
    plt.title(f"{k} χρώματα\nMSE = {mse:.2f}")
    plt.axis('off')
    plt.show()