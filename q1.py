import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from sklearn.cluster import KMeans
from skimage.util import img_as_float
from skimage.metrics import mean_squared_error

# Διαβάζουμε την εικόνα
image = img_as_float(io.imread('/home/thomas/Digital_Image_Processing/DIP-project-1/DIP-project-1/images-project-1/flowers.jpg'))
h, w, c = image.shape
pixels = image.reshape(-1, 3)

# Ο αριθμός των χρωμάτων (clusters) που θα δοκιμάσουμε
n_colors_list = [5, 20, 200, 1000]

# Ορίζουμε plot για εμφάνιση αποτελεσμάτων
plt.figure(figsize=(15, 10))
plt.subplot(1, len(n_colors_list)+1, 1)
plt.imshow(image)
plt.title("Αρχική")
plt.axis('off')

# Κβάντιση για κάθε επίπεδο
for i, n_colors in enumerate(n_colors_list, start=2):
    print(f"-> Επεξεργασία για {n_colors} χρώματα...")

    # Εκπαίδευση KMeans
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init='auto')
    kmeans.fit(pixels)
    new_colors = kmeans.cluster_centers_[kmeans.predict(pixels)]
    quantized_image = new_colors.reshape(h, w, 3)

    # Υπολογισμός MSE
    mse = mean_squared_error(image, quantized_image)
    print(f"MSE για {n_colors} χρώματα: {mse:.6f}")

    # Εμφάνιση εικόνας
    plt.subplot(1, len(n_colors_list)+1, i)
    plt.imshow(quantized_image)
    plt.title(f"{n_colors} χρώματα\nMSE={mse:.5f}")
    plt.axis('off')

plt.tight_layout()
plt.show()
