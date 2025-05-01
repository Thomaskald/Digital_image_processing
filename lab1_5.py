import matplotlib.pyplot as plt
import numpy as np
import skimage as ski

# image_5 = ski.io.imread(f"./dip_lab_assets/petalouda.bmp")
image_5 = ski.io.imread(
    "https://github.com/jgenc/dip-lab-assets/raw/refs/heads/main/petalouda.bmp"
)

# image_5 = ski.io.imread(
#     "https://github.com/jgenc/dip-lab-assets/blob/main/louvain.jpg?raw=true",
#     as_gray=True,
# )/home/thomas/ΑΙ

# image_5 = (image_5 * 255).astype(np.uint8)

N = 8

# Χωρίζει ισόποσα τις τιμές 0-256 σε N κομμάτια
bins = np.linspace(0, 256, N)

image_5_quantized = np.digitize(image_5, bins)

fig, axs = plt.subplots(2, 2, figsize=(15, 10))

axs[0, 0].imshow(image_5, cmap="gray")
axs[0, 0].set_title("Original Image")

axs[0, 1].hist(image_5.ravel(), bins=256, color="black", alpha=0.1, edgecolor="blue")
axs[0, 1].set_title("Histogram of the original image")

axs[1, 0].imshow(image_5_quantized, cmap="gray")
axs[1, 0].set_title("Quantized Image")

axs[1, 1].hist(
    image_5_quantized.ravel(),
    bins=N,
    color="black",
    alpha=0.1,
    edgecolor="blue",
)
axs[1, 1].set_title("Histogram of the quantized image")

plt.show()