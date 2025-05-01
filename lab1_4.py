import matplotlib.pyplot as plt
import numpy as np
import skimage as ski

image_4 = ski.io.imread(
    "https://raw.githubusercontent.com/jgenc/dip-lab-assets/refs/heads/main/a.bmp"
)

# image_4 = ski.io.imread(
#     "https://i1.sndcdn.com/artworks-ZgYuoif9KVy8mifu-ert9Zg-t500x500.jpg",
#     as_gray=True,
# )

size_x, size_y = image_4.shape

resize_factor = 0.8

new_size_x = int((size_x - (size_x * resize_factor)))
new_size_y = int((size_y - (size_y * resize_factor)))
image_4_simple = ski.transform.resize(
    image_4, (new_size_x, new_size_y), anti_aliasing=False
)

fig, axs = plt.subplots(2, 1, figsize=(13, 13))

axs[0].imshow(image_4, cmap="gray")
axs[0].set_title("Original Image")

axs[1].imshow(image_4_simple, cmap="gray")
axs[1].set_title("Aliased Image (Resize, no anti-aliasing)")

plt.plot()

plt.show()