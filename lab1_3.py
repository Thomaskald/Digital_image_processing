import matplotlib.pyplot as plt
import numpy as np
import skimage as ski

image_3 = ski.io.imread(
    "https://github.com/jgenc/dip-lab-assets/blob/main/Flower.jpeg?raw=true"
)

# image_3 = ski.io.imread(
#     "https://github.com/jgenc/dip-lab-assets/blob/main/vase.jpg?raw=true"
# )

# image_3 = ski.data.hubble_deep_field()

plt.imshow(image_3)
plt.title("Original RGB Image")

plt.show()

image_3_R = image_3[:, :, 0]
image_3_G = image_3[:, :, 1]
image_3_B = image_3[:, :, 2]

fig, axs = plt.subplots(2, 3, figsize=(20, 10))

axs[0, 0].imshow(image_3_R, cmap="gray")
axs[0, 0].set_title("blue Channel")

axs[0, 1].imshow(image_3_G, cmap="gray")
axs[0, 1].set_title("Green Channel")

axs[0, 2].imshow(image_3_B, cmap="gray")
axs[0, 2].set_title("Blue Channel")

# Optionally: Show histograms;
axs[1, 0].hist(image_3_R.ravel(), bins=256, color="red", alpha=0.5)
axs[1, 1].hist(image_3_G.ravel(), bins=256, color="green", alpha=0.5)
axs[1, 2].hist(image_3_B.ravel(), bins=256, color="blue", alpha=0.5)

plt.show()

image_3_HSV = ski.color.rgb2hsv(image_3)
image_3_H = image_3_HSV[:, :, 0]
image_3_S = image_3_HSV[:, :, 1]
image_3_V = image_3_HSV[:, :, 2]

fig, axs = plt.subplots(2, 3, figsize=(20, 10))

axs[0, 0].imshow(image_3_H, cmap="gray")
axs[0, 0].set_title("Hue Channel")

axs[0, 1].imshow(image_3_S, cmap="gray")
axs[0, 1].set_title("Saturation Channel")

axs[0, 2].imshow(image_3_V, cmap="gray")
axs[0, 2].set_title("Value Channel")

# Optionally: Show histograms;
axs[1, 0].hist(image_3_H.ravel(), bins=256, color="red", alpha=0.5)
axs[1, 1].hist(image_3_S.ravel(), bins=256, color="green", alpha=0.5)
axs[1, 2].hist(image_3_V.ravel(), bins=256, color="blue", alpha=0.5)

plt.show()