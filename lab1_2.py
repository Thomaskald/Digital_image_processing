import matplotlib.pyplot as plt
import skimage as ski
image_2 = ski.io.imread(
    "https://github.com/jgenc/dip-lab-assets/blob/main/petalouda.jpg?raw=true"
)

# image = ski.io.imread(
#     "https://github.com/jgenc/dip-lab-assets/blob/main/peps.jpg?raw=true"
# )

image_2 = ski.color.rgb2gray(image_2)
image_2 = ski.transform.resize(image_2, (300, 300))

image_2_80 = ski.transform.resize(image_2, (80, 80))
image_2_60 = ski.transform.resize(image_2, (60, 60))
image_2_40 = ski.transform.resize(image_2, (40, 40))

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs[0, 0].imshow(image_2, cmap="gray")
axs[0, 0].set_title("300x300")

axs[0, 1].imshow(image_2_80, cmap="gray")
axs[0, 1].set_title("80x80")

axs[1, 0].imshow(image_2_60, cmap="gray")
axs[1, 0].set_title("60x60")

axs[1, 1].imshow(image_2_40, cmap="gray")
axs[1, 1].set_title("40x40")

plt.show()  