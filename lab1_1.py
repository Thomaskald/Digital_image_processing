import matplotlib.pyplot as plt
import skimage as ski

image_1 = ski.io.imread(
    "https://raw.githubusercontent.com/jgenc/dip-lab-assets/refs/heads/main/lenna.bmp",
    as_gray=True,
)

plt.imshow(image_1, cmap="gray")
plt.title("Original Image")
plt.show()

def bit_plane_slice(image, bit):
    return (image >> bit - 1) & 1

bit_plane_1 = bit_plane_slice(image_1, 1)
bit_plane_2 = bit_plane_slice(image_1, 2)
bit_plane_3 = bit_plane_slice(image_1, 3)
bit_plane_4 = bit_plane_slice(image_1, 4)
bit_plane_5 = bit_plane_slice(image_1, 5)
bit_plane_6 = bit_plane_slice(image_1, 6)
bit_plane_7 = bit_plane_slice(image_1, 7)
bit_plane_8 = bit_plane_slice(image_1, 8)

fix, axs = plt.subplots(2, 4, figsize=(20, 10))

for i, ax in reversed(list(enumerate(axs.flat))):
    ax.imshow(eval(f"bit_plane_{8 - i}"), cmap="gray")
    ax.set_title(f"Bit Plane {i + 1}")

plt.show()