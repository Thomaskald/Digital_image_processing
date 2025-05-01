import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
from scipy.fft import fft2, ifft2

# Φόρτωμα εικόνας
img = ski.io.imread("/home/thomas/Digital_Image_Processing/DIP-project-1/DIP-project-1/images-project-1/cornfield.jpg", as_gray=True)

# Α: Υπολογισμός μετασχηματισμού Fourier
fft_img = fft2(img)
fft_shifted = np.fft.fftshift(fft_img)

# Υπολογισμός πλάτους και φάσης
magnitude = np.abs(fft_shifted)
phase = np.angle(fft_shifted)

# Εμφάνιση εικόνας και φασμάτων
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("Εικόνα")

plt.subplot(1, 3, 2)
plt.imshow(np.log(1 + magnitude), cmap='gray')
plt.title("Φάσμα Πλάτους")

plt.subplot(1, 3, 3)
plt.imshow(phase, cmap='gray')
plt.title("Φάσμα Φάσης")

plt.tight_layout()
plt.show()

# Β: Αντιστροφή της φάσης (κατακόρυφα)
flipped_phase = np.flipud(phase)
modified_fft = magnitude * np.exp(1j * flipped_phase)

# Νέα φάσματα
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(np.log(1 + np.abs(modified_fft)), cmap='gray')
plt.title("Νέο Φάσμα Πλάτους")

plt.subplot(1, 2, 2)
plt.imshow(np.angle(modified_fft), cmap='gray')
plt.title("Αντεστραμμένο Φάσμα Φάσης")

plt.tight_layout()
plt.show()

# Γ: Αντίστροφος μετασχηματισμός
inv_fft_shifted = np.fft.ifftshift(modified_fft)
reconstructed = ifft2(inv_fft_shifted)
reconstructed_real = np.real(reconstructed)

# Προβολή τελικής εικόνας
plt.figure(figsize=(6, 6))
plt.imshow(reconstructed_real, cmap='gray')
plt.title("Ανακατασκευασμένη Εικόνα")
plt.axis('off')
plt.show()