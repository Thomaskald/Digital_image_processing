import matplotlib.pyplot as plt
from skimage import io, filters, feature

# Φόρτωση των εικόνων
girlface = io.imread('/home/thomas/Digital_Image_Processing/DIP-project-1/DIP-project-1/images-project-1/girlface.jpg', as_gray=True)
fruits = io.imread('/home/thomas/Digital_Image_Processing/DIP-project-1/DIP-project-1/images-project-1/fruits.jpg', as_gray=True)
leaf = io.imread('/home/thomas/Digital_Image_Processing/DIP-project-1/DIP-project-1/images-project-1/leaf.jpg', as_gray=True)

# Ερώτημα Α
# Εφαρμογή Otsu thresholding
threshold = filters.threshold_otsu(girlface)
binary = girlface > threshold

# Εμφάνιση αποτελέσματος
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title("Αρχική εικόνα")
plt.imshow(girlface, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Κύρια αντικείμενα εικόνας")
plt.imshow(binary, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

# Ερώτημα Β
# Εφαρμογή Gaussian φίλτρου για μείωση του θορύβου
blurred_image_fruits = filters.gaussian(fruits, sigma=1.4)

# Εφαρμογή φίλτρου Canny για ανίχνευση περιγραμμάτων
edges_fruits = feature.canny(blurred_image_fruits)

# Εμφάνιση αποτελέσματος
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Αρχική εικόνα")
plt.imshow(fruits, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Περιγράμματα αντικειμένων εικόνας")
plt.imshow(edges_fruits, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

# Ερώτημα Γ
# Εφαρμογή Gaussian blur για να έχουμε πιο καθαρό αποτέλεσμα
blurred_image_leaf = filters.gaussian(leaf, sigma=1.4)

# Εφαρμογή φίλτρου Laplacian για τον εντοπισμό των λεπτομερειών
laplacian_image = filters.laplace(blurred_image_leaf)

# Εμφάνιση αποτελέσματος
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title("Αρχική εικόνα")
plt.imshow(leaf, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Λεπτομέρειες αντικειμένων εικόνας")
plt.imshow(laplacian_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()