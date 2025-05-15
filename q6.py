from matplotlib import pyplot as plt
import pytesseract
import cv2

# Μετατροπή εικόνας σε εικόνα αποχρώσεων γκρί
image = cv2.imread('/home/thomas/Digital_Image_Processing/DIP-project-1/DIP-project-1/images-project-1/book-cover.jpeg', cv2.IMREAD_GRAYSCALE)
#text = pytesseract.image_to_string(image)
#text = pytesseract.image_to_string('')
#print(text)

# Gaussian blur για καλύτερα αποτελέσματα
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Υπολογισμός δυαδικής μάσκας
binary_mask = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 0)

# Εμφάνιση εικόνας gray και δυαδικής μάσκας
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Εικόνα αποχρώσεων γκρί')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(binary_mask, cmap='gray')
plt.title('Δυαδική μάσκα')
plt.axis('off')
plt.show()

# Αναγνώριση λέξεων
text = pytesseract.image_to_string(binary_mask, lang='eng')
print(text)