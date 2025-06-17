import glob
import numpy as np
import skimage as ski
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from skimage.feature import graycomatrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean
import skimage
import os




check_paths = glob.glob("chcek/**/*.jp*g", recursive=True)
test_paths = glob.glob("test/**/*.jp*g", recursive=True)

image_paths = check_paths + test_paths

for path in image_paths:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    distance = 1
    angle = 7 * np.pi /4

    glcm = graycomatrix(
        image, distances=[distance], angles=[angle], levels=256, symmetric=True, normed=True
    )
    glcm_vector = glcm[:, :, 0, 0].reshape(-1)

    image_lbp = ski.feature.local_binary_pattern(image, P=16, R=1, method="uniform")
    image_lbp = image_lbp.astype(np.uint8)
    lbp_hist, _ = np.histogram(image_lbp.ravel(), bins=np.arange(0, image_lbp.max() + 2), density=True)


    lbp_vector = lbp_hist.reshape(-1)

    fd, hog_image = ski.feature.hog(
        image,
        orientations=9,
        pixels_per_cell=(8 ,8),
        cells_per_block=(3 ,3),
        visualize=True,
        feature_vector=False,
    )

    hog_vector = fd.reshape(-1)

    hog_image_rescaled = ski.exposure.rescale_intensity(
        hog_image, in_range=(0, 1), out_range=(0, 1)
    )

    print(f"\nΕικόνα: {path}")
    print(f"GLCM shape: {glcm[:, :, 0, 0].shape}, vector: {glcm_vector.shape}")
    print(f"LBP hist shape: {lbp_hist.shape}, vector: {lbp_vector.shape}")
    print(f"HOG shape: {fd.shape}, vector: {hog_vector.shape}")

# Δ
prototype_images = {
    "cow": "chcek/cow/cow_1001.jpg",
    "horse": "chcek/horse/horse_1001.jpeg",
    "nilgai": "chcek/Nilgai/nilgai_1001.jpeg",
    "buffelo": "chcek/water buffelo/buffelo_1001.jpeg"
}

prototypes = {}
for label, path in prototype_images.items():

    FIXED_SIZE = (128, 128)
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, FIXED_SIZE)

    # Υπολογισμός χαρακτηριστικών
    glcm = graycomatrix(image, distances=[1], angles=[ 7 *np.pi /4], levels=256, symmetric=True, normed=True)
    glcm_vec = glcm[:, :, 0, 0].reshape(-1)

    lbp = skimage.feature.local_binary_pattern(image, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, int(lbp.max() + 2)), density=True)
    lbp_vec = lbp_hist.reshape(-1)

    hog_fd, _ = skimage.feature.hog(
        image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3),
        visualize=True, feature_vector=False
    )
    hog_vec = hog_fd.reshape(-1)

    scaler = StandardScaler()
    glcm_vec = scaler.fit_transform(glcm_vec.reshape(-1, 1)).flatten()
    lbp_vec = scaler.fit_transform(lbp_vec.reshape(-1, 1)).flatten()
    hog_vec = scaler.fit_transform(hog_vec.reshape(-1, 1)).flatten()

    prototypes[label] = {
        "GLCM": glcm_vec,
        "LBP": lbp_vec,
        "HOG": hog_vec
    }

for path in test_paths:
    if any(proto in path for proto in prototype_images.values()):
        continue  # skip prototypes

    FIXED_SIZE = (128, 128)
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, FIXED_SIZE)
    # Παράδειγμα: GLCM
    glcm = graycomatrix(image, distances=[1], angles=[ 7 *np.pi /4], levels=256, symmetric=True, normed=True)
    glcm_vec = glcm[:, :, 0, 0].reshape(-1)

    lbp = skimage.feature.local_binary_pattern(image, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, int(lbp.max() + 2)), density=True)
    lbp_vec = lbp_hist.reshape(-1)

    hog_fd, _ = skimage.feature.hog(
        image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3),
        visualize=True, feature_vector=False
    )
    hog_vec = hog_fd.reshape(-1)

    scaler = StandardScaler()
    glcm_vec = scaler.fit_transform(glcm_vec.reshape(-1, 1)).flatten()
    lbp_vec = scaler.fit_transform(lbp_vec.reshape(-1, 1)).flatten()
    hog_vec = scaler.fit_transform(hog_vec.reshape(-1, 1)).flatten()

    similarities = {}
    for label, data in prototypes.items():
        dist1 = euclidean(glcm_vec, data["GLCM"])
        dist2 = euclidean(lbp_vec, data["LBP"])
        dist3 = euclidean(hog_vec, data["HOG"])
        dist_sum = dist1 + dist2 + dist3
        similarities[label] = dist_sum

    predicted_label = min(similarities, key=similarities.get)

    print(f"Predicted label: {predicted_label}")

    sorted_labels = sorted(similarities.items(), key=lambda x: x[1])
    predicted_label = sorted_labels[0][0]

    fig = plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])

    ax0 = plt.subplot(gs[0])
    labels = [lbl for lbl, _ in sorted_labels]
    dists = [score for _, score in sorted_labels]

    # Δημιουργία bar plot με ομοιότητες. Όσο μικρότερη η τιμή, τόσο περισσότερο μοιάζει με το συγκεκριμένο ζώο
    bars = ax0.barh(labels[::-1], dists[::-1], color='skyblue')
    ax0.set_title(f"Πρόβλεψη: {predicted_label}")
    ax0.set_xlabel("Ομοιότητα")

    # Προσθήκη τιμών μέσα στις μπάρες
    for bar, dist in zip(bars, dists[::-1]):
        width = bar.get_width()
        ax0.text(width - 0.02, bar.get_y() + bar.get_height() / 2,f"{dist:.3f}", ha='right', va='center', color='black', fontsize=9)

    ax1 = plt.subplot(gs[1])
    ax1.imshow(image, cmap='gray')
    ax1.axis('off')

    plt.tight_layout()
    plt.show()

# E
true_labels = []
predicted_labels_glcm = []
predicted_labels_lbp = []
predicted_labels_hog = []

for path in test_paths:

    FIXED_SIZE = (128, 128)
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, FIXED_SIZE)

    # GLCM
    glcm = graycomatrix(image, distances=[1], angles=[ 7 *np.pi /4], levels=256, symmetric=True, normed=True)
    glcm_vec = glcm[:, :, 0, 0].reshape(-1)
    glcm_vec = StandardScaler().fit_transform(glcm_vec.reshape(-1, 1)).flatten()

    # LBP
    lbp = skimage.feature.local_binary_pattern(image, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, int(lbp.max() + 2)), density=True)
    lbp_vec = lbp_hist.reshape(-1)
    lbp_vec = StandardScaler().fit_transform(lbp_vec.reshape(-1, 1)).flatten()

    # HOG
    hog_fd, _ = skimage.feature.hog(
        image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3),
        visualize=True, feature_vector=False
    )
    hog_vec = hog_fd.reshape(-1)
    hog_vec = StandardScaler().fit_transform(hog_vec.reshape(-1, 1)).flatten()

    # Υπολογισμός αποστάσεων για κάθε χαρακτηριστικό
    similarities_glcm = {}
    similarities_lbp = {}
    similarities_hog = {}

    for label, data in prototypes.items():
        similarities_glcm[label] = euclidean(glcm_vec, data["GLCM"])
        similarities_lbp[label] = euclidean(lbp_vec, data["LBP"])
        similarities_hog[label] = euclidean(hog_vec, data["HOG"])

    predicted_glcm = min(similarities_glcm, key=similarities_glcm.get)
    predicted_lbp = min(similarities_lbp, key=similarities_lbp.get)
    predicted_hog = min(similarities_hog, key=similarities_hog.get)

    true_label = os.path.basename(os.path.dirname(path)).lower()
    true_labels.append(true_label)

    predicted_labels_glcm.append(predicted_glcm)
    predicted_labels_lbp.append(predicted_lbp)
    predicted_labels_hog.append(predicted_hog)

    print("Predicted GLCM label:", predicted_glcm)
    print("Predicted LBP label:", predicted_lbp)
    print("Predicted HOG label:", predicted_hog)
    #plt.imshow(image, cmap='gray')

    #plt.show()

acc_glcm = accuracy_score(true_labels, predicted_labels_glcm)
acc_lbp = accuracy_score(true_labels, predicted_labels_lbp)
acc_hog = accuracy_score(true_labels, predicted_labels_hog)

print("\nAccuracy Results")
print(f"GLCM Accuracy: {acc_glcm:.2%}")
print(f"LBP  Accuracy: {acc_lbp:.2%}")
print(f"HOG  Accuracy: {acc_hog:.2%}")