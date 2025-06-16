import cv2
import numpy as np
import glob
from skimage.feature import SIFT
from scipy.cluster.vq import vq
import os
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from scipy.spatial.distance import cdist

# Φόρτωση path εικόνων
queen_paths = glob.glob("Queen-Resized/*.jpg")
rook_paths = glob.glob("Rook-resize/*.jpg")
bishop_paths = glob.glob("bishop_resized/*.jpg")
knight_paths = glob.glob("knight-resize/*.jpg")
pawn_paths = glob.glob("pawn_resized/*.jpg")

# Συγχώνευση όλων των διαδρομών
all_image_paths = queen_paths + rook_paths + bishop_paths + knight_paths + pawn_paths

# Ανάγνωση εικόνων
images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in all_image_paths]

# Α

sift = SIFT()

descriptors_list = []
valid_images = []

# Εξαγωγή SIFT χαρακτηριστικών από κάθε εικόνα
for image in images:
    try:
        sift.detect_and_extract(image)
        if sift.descriptors is not None and len(sift.descriptors) > 0:
            descriptors_list.append(sift.descriptors)
            valid_images.append(image)
    except RuntimeError:
        continue  # Παράλειψη εικόνων που αποτυγχάνουν

images_gray = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in all_image_paths]
valid_paths = []
for img, path in zip(images_gray, all_image_paths):
    for v_img in valid_images:
        if np.array_equal(img, v_img):
            valid_paths.append(path)
            break

# Συγκέντρωση όλων των περιγραφών SIFT
all_descriptors = np.vstack(descriptors_list).astype(np.float32)

# K-means για δημιουργία οπτικού λεξικού
k = 200
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
_, _, visual_words = cv2.kmeans(all_descriptors, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

image_word_indices = []
for desc in descriptors_list:
    words, _ = vq(desc, visual_words)
    image_word_indices.append(words)

# Δημιουργεία BoVW
bovw_vectors = []
for word_indices in image_word_indices:
    histogram = np.zeros(k)
    for word in word_indices:
        histogram[word] += 1
    bovw_vectors.append(histogram)

bovw_vectors = np.stack(bovw_vectors)

# TF-IDF για ενίσχυση πληροφορίας
N = len(bovw_vectors)
df = np.sum(bovw_vectors > 0, axis=0)
idf = np.log(N / (df + 1e-6))
tfidf_vectors = bovw_vectors * idf

# Β

# Ορισμός εικόνων αναζήτησης
query_images = [
    "Queen-Resized/00000000_resized.jpg",
    "Queen-Resized/00000001_resized.jpg",
    "Rook-resize/00000001_resized.jpg",
    "Rook-resize/00000002_resized.jpg",
    "bishop_resized/00000000_resized.jpg",
    "bishop_resized/00000002_resized.jpg",
    "knight-resize/00000001_resized.jpg",
    "knight-resize/00000002_resized.jpg",
    "pawn_resized/00000001_resized.jpg",
    "pawn_resized/00000002_resized.jpg"
]

image_index_map = {os.path.normpath(p): i for i, p in enumerate(valid_paths)}

query_indices = []
for query_path in query_images:
    norm_path = os.path.normpath(query_path)
    if norm_path in image_index_map:
        query_indices.append(image_index_map[norm_path])
    else:
        print(f"Η εικόνα δεν βρέθηκε: {query_path}")

# Υπολογισμός ευκλείδειων αποστάσεων
features = tfidf_vectors
distance_matrix = cdist(features, features, metric='euclidean')

# Γ

accuracies = []

for idx in query_indices:
    distances = distance_matrix[idx]
    distances[idx] = np.inf

    top10_indices = np.argsort(distances)[:10]

    query_path = valid_paths[idx]
    query_label = os.path.normpath(query_path).split(os.sep)[0].lower()

    result_paths = [valid_paths[i] for i in top10_indices]
    result_labels = [os.path.normpath(p).split(os.sep)[0].lower() for p in result_paths]

    correct_matches = sum(1 for lbl in result_labels if lbl == query_label)
    accuracy = correct_matches / 10.0
    accuracies.append(accuracy)

    # Δ

    fig, axs = plt.subplots(1, 11, figsize=(20, 4))
    fig.suptitle(f"Query: {query_label} | Ακρίβεια: {accuracy:.2f}", fontsize=14)

    axs[0].imshow(mpimg.imread(query_path), cmap='gray')
    axs[0].set_title("Query")
    axs[0].axis("off")

    for j, (img_path, label) in enumerate(zip(result_paths, result_labels), start=1):
        axs[j].imshow(mpimg.imread(img_path), cmap='gray')
        axs[j].set_title(label[:6])
        axs[j].axis("off")

    plt.tight_layout()
    plt.show()

# Υπολογισμός μέσης ακρίβειας ανάκτησης
mean_accuracy = np.mean(accuracies)
print(f"\nΜέση Ακρίβεια Ανάκτησης : {mean_accuracy:.2f}")