from PIL import Image
import numpy as np
import math
import heapq


# --- Συνάρτηση για υπολογισμό πιθανοτήτων ---
def calculate_probabilities(image_array):
    values, counts = np.unique(image_array, return_counts=True)
    probs = counts / counts.sum()
    # Ταξινόμηση κατά φθίνουσα πιθανότητα
    return sorted(zip(values, probs), key=lambda x: x[1], reverse=True)


# --- Υλοποίηση Shannon-Fano ---
def shannon_fano_recursive(symbols_probs):
    if len(symbols_probs) == 1:
        return {symbols_probs[0][0]: ""}

    total_prob = sum(prob for _, prob in symbols_probs)
    accum_prob = 0
    split_idx = 0
    for i, (_, prob) in enumerate(symbols_probs):
        accum_prob += prob
        if accum_prob >= total_prob / 2:
            split_idx = i + 1
            break

    left = symbols_probs[:split_idx]
    right = symbols_probs[split_idx:]

    left_codes = shannon_fano_recursive(left)
    right_codes = shannon_fano_recursive(right)

    for key in left_codes:
        left_codes[key] = '0' + left_codes[key]
    for key in right_codes:
        right_codes[key] = '1' + right_codes[key]

    left_codes.update(right_codes)
    return left_codes


# --- Υπολογισμός μέσου μήκους κωδικών ---
def average_code_length(codes, symbols_probs):
    return sum(len(codes[symbol]) * prob for symbol, prob in symbols_probs)


# --- Υπολογισμός εντροπίας ---
def entropy(symbols_probs):
    return -sum(prob * math.log2(prob) for _, prob in symbols_probs)


# --- Υλοποίηση Huffman ---
class Node:
    def __init__(self, prob, symbol=None, left=None, right=None):
        self.prob = prob
        self.symbol = symbol
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.prob < other.prob


def build_huffman_tree(symbols_probs):
    heap = [Node(prob, symbol) for symbol, prob in symbols_probs]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(left.prob + right.prob, left=left, right=right)
        heapq.heappush(heap, merged)
    return heap[0]


def generate_huffman_codes(node, prefix=''):
    if node.symbol is not None:
        return {node.symbol: prefix}
    codes = {}
    codes.update(generate_huffman_codes(node.left, prefix + '0'))
    codes.update(generate_huffman_codes(node.right, prefix + '1'))
    return codes


# --- Κύρια συνάρτηση επεξεργασίας εικόνας ---
def process_image(image_path):
    # Φόρτωση εικόνας grayscale
    img = Image.open(image_path).convert('L')
    img_arr = np.array(img)
    size = img_arr.size  # αριθμός pixels

    # Υπολογισμός πιθανοτήτων
    symbols_probs = calculate_probabilities(img_arr)

    # Shannon-Fano κωδικοποίηση
    shannon_codes = shannon_fano_recursive(symbols_probs)
    shannon_avg_len = average_code_length(shannon_codes, symbols_probs)
    ent = entropy(symbols_probs)
    shannon_compression_ratio = (size * 8) / (size * shannon_avg_len)

    # Huffman κωδικοποίηση
    huffman_tree = build_huffman_tree(symbols_probs)
    huffman_codes = generate_huffman_codes(huffman_tree)
    huffman_avg_len = average_code_length(huffman_codes, symbols_probs)
    huffman_compression_ratio = (size * 8) / (size * huffman_avg_len)

    # Εμφάνιση αποτελεσμάτων
    print(f"Αποτελέσματα για εικόνα: {image_path}")
    print("\n--- Shannon-Fano ---")
    print("Κωδικές λέξεις (τιμή φωτεινότητας : κωδικός):")
    for symbol, _ in symbols_probs:
        print(f" {symbol} : {shannon_codes[symbol]}")
    print(f"Μέσο μήκος κωδικής λέξης: {shannon_avg_len:.4f} bits")
    print(f"Εντροπία: {ent:.4f} bits")
    print(f"Λόγος συμπίεσης: {shannon_compression_ratio:.4f}")

    print("\n--- Huffman ---")
    print("Κωδικές λέξεις (τιμή φωτεινότητας : κωδικός):")
    for symbol, _ in symbols_probs:
        print(f" {symbol} : {huffman_codes[symbol]}")
    print(f"Μέσο μήκος κωδικής λέξης: {huffman_avg_len:.4f} bits")
    print(f"Λόγος συμπίεσης: {huffman_compression_ratio:.4f}")

    print("\n--- Σύγκριση ---")
    if shannon_avg_len > huffman_avg_len:
        print("Η Huffman κωδικοποίηση πετυχαίνει μικρότερο μέσο μήκος κωδικής λέξης.")
    elif shannon_avg_len < huffman_avg_len:
        print("Η Shannon-Fano κωδικοποίηση πετυχαίνει μικρότερο μέσο μήκος κωδικής λέξης.")
    else:
        print("Οι δύο μέθοδοι πετυχαίνουν ίδιο μέσο μήκος κωδικής λέξης.")

    print("\n" + "=" * 50 + "\n")


# --- Εκτέλεση για τις δύο εικόνες ---
if __name__ == "__main__":
    process_image("airplane.jpg")
    process_image("bridge.jpg")
