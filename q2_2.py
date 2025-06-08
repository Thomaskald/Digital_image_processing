from collections import defaultdict
import cv2
import numpy as np
import math
import heapq

def calculate_frequencies(image):
    freq = defaultdict(int)
    for pixel in image.flatten():
        freq[pixel] += 1
    total = image.size
    prob = {k: v / total for k, v in freq.items()}
    return dict(sorted(prob.items(), key=lambda item: item[1], reverse=True))

def shannon_fano(symbols_probs):
    def build(symbols):
        if len(symbols) == 1:
            return {symbols[0][0]: ''}
        total = sum(p for _, p in symbols)
        acc = 0
        for i in range(len(symbols)):
            acc += symbols[i][1]
            if acc >= total / 2:
                break
        left = symbols[:i + 1]
        right = symbols[i + 1:]
        left_codes = build(left)
        right_codes = build(right)
        for k in left_codes:
            left_codes[k] = '0' + left_codes[k]
        for k in right_codes:
            right_codes[k] = '1' + right_codes[k]
        return {**left_codes, **right_codes}

    symbols = list(symbols_probs.items())
    return build(symbols)

    # === 3. Κωδικοποίηση εικόνας ===


def encode_image(image, codebook):
    return ''.join(codebook[pixel] for pixel in image.flatten())

    # === 4. Εντροπία ===


def compute_entropy(probabilities):
    return -sum(p * math.log2(p) for p in probabilities.values())

    # === 5. Μέσο μήκος κωδικής λέξης ===


def average_code_length(probabilities, codebook):
    return sum(probabilities[sym] * len(codebook[sym]) for sym in codebook)

    # === 6. Μέγεθος & λόγος συμπίεσης ===


def compute_compression(original_image, encoded_str):
    original_bits = original_image.size * 8
    compressed_bits = len(encoded_str)
    ratio = original_bits / compressed_bits
    return original_bits, compressed_bits, ratio

    # === 7. Huffman ===


import heapq


def huffman_encoding(probabilities):
    heap = [[w, [sym, '']] for sym, w in probabilities.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]: pair[1] = '0' + pair[1]
        for pair in hi[1:]: pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return dict(sorted({sym: code for sym, code in heap[0][1:]}.items()))


# === 8. Διαδικασία πλήρης ===
def process_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    probs = calculate_frequencies(img)

    sf_codebook = shannon_fano(probs)
    huff_codebook = huffman_encoding(probs)

    sf_encoded = encode_image(img, sf_codebook)
    huff_encoded = encode_image(img, huff_codebook)

    entropy = compute_entropy(probs)
    avg_len_sf = average_code_length(probs, sf_codebook)
    #avg_len_huff = average_code_length(probs, huff_codebook)

    orig_bits, sf_bits, sf_ratio = compute_compression(img, sf_encoded)
    _, huff_bits, huff_ratio = compute_compression(img, huff_encoded)

    print(f"\n Εικόνα: {path}")
    print(f" Εντροπία: {entropy:.4f} bits/symbol")
    print(f" Μέσο μήκος (Shannon-Fano): {avg_len_sf:.4f}")
    #print(f" Μέσο μήκος (Huffman): {avg_len_huff:.4f}")
    print(f" Συμπίεση (Shannon-Fano): {sf_ratio:.2f}x")
    print(f" Συμπίεση (Huffman): {huff_ratio:.2f}x")
    print(f" Διαφορά: {(huff_ratio - sf_ratio):.4f}")

    print(f"\n Κωδικές λέξεις:")
    #for k, v in list(sf_codebook.items())[:10]:  # Εμφάνιση πρώτων 10 για συντομία
    for k, v in sf_codebook.items():
        print(f" {k}: {v}")
    #print("...")S

    return {
        'entropy': entropy,
        'avg_sf': avg_len_sf,
        #'avg_huff': avg_len_huff,
        'sf_ratio': sf_ratio,
        'huff_ratio': huff_ratio,
        'sf_codebook': sf_codebook,
        'huff_codebook': huff_codebook
    }


# === 9. Εκτέλεση ===
result_airplane = process_image("airplane.jpg")
result_bridge = process_image("bridge.jpg")
