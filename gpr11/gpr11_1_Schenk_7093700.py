import re

from collections import Counter
import math
import heapq

# Der gegebene Satz
sentence = "One Ring to rule them all, One Ring to find them, One Ring to bring them all, and in the darkness bind them"

# Konvertierung in Kleinbuchstaben und Zählen der Häufigkeiten
sentence = sentence.lower()
char_counts = Counter(sentence)
# Gesamtanzahl der Zeichen
total_chars = len(sentence)
# Berechnung der relativen Häufigkeiten und Entropiebeiträge
frequencies = {char: count / total_chars for char, count in char_counts.items()}
entropy_contributions = {char: -frequencies[char] * math.log2(frequencies[char]) for char in frequencies}

# Berechnung der Gesamtentropie
total_entropy = sum(entropy_contributions.values())
print(frequencies, entropy_contributions, total_entropy)





# Funktion zur Erstellung des Huffman-Baumes
def create_huffman_tree(frequencies):
    heap = [[weight, [symbol, ""]] for symbol, weight in frequencies.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return heap[0]


# Erstellung des Huffman-Baumes
huffman_tree = create_huffman_tree(frequencies)
# Funktion zur Extraktion der Codetabelle aus dem Huffman-Baum
def create_huffman_code(tree):
    huffman_code = {}
    for pair in tree[1:]:
        huffman_code[pair[0]] = pair[1]
    return huffman_code


# Erstellung der Codetabelle
huffman_code = create_huffman_code(huffman_tree)
print(huffman_code)


# Codierung der Nachricht
encoded_message = ''.join(huffman_code[char] for char in sentence if char in huffman_code)
# Anzahl der benötigten Bytes
bytes_required = len(encoded_message) // 8
print(encoded_message, bytes_required)


