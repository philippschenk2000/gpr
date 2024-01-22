""" This script contains the new exercise in gpr """


__author__ = "7093700, Schenk"

import math
import heapq

# Der gegebene Satz
sentence = "One Ring to rule them all, One Ring to find them, One Ring to bring them all, and in the darkness bind them"
sentence = sentence.lower()
characters_len = len(sentence)
unique_chars = ''.join(sorted(set(sentence)))
# Berechnung der relativen Häufigkeiten und Entropiebeiträge
[print('\ncharacter:', letter, ', absolute:', sentence.count(letter), ', relative:', round(sentence.count(letter)/characters_len, 3), ', entropy:', round(-sentence.count(letter)/characters_len * math.log2(sentence.count(letter)/characters_len), 3)) for letter in unique_chars]
# Häufigkeiten in dictionary
frequencies = {letter: round(-sentence.count(letter)/characters_len * math.log2(sentence.count(letter)/characters_len), 3) for letter in unique_chars}





# Funktion zur Erstellung des Huffman-Baumes
def create_huffman_tree(frequencies):
    heap = [[weight, [symbol, ""]] for symbol, weight in frequencies.items()]
    print(heap)
    print('-' * 100, ' Steps in between')
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        print(lo, hi)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    print('-' * 100)
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
print('Final Huffman code: \n', huffman_code)


# Codierung der Nachricht
encoded_message = ' '.join(huffman_code[char] for char in sentence if char in huffman_code and char != ' ')
# Anzahl der benötigten Bytes
bytes_required = len(encoded_message) // 8
print('Encoded message: \n', encoded_message)
print('Bytes required: \n', bytes_required, '\nFulfilled table:')

# Berechnung der mittleren Codewortlänge
average_code_length = sum(len(huffman_code[char]) * frequencies[char] for char in huffman_code)
average_code_length_rounded = round(average_code_length, 3)
[print('character:', letter, ', absolute:', sentence.count(letter), ', relative:', round(sentence.count(letter)/characters_len, 3), ', entropy:', round(-sentence.count(letter)/characters_len * math.log2(sentence.count(letter)/characters_len), 3), ', huffman-code:', huffman_code[letter], ', mid code-word-lenth:', average_code_length_rounded) for letter in unique_chars]
print('u, f, k, b, s, ,, a, m, g, l, h, o, r, i, t, e, n,  ')