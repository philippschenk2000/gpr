""" This script contains the new exercise in gpr """


__author__ = "7093700, Schenk"

# Huffman Codetabelle
huffman_codes = {
    ' ': '1100', ',': '010', 'a': '00010', 'd': '0111', 'e': '110',
    'g': '00011', 'i': '1010', 'l': '0010', 'n': '1011', 'r': '0011',
    's': '1110', 't': '1111', 'u': '0110', 'w': '0000'
}
reverse_huffman_codes = {v: k for k, v in huffman_codes.items()}

def encode(text):
    try:
        return ''.join(huffman_codes[char] for char in text)
    except KeyError:
        return "Fehler: Eingabe enthält nicht codierbare Zeichen."

def decode(code):
    decoded_text = ""
    while code:
        for char_code in reverse_huffman_codes:
            if code.startswith(char_code):
                decoded_text += reverse_huffman_codes[char_code]
                code = code[len(char_code):]
                break
        else:
            return "Fehler: Ungültiger Code."
    return decoded_text


# Testfälle für Codierung
test_strings = ["ein test", "test", "daten", "algorithmus", "python"]
encoded_strings = [encode(s) for s in test_strings]
print("Codierung:")
for s, encoded in zip(test_strings, encoded_strings):
    print(f"'{s}' -> '{encoded}'")

# Testfälle für Decodierung
encoded_strings = ["1011111101111", "11110111", "0001010111010111", "Fehler", "11010101100"]
decoded_strings = [decode(s) for s in encoded_strings]
print("\nDecodierung:")
for s, decoded in zip(encoded_strings, decoded_strings):
    print(f"'{s}' -> '{decoded}'")

# Codierung von "ein test"
encoded_ein_test = encode("ein test")
print("\nCodierung von 'ein test':", encoded_ein_test)

# Erklärung, warum "eine probe" nicht codiert werden kann
print("\nErklärung für 'eine probe':")
print("Die Nachricht 'eine probe' kann nicht codiert werden, weil die Buchstaben 'b', 'o' und 'p' in der Huffman-Codetabelle nicht definiert sind.")



