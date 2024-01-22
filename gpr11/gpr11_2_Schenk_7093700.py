""" This script contains the new exercise in gpr """


__author__ = "7093700, Schenk"

# Huffman Codetabelle
huffman_codes = {' ': '00', 'l': '0100', 'u': '010100', 'b': '010101', ',': '01011', 'n': '011', 'o': '1000', 'r': '1001',
                 'i': '1010', 'a': '10110', 'd': '10111', 'f': '1100000', 'k': '1100001', 's': '110001', 'g': '11001',
                 't': '1101', 'm': '11100', 'h': '11101', 'e': '1111'}
reverse_huffman_codes = {v: k for k, v in huffman_codes.items()}

def encode(text):
    try:
        return ' '.join(huffman_codes[char] for char in text)
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


# Testfälle für Decodierung
encoded_strings = ["1111 1010 011 00 1101 1111 110001 1101", "10111 10110 1101 1111 011", "00", "fk", "1111101001100110111111100011101"]
decoded_strings = [decode(s.replace(' ', '')) for s in encoded_strings]
print("Decodierung:")
for s, decoded in zip(encoded_strings, decoded_strings):
    print(f"'{s}' -> '{decoded}'")


# Testfälle für Codierung
test_strings = ["ein test", "test", "daten", "ein", "amvp3plv"]
encoded_strings = [encode(s) for s in test_strings]
print("\nCodierung:")
for s, encoded in zip(test_strings, encoded_strings):
    print(f"'{s}' -> '{encoded}'")

# Codierung von "ein test"
encoded_ein_test = encode("ein test")
print("\nCodierung von 'ein test':", encoded_ein_test)

# Erklärung, warum "eine probe" nicht codiert werden kann
print("\nErklärung für 'eine probe':")
print("Die Nachricht 'eine probe' kann nicht codiert werden, weil die Buchstaben 'b', 'o' und 'p' in der Huffman-Codetabelle nicht definiert sind.")



