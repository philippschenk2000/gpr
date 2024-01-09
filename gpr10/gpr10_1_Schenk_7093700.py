import re


def main():
    filename = 'Kleopatra.txt'
    with open(filename, 'r') as file:
        text = file.read()
    # CAREFUL: JUST THE WORD 'SIE' NOT JUST THE THREE COMBINED LETTERS --> SPACE BEFORE AND AFTER
    text = re.sub(r' sie ', ' er ', text)
    text = re.sub(r' Sie ', ' Er ', text)
    print(text)


main()
