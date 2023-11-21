def main():
    __author__ = "7093700, Schenk"
    try:
        # INPUT OF NUMBER
        zahl = float(input('Enter a beautiful number: '))
        # CASE: INPUT = 0
        if zahl <= 0 or zahl >= 1:
            print("Fehler: x muss zwischen 0 und 1 liegen.")
            return ''
        # USUAL PROCESSING: DIVIDING BY 2 TIL NO REST
        binary_list = []
        while zahl > 0:
            if len(binary_list) >= 32:
                break
            zahl = zahl * 2
            if zahl >= 1:
                binary_list.append(1)
                zahl -= 1
            else:
                binary_list.append(0)
            print(zahl)
        # REVERSE THE APPENDED LIST AND THEN PRINT EVERY ELEMENT
        reversed_list = list(reversed(binary_list))
        # print(reversed_list)
        print('The result number would be:')
        print('0.', end='')
        for r in reversed_list:
            print(str(int(r)), end='')
    # CASE: INPUT NOT INTEGER
    except:
        print('Input has to be a number.')


main()


# TEST 1:
# IN: 13.578125
# SHOULD: Fehler: x muss zwischen 0 und 1 liegen.
# OUT: Fehler: x muss zwischen 0 und 1 liegen.
# TEST 2:
# IN: 0.625
# SHOULD: 0.101
# OUT: 0.101
# TEST 3:
# IN: 0.3733
# SHOULD: 0.11011101011010010000100111111010
# OUT: 0.11011101011010010000100111111010

