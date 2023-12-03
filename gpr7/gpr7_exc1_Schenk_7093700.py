""" This script contains the new exercise in gpr """

__author__ = "7093700, Schenk"


def main():
    """ This function plays with the coding of two numbers by the user.

    """
    try:
        # INPUT THE TWO NUMBERS AND THEN TEST FOR INTEGER
        a = int(input('First number: '))
        b = int(input('Second number: '))
        coding = '⨂⊟⊚⊛⊘⨁⊙✶✷✸'
        c = a + b

        complete_coding = ''
        # FOR LOOP FOR CODING EACH NUMBER
        for n in range(0, len([a, b, c])):
            number = [a, b, c][n]
            for count in str(number):
                complete_coding = complete_coding + coding[int(count)]
            if n == 0:
                complete_coding = complete_coding + ' + '
            elif n == 1:
                complete_coding = complete_coding + ' = '
        # EVERYTHING IS CODED, SO READY FOR OUTPUT
        print(complete_coding)
    except ValueError:
        print('Please just enter complete numbers.')


main()

