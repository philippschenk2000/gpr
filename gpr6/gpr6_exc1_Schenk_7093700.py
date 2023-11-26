""" This script contains the new exercise in gpr """

__author__ = "7093700, Schenk"


def main():
    """ This function plays with the string how it should be done in the exercise.

    """
    text = "Ich mag programmieren"
    a = text[::-1][::2]
    print(a)
    b = text.split(' ')[1]
    print(b)
    c = text.replace(text.split(' ')[-1], '')
    print(c)
    d = text[1:]
    print(d)
    e = text[-2:][::-1]
    print(e)
    for g in range(len(text), 0, -1):
        if g in [2, 4, 6, 8]:
            print(text[len(text) - g:len(text) - g+1], end='')


main()

