def main():
    __author__ = "7093700, Schenk"
    try:
        a = float(input('First number: '))
        b = float(input('Second number: '))
        while b != 0:
            h = a % b
            a = b
            b = h
            print(a)
    except:
        print('Input has to be a number.')



main()
