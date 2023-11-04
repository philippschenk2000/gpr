def main():
    a = float(input('First number: '))
    b = float(input('Second number: '))
    while b != 0:
        h = a % b
        a = b
        b = h
        print(a)




main()
