def main():
    __author__ = "7093700, Schenk"
    try:
        zahl = int(input('Enter a beautiful number: '))
        if zahl == 0:
            print('The binary number would be: \n0')
        elif zahl < 0:
            print('BLABLA')
        else:
            binary_list = []
            while zahl != 0:
                rest = zahl % 2
                ohnerest = zahl - rest
                zahl = ohnerest / 2
                print(zahl, 'rest is:', rest)
                binary_list.append(rest)
            reversed_list = list(reversed(binary_list))
            # print(reversed_list)
            print('The binary number would be:')
            for r in reversed_list:
                print(str(int(r)), end='')
    except:
        print('Input has to be a integer.')


main()

# 11 Dezimal in binary:
#11 : 2 = 5 R 1
# 5 : 2 = 2 R 1
# 2 : 2 = 1 R 0
# 1 : 2 =

# Hexa / Oktal?
# 1011 = 1*2^0 + 1*2^1 + 0*2^2 + 1*2^3 = 1+2+8 = 11