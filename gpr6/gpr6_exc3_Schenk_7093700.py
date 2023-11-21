def main():
    __author__ = "7093700, Schenk"

    needle = input('needle: ')
    haystack = input('haystack: ')
    # INPUT IS EVERY TIME A STRING

    # IF NEEDLE IN HAYSTACK:
    for i in range(len(haystack)):
        print(haystack[i:i + len(needle)])
        if haystack[i:i + len(needle)] == needle:
            print(-1)
            break


main()

