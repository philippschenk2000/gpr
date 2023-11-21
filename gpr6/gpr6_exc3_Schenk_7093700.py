def main():
    __author__ = "7093700, Schenk"

    needle = input('needle: ')
    haystack = input('haystack: ')
    # INPUT IS EVERY TIME A STRING

    # ITERATION THRU EVERY CHARACTER IN HAYSTACK:
    printing = -1
    for i in range(len(haystack)):
        #print(haystack[i:i + len(needle)])
        if haystack[i:i + len(needle)] == needle:
            printing = i
            # WAS ALREADY FOUND
            break
    print(printing)


main()


# TEST 1:
# IN: du, hallo wie gehts dir du knecht
# SHOULD: -1
# OUT: -1
# TEST 2:
# IN: du, hallo wie gehts dir du knecht
# SHOULD: -1
# OUT: -1
# TEST 3:
# IN: du, hallo wie gehts dir du knecht
# SHOULD: -1
# OUT: -1
