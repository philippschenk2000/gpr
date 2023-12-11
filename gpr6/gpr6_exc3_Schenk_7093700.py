""" This script contains the new exercise in gpr """

__author__ = "7093700, Schenk"


def main():
    """ This function checks one string (haystack) on the existence of one other string (needle).

    """
    # INPUT IS EVERY TIME A STRING
    needle = input('needle: ')
    haystack = input('haystack: ')
    # ITERATION THRU EVERY CHARACTER IN HAYSTACK:
    printing = -1
    for i in range(len(haystack) - len(needle)):
        print(haystack[i:i + len(needle)])
        if haystack[i:i + len(needle)] == needle:
            printing = i
            # WAS ALREADY FOUND ONCE
            break
    print(printing)


main()


# TEST 1:
# IN: du, hallo wie gehts dir du knecht
# SHOULD: 20
# OUT: 20
# TEST 2:
# IN: 124, 1389502105205380457930q
# SHOULD: -1
# OUT: -1
# TEST 3:
# IN: meine amen dha, amen
# SHOULD: -1
# OUT: -1
