""" This script contains 5 different functions for the new exercise in epr """

__author__ = "7093700, Schenk"


def main():
    """ This function creates a list full of tuples containing a card color and a card value.

    """
    # INPUT IS EVERY TIME A STRING
    needle = input('needle: ')
    haystack = input('haystack: ')
    # ITERATION THRU EVERY CHARACTER IN HAYSTACK:
    printing = -1
    for i in range(len(haystack)):
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
