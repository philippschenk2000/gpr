""" This script contains the new exercise in gpr """

__author__ = "7093700, Schenk"


def main(n):
    """ This function calculates possible steps recursively.

    """
    # BASE: 0 AND UNDER 0 STEPS ARE POSSIBLE
    if n == 0:
        return 1
    if n < 0:
        return 0
    # ELSE: RECURSIVELY CALLING THE FUNCTION BUT WITH FEWER STEPS LEFT
    return main(n-1) + main(n-2) + main(n-3)


possibilities = main(29)
print(possibilities)


# TEST 1:
# IN: 5
# SHOULD: 13
# OUT: 13
# TEST 2:
# IN: 6
# SHOULD: 24
# OUT: 24
