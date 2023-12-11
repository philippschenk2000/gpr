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


possibilities = main(6)
print(possibilities)
