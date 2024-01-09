""" This script contains the new exercise in gpr """


import re

__author__ = "7093700, Schenk"


def main():
    """ This function reads a .txt file, prints results to the console and calls further functions.

    """
    # READ THE TXT
    filename = 'Wortgitter_mitZahlen.txt'
    with open(filename, 'r') as file:
        text = file.read()
    print(text)
    # EMPTY LISTS FOR ALL POSSIBILITIES LATER
    exercise_ii_list = []
    exercise_iii_list = []
    for i in range(0, len(text) - 2):
        # EXERCISE II (BUT ALSO CONTAINS THE EXERCISE I)
        exercise_ii = test_for_null_number_character(text[i], text[i + 1], text[i + 2])
        # EXERCISE III
        exercise_iii = test_for_two_or_three_numbers(text[i], text[i + 1], text[i + 2])

        # APPEND EVERYTHING BUT FALSE
        if exercise_ii:
            exercise_ii_list.append(exercise_ii)
        if exercise_iii:
            exercise_iii_list.append(exercise_iii)

    # TO THE CONSOLE
    if len(exercise_ii_list) > 0:
        exercise_i = exercise_ii_list[0]
        print('Exercise i (first one):\n', exercise_i)
    print('Exercise ii (all possible):\n', exercise_ii_list)
    print('Exercise iii (all possible with 2 or 3 numbers):\n', exercise_iii_list)

    # Umgang mit mehr als 3 Zahlen hintereinander (Bsp: '52515')
    # '52515' --> ['525', '251', '515', '15']
    # PROGRAMM GIBT NICHT DIE GESAMTE ZAHL '52515' AUS, SONDERN BLEIBT BEI EINER BEVORZUGTEN LÄNGE VON 3 EINZELNEN ZAHLEN
    # UND SPEICHERT ALLE MÖGLICHKEITEN AB





def test_for_null_number_character(first, next, last):
    """ This function tests parts of the .txt file for specific combinations of characters.

    """
    # CHECK IF THE FIRST ONE IS A ZERO
    if first == '0':
        # CHECK IF THE NEXT ONE IS A NUMBER
        if next in [str(g) for g in range(0, 9)]:
            new_last = re.sub('[A-Z]', '', last.upper())
            # CHECK IF LAST WAS A CHARACTER (WITH CUTTING OUT THE A-Z)
            if len(new_last) == 0:
                return first + next + last
    return False


def test_for_two_or_three_numbers(first, next, last):
    """ This function tests parts of the .txt file for specific combinations of characters.

    """
    # CHECK IF THE FIRST ONE AND NEXT ONE IS A NUMBER
    if first in [str(g) for g in range(0, 9)] and next in [str(g) for g in range(0, 9)]:
        # IN CASE THE THIRD ONE ALSO IS A NUMBER, ELSE IT JUST RETURNS THE COMBINATION OF TWO CHARACTERS
        if last in [str(g) for g in range(0, 9)]:
            return first + next + last
        return first + next
    return False


main()
