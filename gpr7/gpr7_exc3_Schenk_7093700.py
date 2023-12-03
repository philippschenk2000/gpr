""" This script contains the new exercise in gpr """

__author__ = "7093700, Schenk"


def main():
    """ This function plays with the tupel received by the user.

    """
    try:
        # INPUT OF THE TUPLE
        #T = (('Anna', 'Mathematik'), ('Peter', 'Geologie'), ('Sina', 'Mathematik'), ('Sina', 'Geologie'))
        T = eval(input('Enter your tuple: '))
        # FORMAT THE TUPLE INTO TWO LISTS
        names, study_fields = zip(*T)
        a = len(list(set(names)))
        b = len(list(study_fields))
        most_frequent = ''
        max_frequency = 0
        # COUNT THE FREQUENCY OF THE STUDY FIELDS AND FIND MOST FREQUENT
        for i in list(set(study_fields)):
            print(i, list(study_fields).count(i))
            if int(list(study_fields).count(i)) > max_frequency:
                most_frequent = i
                max_frequency = int(list(study_fields).count(i))
            elif int(list(study_fields).count(i)) == max_frequency:
                most_frequent = most_frequent + ',' + i
        # CREATE TUPLE FOR OUTPUT BASED ON THE EXERCISES
        T_out = (('exercise_a', a), ('exercise_b', b), ('exercise_c', most_frequent))
        print(T_out)

    except ValueError:
        print('Please just enter correct tupel.')


main()


# TEST 1:
# IN: (('Anna', 'Mathematik'), ('Peter', 'Geologie'), ('Sina', 'Mathematik'), ('Sina', 'Geologie'))
# SHOULD: (('exercise_a', 3), ('exercise_b', 4), ('exercise_c', 'Geologie,Mathematik'))
# OUT: (('exercise_a', 3), ('exercise_b', 4), ('exercise_c', 'Geologie,Mathematik'))
# TEST 2:
# IN: (('Anna', 'Mathematik'), ('Peter', 'Geologie'), ('Sina', 'Mathematik'), ('Sina', 'Informatik'))
# SHOULD: (('exercise_a', 3), ('exercise_b', 4), ('exercise_c', 'Mathematik'))
# OUT: (('exercise_a', 3), ('exercise_b', 4), ('exercise_c', 'Mathematik'))
