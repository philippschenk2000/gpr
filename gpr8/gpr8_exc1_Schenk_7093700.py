""" This script contains the new exercise in gpr """

__author__ = "7093700, Schenk"


def main(data):
    """ This function selects the best three students from a dictionary.

    """
    lecture = 'EPI'
    grades = []

    # ITERATE THE DICT AND SEARCH FOR THE SPECIFIED LECTURE
    for name, daten in data.items():
        if daten[0] == lecture:
            # COLLECT NAME AND MEAN GRADE FOR USER
            mean_grade = sum(daten[1]) / len(daten[1])
            grades.append((name, mean_grade))

    # SORT THE LIST DESCENDING
    #print(grades)
    grades.sort(key=lambda x: x[1])

    # SELECT THE BEST THREE STUDENTS
    best_three = dict(grades[:3] if len(grades) > 2 else grades)
    return best_three


# Beispiel-Daten
data = {'Max': ['EPI', (1.0, 1.3)], 'Anna': ['EPI', (1.7, 2.0)], 'Tim': ['EPI', (2.3, 2.7)],
        'Lena': ['EPI', (1.0, 1.0)], 'Tom': ['Mathematik', (1.0, 1.0)]}

# Funktion aufrufen
best = main(data)
print(best)
