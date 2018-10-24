# Author: Mikita Sazanovich

import csv
import matplotlib.pyplot as plt


with open('rewards.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    y = []
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            y.append(row[2])
            line_count += 1
    print(f'Processed {line_count} lines.')
    plt.plot(list(map(lambda x: float(x) > 1, y)))
    plt.show()
