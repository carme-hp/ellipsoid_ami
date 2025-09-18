import numpy as np
import matplotlib.pyplot as plt
import sys
import csv

parameter = sys.argv[1]

lengths_muscle_1 = []
lengths_muscle_2 = []

f = open("build_release/muscle_length_contraction"+parameter+"_1.csv", "r")
reader = csv.reader(f, delimiter=",")
for row in reader:
    for i in row:
        if i != "":
            lengths_muscle_1.append(float(i))
f.close()

f = open("build_release/muscle_length_contraction"+parameter+"_2.csv", "r")
reader = csv.reader(f, delimiter=",")
for row in reader:
    for i in row:
        if i != "":
            lengths_muscle_2.append(float(i))
f.close()

x = np.linspace(0,100,len(lengths_muscle_1))

plt.plot(x, lengths_muscle_1, label="muscle 1")
plt.plot(x, lengths_muscle_2, label="muscle 2")
plt.xlabel("time (ms)")
plt.ylabel("length of muscles (cm)")
plt.legend()
plt.show()