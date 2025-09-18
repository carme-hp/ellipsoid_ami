import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

with open('build_release/tendon.txt', 'r') as f:
    lines = f.readlines()
    tendont = [float(line.split()[0]) for line in lines]
    tendons = [float(line.split()[1]) for line in lines]
    tendone = [float(line.split()[2]) for line in lines]

with open('build_release/muscle_left.txt', 'r') as f:
    lines = f.readlines()
    leftt = [float(line.split()[0]) for line in lines]
    lefts = [float(line.split()[1]) for line in lines]
    lefte = [float(line.split()[2]) for line in lines]

with open('build_release/muscle_right.txt', 'r') as f:
    lines = f.readlines()
    rightt = [float(line.split()[0]) for line in lines]
    rights = [float(line.split()[1]) for line in lines]
    righte = [float(line.split()[2]) for line in lines]


# paper: markersize: 0.8, font.size: 18

tendon_length = tendone[1]-tendons[1]
left_muscle = lefte[1]-lefts[1]
right_muscle = righte[1]-rights[1]

plt.rcParams.update({'font.size': 16})
linewidthh = 2

plt.figure(figsize=(8, 8))

plt.plot(leftt[:-50],np.array(lefte[:-50]) - np.array(lefts[:-50])-left_muscle,linewidth=linewidthh,label="muscle 1", color="orange")
plt.plot(rightt[:-50],np.array(righte[:-50]) - np.array(rights[:-50])-right_muscle, linewidth=linewidthh,label="muscle 2", color="b")
plt.plot(tendont[:-50],np.array(tendone[:-50]) - np.array(tendons[:-50])-tendon_length, linewidth=linewidthh,label="tendon", color="green")
plt.xlabel("Time (ms)")
plt.ylabel("Change in length (cm)")
plt.legend()
plt.show()

plt.figure(figsize=(8, 8))

# Load data (skip the header by using 'skip_header')
data = np.loadtxt('build_release/precice-TendonSolver-iterations.log', delimiter=None, skiprows=1)

# Plot the first and second columns
plt.plot(np.array(data[:, 0])*0.01, data[:, 2])

# Add labels and title
plt.xlabel('Time (ms)')
plt.ylabel('# Coupling Iterations')

# Show the plot
plt.show()