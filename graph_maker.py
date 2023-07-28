
import matplotlib.pyplot as plt
import numpy as np

# X = Run time, Y = Num Threads.
# Read Backwards Substitution Row, Col
# Read Init
# Read Gauss

# NumThreads
x = np.array([1, 2, 4, 8, 16, 32])
lables = [1, 2, 4, 8, 16, 32]
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams['figure.figsize'] = [10, 7]
# colors = colors[0:6]

# Making threads vs Bsub for Col

serial_y = np.array([])
serial_Parallel_y = np.array([])
parallel_y = np.array([])
# Read in the values for x
j = 0
with open("results__Gaus.txt", 'r') as f:
    # Read fist line to get Specific line
    while j < 8:  # number of size instances
        f.readline()  # Skip empty space
        points_name_key = f.readline()  # name of the plots
        # Read 2 more lines for serial
        f.readline()  # Skip Name
        serial = f.readline()
        # Read 2 more for Serial Parallel Version
        f.readline()
        serial_Parallel_y = f.readline()  # np.append(serial_Parallel_x, float)
        f.readline()
        # Read the parallel information
        for i in range(len(x)):
            # serial_parallel = f.readline().split("BSUB:   ")
            time = f.readline().split("  ")
            # print(serial_parallel)
            # index 3 is init
            # index 5 is gausian
            time = time[5][0:-2]
            # print(serial_parallel)
            parallel_y = np.append(parallel_y, float(time))
            # parallel_y = np.append(parallel_y, float(serial_parallel[1][: -2]))
        j += 1
        # Now add to the plot
        plt.scatter(x, parallel_y, color=colors[j % len(colors)])
        plt.plot(x, parallel_y, color=colors[j % len(colors)],
                 label=points_name_key)
        # plt.show()
        plt.xticks(x, lables)

        parallel_y = np.array([])

plt.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, 1.2))
plt.xlabel("Number of Threads")
plt.ylabel("Time in seconds")
plt.title("Run time of Gaussian Elimination")
plt.tight_layout()
plt.savefig("data_Gaus.png")
plt.show()
