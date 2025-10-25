import os
import numpy as np

if __name__ == "__main__":
    all_data = []

    with open("res.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line[: -1]
            items = line.split()
            if len(items) != 7:
                continue
            all_data.append([float(items[0]), float(items[-1])])

    all_data = np.array(all_data, dtype=np.float128)
    # all_data = np.loadtxt("res.txt")
    time_stamp = all_data[:, 0]

    time_interval = []

    for i in range(len(time_stamp) -1):
        time_interval.append(time_stamp[i + 1] - time_stamp[i])

    time_inte_array = np.array(time_interval)

    print("average time interval: \n", np.mean(time_inte_array))
    print("std deviation: \n", np.std(time_inte_array))
    print("output frequency: \n", 1.0 / np.mean(time_inte_array))
    print("time inte array: \n", time_inte_array)
    print("max value in array: \n", np.max(time_inte_array))
    print("min value in array: \n", np.min(time_inte_array))

    print("operation_times: \n", all_data[:, 1])
    print("average operatation time: \n", np.mean(all_data[:, 1]))
    print("standard deviation operation time: \n", np.std(all_data[:, 1]))

    np.where("")
