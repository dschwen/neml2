import numpy as np
import matplotlib.pyplot as plt

data1 = np.loadtxt("result1.csv", delimiter=",")
data2 = np.loadtxt("result2.csv", delimiter=",")
plt.plot(data1[:, 0], data1[:, 1], label="parameter_4 = 0")
plt.plot(data2[:, 0], data2[:, 1], label="parameter_4 = 1")
plt.xlabel("strain")
plt.ylabel("stress")
plt.legend()
plt.tight_layout()
plt.savefig("comparison.png")
