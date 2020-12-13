#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 21000, 1000)
r = np.log(0.5)
t1 = 5730
t2 = 1600
y1 = np.exp((r / t1) * x)
y2 = np.exp((r / t2) * x)

plt.plot(x, y1, ls="--", color="#EE3B1D")
plt.plot(x, y2, color="#479000")
plt.xlim([0, 20000])
plt.title("Exponential Decay of Radioactive Elements")
plt.xlabel("Time (years)")
plt.ylabel("Fraction Remaining")
plt.legend(labels=("C-14", "Ra-226"), loc="upper right")

plt.show()
