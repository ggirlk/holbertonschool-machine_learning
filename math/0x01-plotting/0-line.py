#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3

plt.plot(y, color="red")
plt.axis([0, len(y)-1, 0, y[-1]])
plt.show()
