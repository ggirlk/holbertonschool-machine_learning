#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

gridsize = (3, 2)
fig = plt.figure(figsize=(12, 8))
ax1 = plt.subplot2grid(gridsize, (0, 0))
ax2 = plt.subplot2grid(gridsize, (0, 1))
ax3 = plt.subplot2grid(gridsize, (1, 0))
ax4 = plt.subplot2grid(gridsize, (1, 1))
ax5 = plt.subplot2grid(gridsize, (2, 0), colspan=2)

fig.suptitle('All in One')

ax1.plot(y0, color="red")

ax2.scatter(x1, y1, color="#CE4FCA")
ax2.set_title('Men\'s Height vs Weight')
ax2.set(xlabel='Height (in)', ylabel='Weight (lbs)')

ax3.plot(x2, y2)
ax3.set_title('Exponential Decay of C-14')
ax3.set(xlabel='Time (years)', ylabel='Fraction Remaining', yscale="log")

ax4.plot(x3, y31, ls="--", color="#EE3B1D")
ax4.plot(x3, y32, color="#479000")
ax4.set_title("Exponential Decay of Radioactive Elements")
ax4.set(xlabel='Time (years)', ylabel='Fraction Remaining')


bins = np.arange(0, 11) * 10
ax5.hist(student_grades, bins, edgecolor="black")
ax5.set_title("Project A")
ax5.set(xlabel="Grades", ylabel="Number of Students")
ax5.set_ylim([0, 30])

fig.tight_layout(pad=3.0)

plt.show()
