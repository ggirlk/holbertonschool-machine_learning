#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

bins = np.arange(0, 11) * 10
plt.hist(student_grades, bins, edgecolor="#000")
plt.title("Project A")
plt.xlabel("Grades")
plt.ylabel("Number of Students")
plt.gca().set_ylim([0,30])
plt.show()
