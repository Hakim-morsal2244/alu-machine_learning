#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# Plot histogram
plt.hist(student_grades, bins=np.arange(0, 101, 10), edgecolor='black')

# Labels and title
plt.xlabel('Grades')
plt.ylabel('Number of Students')
plt.title('Project A')

# Show plot
plt.show()
