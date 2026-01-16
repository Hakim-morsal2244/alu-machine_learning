#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1_data = np.random.multivariate_normal(mean, cov, 2000).T
y1_data += 180

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

# Create 3x2 grid
fig, axs = plt.subplots(3, 2, figsize=(10, 10))

# First plot: y0
axs[0, 0].plot(y0, 'b')
axs[0, 0].set_title('y0 Plot', fontsize='x-small')
axs[0, 0].set_xlabel('x', fontsize='x-small')
axs[0, 0].set_ylabel('y', fontsize='x-small')

# Second plot: x1 vs y1_data
axs[0, 1].scatter(x1, y1_data, c='m')
axs[0, 1].set_title('Scatter Plot', fontsize='x-small')
axs[0, 1].set_xlabel('x', fontsize='x-small')
axs[0, 1].set_ylabel('y', fontsize='x-small')

# Third plot: x2 vs y2
axs[1, 0].plot(x2, y2, 'g')
axs[1, 0].set_title('Exponential Decay', fontsize='x-small')
axs[1, 0].set_xlabel('Time', fontsize='x-small')
axs[1, 0].set_ylabel('Fraction Remaining', fontsize='x-small')

# Fourth plot: Task 3 lines
axs[1, 1].plot(x3, y31, 'r--', label='C-14')
axs[1, 1].plot(x3, y32, 'g-', label='Ra-226')
axs[1, 1].set_title('Exponential Decay', fontsize='x-small')
axs[1, 1].set_xlabel('Time (years)', fontsize='x-small')
axs[1, 1].set_ylabel('Fraction Remaining', fontsize='x-small')
axs[1, 1].set_xlim(0, 20000)
axs[1, 1].set_ylim(0, 1)
axs[1, 1].legend(fontsize='x-small', loc='upper right')

# Fifth plot: histogram
axs[2, 0].hist(student_grades, bins=np.arange(0, 101, 10), edgecolor='black')
axs[2, 0].set_title('Project A', fontsize='x-small')
axs[2, 0].set_xlabel('Grades', fontsize='x-small')
axs[2, 0].set_ylabel('Number of Students', fontsize='x-small')

# Sixth plot spans two columns
axs[2, 1].axis('off')  # leave empty or can merge axes if needed

# Main title
fig.suptitle('All in One')
plt.tight_layout()
plt.show()
