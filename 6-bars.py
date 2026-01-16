#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

labels = ['Farrah', 'Fred', 'Felicia']
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']  # apples, bananas, oranges, peaches

# Stacked bar chart
bottom = np.zeros(3)
for i in range(fruit.shape[0]):
    plt.bar(labels, fruit[i], bottom=bottom, color=colors[i], width=0.5, label=['Apples','Bananas','Oranges','Peaches'][i])
    bottom += fruit[i]

# Labels and title
plt.ylabel('Quantity of Fruit')
plt.ylim(0, 80)
plt.yticks(np.arange(0, 81, 10))
plt.title('Number of Fruit per Person')
plt.legend()

plt.show()
