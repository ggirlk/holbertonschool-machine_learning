#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

legends = ['apples', 'bananas', 'oranges', 'peaches']

colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

labels = ['Farrah', 'Fred', 'Felicia']

y_offset = np.zeros(len(labels))

width = 0.5
for i in range(0, len(fruit)):
    plt.bar(
        labels,
        fruit[i],
        width=width,
        label=legends[i],
        color=colors[i],
        bottom=y_offset
       )
    y_offset = y_offset + fruit[i]

plt.title('Number of Fruit per Person')
plt.ylabel("Quantity of Fruit")
plt.legend()
plt.gca().set_ylim([0, 80])
plt.show()
