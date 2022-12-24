# -*- coding: utf-8 -*-
"""20210522_Advanced_Mathematics_CW.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15lgbFNqiHt8vb1S9h6Kotvy1ktbVE1Ws
"""

#Question 1:a)

import sympy as sym
import numpy as np
import matplotlib.pyplot as plt

x = sym.symbols('x')

def f(x):
    if 0 > x >= -np.pi:
        return x ** 2 + 1
    elif 0 <= x <= np.pi:
        return x * np.exp(-1)

xValues = np.arange(-4 * np.pi, 4 * np.pi, 0.01)
yValues = [f(x) for x in xValues]

plt.title("PF")
plt.xlabel=("y")
plt.ylabel=("x")
plt.plot(xValues, yValues)
plt.show()

#Question 01:b)
import math
import numpy as np
import sympy as sym
from scipy import integrate
import matplotlib.pyplot as plt

x = sym.symbols('x')
def f(x):
    if - np.pi <= x < 0:
        return x ** 2 + 1
    elif 0 <= x <= np.pi:
        return x * np.exp(-1)

a0 = (1 / np.pi) * integrate.quad(f, -np.pi, np.pi)[0]
a = [0]
b = [0]

for i in range(6):
    a.append((2/np.pi) * integrate.quad(lambda x: f(x) * np.cos(i*x), -np.pi, np.pi)[0])
    b.append((2/np.pi) * integrate.quad(lambda x: f(x) * np.sin(i*x), -np.pi, np.pi)[0])

def f(x):
    result = a0
    for n in range(6):
        result += a[n-1] * np.cos(n*x) + b[n-1] * np.sin(n*x)
    return result

print(f(4))
