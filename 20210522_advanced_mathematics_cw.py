#Question 1:a)

import numpy as np
import sympy as sym
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
#Question 1:b)
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from scipy import integrate

#Defining variables
x = sym.symbols("x")
n = sym.symbols("n", integer='True', positive='True')

#an empty array to store the fourier series expansion
ms = np.empty(11, dtype=object)

#equations
eq1 = (x ** 2) + 1
eq2 = x * sym.exp(-x)

#Defining the general formulas of the fourier series co-efficients
a0 = (1 / sym.pi) * (eq1.integrate((x, -sym.pi, 0)) + eq2.integrate((x, 0, sym.pi)))
print("a0 = ", a0)

an = (1 / sym.pi) * (sym.integrate((eq1 * sym.cos(n * x)), (x, -sym.pi, 0)) + sym.integrate((eq2 * sym.cos(n * x)), (x, 0, sym.pi)))
print("an = ", an)

bn = (1 / sym.pi) * (sym.integrate((eq1 * sym.sin(n * x)), (x, -sym.pi, 0)) + sym.integrate((eq2 * sym.sin(n * x)), (x, 0, sym.pi)))
print("bn = ", bn)

#First value- a0/2
ms[0] = a0/2

#To iterate and place values inside the ms[] array
i = 1
#first 5 terms of the fourier series expansion for an values
for n in range(1, 6):
    an = (1 / sym.pi) * (sym.integrate((eq1 * sym.cos(n * x)), (x, -sym.pi, 0)) + sym.integrate((eq2 * sym.cos(n * x)), (x, 0, sym.pi))) * sym.cos(n * x)
    ms[i] = an
    i += 1

i = 6
#first 5 terms of the fourier series expansion for bn values
for n in range(1, 6):
    bn = (1 / sym.pi) * (sym.integrate((eq1 * sym.sin(n * x)), (x, -sym.pi, 0)) + sym.integrate((eq2 * sym.sin(n * x)), (x, 0, sym.pi))) * sym.sin(n * x)
    ms[i] = bn
    i += 1
print("Fourier Series Expansion up to first 5 terms = ", ms, end="")
#Question 1:c)
import math
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from scipy.signal.bsplines import arange
for m in range(0,1000):
  if 0 < arange[m] < np.pi:
    y[100,m]=arange[m]**2
  elif 2*np.pi < arange[m] <= 3*np.pi:
    y[100,m]= (arange[m] -2*np.pi)**2
  elif -2*np.pi < arange[m] <= np.pi:
    y[100,m]= (arange[m] +2*np.pi)**2
plt.plot(arange, y[0,:])
plt.plot(arange, y[4,:])
plt.plot(arange, y[149,:])
plt.legend(["1", "5", "150","func"])
plt.show()
#Question 1:d)
import math
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from scipy.signal.bsplines import arange
def f(x):
    return np.where((x>= -np.pi)&(x<0),(x**2)+1,np.where((x>=0)&(x<np.pi),x*np.exp(-x),0))
def har(x,n):
    return np.sin(n*x)/n
def rmse(f,har,x):
    return np.sqrt(np.mean(np.square(f-har)))
    
xrange = np.linspace(-4*np.pi, 4*np.pi, 1000)
RMSE = []
for n in range(1,151):
    RMSE.append(rmse(f(xrange),har(xrange,n),xrange))

print("root mean square error between f(x) and It's first harmonic = ", RMSE[0] )
print("root mean square error between f(x) and Upto it's fifth harmonic = ", RMSE[4])
print("root mean square error between f(x) and Up to the 150th harmonic = ",RMSE[149])
#Question 2
#range valuess
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as sfft

Range1 = np.arange(-np.pi, np.pi, 0.5)
Range2 = np.arange(-np.pi, np.pi, 0.05)

#functions
Equation1 = np.sin(Range1) + 0.25 * np.cos(2 * Range1)
Equation2 = np.sin(2 * Range2) + np.sin(2 * Range2)

#fast fourier transformation
Equation1_dft1 = sfft.fft(Equation1)
Equation2_dft2 = sfft.fft(Equation2)

#plotting
plt.plot(Range1, Equation1)
plt.plot(Range2, Equation2)

#plotting of transformed functions 
plt.plot(Range1, np.real(Equation1_dft1))
plt.plot(Range2, np.real(Equation2_dft2))

plt.legend(["200Hz", "100Hz", "fft 200Hz", "fft 100Hz"])
plt.show()

#I have taken two frequency signals with 200Hz and 100Hz.it will plot a graph with two peaks. 
#But because of misidentification of a signal frequency, it might show additional change in frequency values if the sample rate is not high enough. 
#Therefore, in conclusion we can see that if the sampling rate isn't accurate to show high-frequencies which are prone to aliasing
#Question 3:a)
import math
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

x = sym.symbols('x')
eq = x*sym.cos(x/2)
f = sym.lambdify(x,eq,'numpy')
x1 = np.arange(-5*math.pi,7*math.pi+1,0.01)
y1 = f(x1)
plt.plot(x1, y1)
plt.show()
#Question 3:b)
def func_cos(x,n):
    cos_approx = 0
    for i in range(n):
        coef = (-1)**i
        num = x**(2*i)
        denom = math.factorial(2*i)
        cos_approx += (coef)*((num)/(denom))

    return cos_approx

angle_rad = (math.radians(90))
out = func_cos(angle_rad,5)
print(out)
#Question 3:c)
x = sym.symbols('x')
eq = sym.cos(x)
ms = np.empty(60,dtype=object)

xrange = np.linspace(5,-5,500)
y = np.zeros([61,500])
ms[0] = eq.subs(x,0)
ms[0]
f= sym.lambdify(x,ms[0],'numpy')
y[0,:] = f(xrange)

for n in range(1,60):
  ms[n]= ms[n-1]+eq.diff(x,n).subs(x,0)*(x**n)/(np.math.factorial(n))
  print(n+1, ":", ms[n])
  f=sym.lambdify(x, ms[n],'numpy')
  y[n,:] = f(xrange)

f= sym.lambdify(x,eq,'numpy')
y[20,:] = f(xrange)
plt.plot(xrange, y[0,:])
plt.plot(xrange, y[4,:])
plt.plot(xrange, y[9,:])
plt.plot(xrange, y[60,:])

plt.legend(["1","5","10","60"])
plt.show()
#Question 3:d)
import math
import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
def func_cos(x, n):
    cos_approx = 0
    for i in range(n):
        coef = (-1)**i
        num = x**(2*i)
        denom = math.factorial(2 * i)
        cos_approx += (coef)*((num)/(denom))
    return cos_approx
angle_rad = (math.radians(30))
coef_term = math.radians(60)
out_new1 = coef_term*func_cos(angle_rad,5)
print(out_new1)
angles_d = np.arange(-2*np.pi,2*np.pi,0.1)
p_cos_d = np.cos(angles_d)
t_cos_d = [coef_term*func_cos(angle,3) for angle in angles_d]
fig, ax = plt.subplots()
ax.plot(angles_d,p_cos_d)
ax.plot(angles_d,t_cos_d)
ax.set_ylim([-10,10])
ax.legend(['Taylor Series -3terms','cos() function'])
plt.show()
original_value = coef_term *  math.cos(angle_rad)
print(original_value)
#Question 04:a)
import numpy as np
import scipy.fftpack as sfft
import matplotlib.pyplot as plt

img = mpimg.imread("/Fruit.jpg")

#fft
imgf = sfft.fft2(img)
plt.imshow(np.abs(imgf))
plt.show()

#with fft shift
imgf = sfft.fftshift(imgf)
plt.imshow(np.abs(imgf))
plt.show()

#inverse fft
img1 = sfft.ifft2(imgf)
plt.imshow(np.abs(img1))
plt.show()

#removing high frequencies
imgf1 = np.zeros((360,360),dtype=complex)
c = 180
r = 50
for m in range(0,360):
    for n in range(0,360):
        if (np.sqrt(((m-c)**2 + (n-c)**2))<r):
            imgf1[m,n] = imgf[m,n]

plt.imshow(np.abs(imgf1))
plt.show()

img1 = sfft.ifft2(imgf1)
plt.imshow(np.abs(img1))
plt.show()

#removing low frequencies
imgf1 = np.zeros((360,360),dtype=complex)
c = 180
r = 90
for m in range(0,360):
    for n in range(0,360):
        if (np.sqrt(((m-c)**2 + (n-c)**2))>r):
            imgf1[m,n] = imgf[m,n]

plt.imshow(np.abs(imgf1))
plt.show()

img1 = sfft.ifft2(imgf1)
plt.imshow(np.abs(img1))
plt.show()
#Quetion 4:b)
import numpy as np
import scipy.fftpack as sfft
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread("/Fruit.jpg")

kernel = np.outer(signal.gaussian(360, 5), signal.gaussian(360, 5))
kf = sfft.fft2(sfft.ifftshift(kernel)) 

plt.imshow(np.abs(kf))
plt.show()
imgf = sfft.fft2(img)

plt.imshow(np.abs(kf))
plt.show()
img_b = imgf*kf

plt.imshow(np.abs(img_b))
plt.show()
img1 = sfft.ifft2(img_b)

plt.imshow(np.abs(img1))
plt.show()
#Quetion 4:c)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.fftpack as sfft

img = mpimg.imread("/Fruit.jpg")

#DCT
imgc = sfft.dct((sfft.dct(img,norm='ortho')).T,norm='ortho')
plt.imshow(imgc)
plt.show()

#IDCT
img1 = sfft.idct((sfft.idct(imgc,norm='ortho')).T,norm='ortho')
plt.imshow(img1)
plt.show()

#Removing high frequency components
imgc1 = np.zeros((360,360))
imgc1[:120,:120] = imgc[:120,:120]
img1 = sfft.idct((sfft.idct(imgc1,norm='ortho')).T,norm='ortho')
plt.imshow(img1)
plt.show()

#Scaling
imgc2 = imgc[0:120,0:120]
img1 = sfft.idct((sfft.idct(imgc2,norm='ortho')).T,norm='ortho')
plt.imshow(img1)
plt.show()
#Question 4:d)
import matplotlib.image as mpimg
img = mpimg.imread("/Fruit.jpg")
imgc = sfft.dct((sfft.dct(img, norm='ortho')).T, norm='ortho')
mgc1 = np.zeros((360, 360))
imgc1 = imgc[:120, :120]

img1 = sfft.idct((sfft.idct(imgc1, norm='ortho')).T, norm='ortho')
plt.imshow(imgc)
plt.show()
#Question 5:a)
import math
import sympy as sym
import matplotlib.pyplot as plt
import numpy as np
from sympy import diff
from sympy.abc import t, y
x = sym.symbols('x')
eq = 1/(1 + sym.exp(-x))
f = sym.lambdify(x,eq,'numpy')
range = np.arange(1,10,0.01)
y = f(range)
plt.plot(range, y)
plt.show()
#Question 5:b)
import math
import sympy as sym
import matplotlib.pyplot as plt
import numpy as np
from sympy.abc import t, y
x = sym.symbols('x')
eq1 = 1/(1 + sym.exp(-x))
eq2 = sym.diff(eq1)
f = sym.lambdify(x, eq2, 'numpy')
valuesOfx = np.arange(-100, 100, 0.01)
valuesOfy = f(valuesOfx)
plt.plot(valuesOfx, valuesOfy)
plt.show()
#Question 5:c) a)
import math
import sympy as sym
import matplotlib.pyplot as plt
import numpy as np
from sympy import diff
from sympy.abc import t, y
x1 = sym.symbols('2x')
eq2 = sym.sin(sym.sin(x1))
f2 = sym.lambdify(x1,eq2,'numpy')
range_2 = np.arange(-30,30,0.1)
y2 = f2(range_2)
plt.plot(range_2, y2)
plt.show()
#Question 5:c) b)
import math
import numpy as np
import sympy as sym
from sympy import diff
from sympy.abc import t, y
import matplotlib.pyplot as plt
x = sym.symbols('x')
eq = (-x*3)-(2*x**2)+(3*x)+10
f = sym.lambdify(x, eq, 'numpy')
valuesOfx =  np.arange(-100, 100, 0.01)
valuesOfy = f(valuesOfx)
plt.plot(valuesOfx, valuesOfy)
plt.show()
#Question 5:c) c)
import math
import sympy as sym
import matplotlib.pyplot as plt
import numpy as np
from sympy.abc import t, y
eq4 = sym.exp(-0.8*x)
f4 = sym.lambdify(x, eq4, 'numpy')
range_4 = np.arange(-30, 30,0.1)
y4 = f4(range_4)
plt.plot(range_4, y4)
plt.show()
#Question 5:c) d)
import math
import sympy as sym
import matplotlib.pyplot as plt
import numpy as np
from sympy.abc import t, y
eq5 = x**2*sym.cos(sym.cos(2*x)) - 2*sym.sin(sym.sin(x-(math.pi/3)))
f5 = sym.lambdify(x, eq5, 'numpy')
range_5 = np.arange(-30, 30, 0.1)
y5 = f5(range_5)
plt.plot(range_5, y5)
plt.show()
#Question 5:c) e)
import math
import sympy as sym
import matplotlib.pyplot as plt
import numpy as np
from sympy.abc import t, y
# defining the function
# if − 𝜋 ≤ 𝑥 < 0 the function is 2 * sym.cos(x + sym.pi / 6)
# else if 0 ≤ 𝑥 ≤ 𝜋 the function is x * sym.exp(-0.4 * (x ** 2))
# else if x < -𝜋 add 2𝜋 to x
# else if x > 𝜋 reduce 2𝜋 to x
def g(x):
    if (x >= -sym.pi) and (x < 0):
        y = 2 * sym.cos(x + sym.pi / 6)
    if (x >= 0) and (x < sym.pi):
        y = x * sym.exp(-0.4 * (x ** 2))
    if x < -sym.pi:
        X = x + 2 * sym.pi
        y = g(X)
    if x > sym.pi:
        X = x - 2 * sym.pi
        y = g(X)
    return y


# x-axis values are with thin range of [-4π,4π].
valuesOfx = np.arange(-4 * np.pi, 4 * np.pi, 0.01)
# y-axis values are found by substituting valuesOfx into the correct function
# therefore y values = [g(x) for x in valuesOfx]
plt.plot(valuesOfx, [g(x) for x in valuesOfx])
plt.show()
#Question 5:d) a)
import math
import sympy as sym
import matplotlib.pyplot as plt
import numpy as np
from sympy.abc import t, y
x=sym.symbols('x')
eq = (1/1 + sym.exp(-(sym.sin(sym.sin(2*x)))))
f = sym.lambdify(x, eq, 'numpy')
valuesOfx = np.arange(-100, 100, 0.01)
valuesOfy = f(valuesOfx)
plt.plot(valuesOfx, valuesOfy)
plt.show()
#Question 5:d) b)
import math
import sympy as sym
import numpy as np
from sympy.abc import t, y
x=sym.symbols('x')
eq = 1/(1 + sym.exp(-(-x**3 - 2*x**2 + 3*x + 10)))
f = sym.lambdify(x, eq, 'numpy')
valuesOfx = np.arange(-100, 100, 0.01)
valuesOfy = f(valuesOfx)
plt.plot(valuesOfx, valuesOfy)
plt.show()
#Question 5:d) c)
import math
import sympy as sym
import numpy as np
from sympy.abc import t, y
x=sym.symbols('x')
eq = 1/(1 + sym.exp(-sym.exp(-0.8*x)))
f = sym.lambdify(x, eq, 'numpy')
valuesOfx = np.arange(-100, 100, 0.01)
valuesOfy = f(valuesOfx)
plt.plot(valuesOfx, valuesOfy)
plt.show()
#Question 5:d) d)
import math
import sympy as sym
import numpy as np
from sympy.abc import t, y
x=sym.symbols('x')
eq = 1/(1 + sym.exp(-x**2*sym.cos(sym.cos(2*x)) - 2*sym.sin(sym.sin(x-(math.pi/3)))))
f = sym.lambdify(x, eq, 'numpy')
valuesOfx = np.arange(-100, 100, 0.01)
valuesOfy = f(valuesOfx)
plt.plot(valuesOfx, valuesOfy)
plt.show()
#Question 5:d) e)
import math
import sympy as sym
import numpy as np
from sympy.abc import t, y
def g(x):
    if (x >= -sym.pi) and (x < 0):
        y = 2 * sym.cos(x + sym.pi / 6)
    if (x >= 0) and (x < sym.pi):
        y = x * sym.exp(-0.4 * (x ** 2))
    if x < -sym.pi:
        X = x + 2 * sym.pi
        y = g(X)
    if x > sym.pi:
        X = x - 2 * sym.pi
        y = g(X)
    return y
valuesOfx = np.arange(-4*np.pi, 4*np.pi, 0.01)
valuesOfy = f(valuesOfx)
plt.plot(valuesOfx, [g(x) for x in valuesOfx])
plt.show()
