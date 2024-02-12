import numpy as np
import matplotlib.pyplot as plt

plt.figure()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Graph of f(x) = e^(-(x-u)^2)')
plt.legend("f(x) = e^(-(x-u)^2)")

mu = 0
x = np.linspace(-10, 10, 1000)

y_fit = np.exp(-(x-mu)**2)

plt.grid()
plt.legend()

plt.plot(x, y_fit)

plt.show()