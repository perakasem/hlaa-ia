import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import fsolve

def f(t, y, v, b, k):
    return -b * v - k * y

def backward_euler_step(t, y, v, h, b, k):
    # Function to represent the system of equations for Backward Euler
    def equations(next_vars):
        y_next, v_next = next_vars
        return [y_next - y - h * v_next, v_next - v - h * f(t + h, y_next, v_next, b, k)]

    # Solve for the next values of y and v
    y_next, v_next = fsolve(equations, [y, v])
    return y_next, v_next

# Damping and stiffness parameters
b = 0.5  # Damping coefficient
k = 0.1  # Stiffness coefficient

# Initial conditions
y0 = 1  # Initial position
v0 = 0  # Initial velocity

# Time parameters
t0 = 0
tf = 40
h = 0.01  # Step size
N = int((tf - t0) / h)

# Arrays for storing values
t_values = np.linspace(t0, tf, N)
y_values = []
v_values = []
analytical_values = []
errors = []

# Initial values
y = y0
v = v0

# Backward Euler Method
for i in range(10):
    start_time = time.time()
    for t in t_values:
        y, v = backward_euler_step(t, y, v, h, b, k)
        y_values.append(y)
        v_values.append(v)

    print((time.time() - start_time))