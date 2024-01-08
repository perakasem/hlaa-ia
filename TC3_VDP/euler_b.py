import numpy as np
import time
from scipy.optimize import fsolve

def f(t, y, v, mu):
    return mu * (1 - y**2) * v - y

def backward_euler_step(t, y, v, h, b):
    # Function to represent the system of equations for Backward Euler
    def equations(next_vars):
        y_next, v_next = next_vars
        return [y_next - y - h * v_next, v_next - v - h * f(t + h, y_next, v_next, b)]

    # Solve for the next values of y and v
    y_next, v_next = fsolve(equations, [y, v])
    return y_next, v_next

# Damping and stiffness parameters
b = 3  # Damping coefficient

# Initial conditions
y0 = 0.1  # Initial position
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

for i in range(10):
    # Backward Euler Method
    start_time = time.time()
    for t in t_values:
        y, v = backward_euler_step(t, y, v, h, b)
        y_values.append(y)
        v_values.append(v)

    print((time.time() - start_time))