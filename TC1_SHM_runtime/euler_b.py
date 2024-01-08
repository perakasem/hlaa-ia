import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import fsolve

def f(t, y, omega):
    return -omega**2 * y

def backward_euler_step(t, y, v, h, omega):
    def equations(next_vars):
        y_next, v_next = next_vars
        return [y_next - y - h * v_next, v_next - v - h * f(t + h, y_next, omega)]

    y_next, v_next = fsolve(equations, [y, v])
    return y_next, v_next


# Parameters and initial conditions
omega = np.sqrt(10)  # Adjust as needed
y0 = 1
v0 = 0
t0 = 0
tf = 40
h = 0.01
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
        y, v = backward_euler_step(t, y, v, h, omega)
        y_values.append(y)
        v_values.append(v)
    print((time.time() - start_time))
