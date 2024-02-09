import numpy as np
import matplotlib.pyplot as plt
import time

def f(t, y, v, b, k):
    return -b * v - k * y

def rk_midpoint_step(t, y, v, h, b, k):
    # Midpoint estimates for y and v
    k1_v = f(t, y, v, b, k)
    k1_y = v
    midpoint_v = v + 0.5 * h * k1_v
    midpoint_y = y + 0.5 * h * k1_y

    # Final update using midpoint estimates
    k2_v = f(t + 0.5 * h, midpoint_y, midpoint_v, b, k)
    k2_y = midpoint_v
    y_next = y + h * k2_y
    v_next = v + h * k2_v

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

for i in range(100):
    # RK2 Midpoint Method
    start_time = time.time()
    for t in t_values:
        y, v = rk_midpoint_step(t, y, v, h, b, k)
        y_values.append(y)
        v_values.append(v)

    print((time.time() - start_time))