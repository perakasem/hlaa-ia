import numpy as np
import matplotlib.pyplot as plt
import time

def f(t, y, v, mu):
    return mu * (1 - y**2) * v - y


def rk_midpoint_step(t, y, v, h, b):
    # Midpoint estimates for y and v
    k1_v = f(t, y, v, b)
    k1_y = v
    midpoint_v = v + 0.5 * h * k1_v
    midpoint_y = y + 0.5 * h * k1_y

    # Final update using midpoint estimates
    k2_v = f(t + 0.5 * h, midpoint_y, midpoint_v, b)
    k2_y = midpoint_v
    y_next = y + h * k2_y
    v_next = v + h * k2_v

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

# RK2 Midpoint Method
for i in range(10):
    start_time = time.time()
    for t in t_values:
        y, v = rk_midpoint_step(t, y, v, h, b)
        y_values.append(y)
        v_values.append(v)
    print((time.time() - start_time))
