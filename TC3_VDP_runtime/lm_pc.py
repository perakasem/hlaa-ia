import numpy as np
import matplotlib.pyplot as plt
import time

def f(t, y, v, mu):
    return mu * (1 - y**2) * v - y


def predictor_corrector_step(t, y, v, v_prev, h, b):
    # Predictor (Euler method)
    v_predict = v + h * f(t, y, v, b)
    y_predict = y + h * v

    # Corrector (Adams-Bashforth method)
    if v_prev is not None:
        v_correct = v + h * (1.5 * f(t + h, y_predict, v_predict, b) - 0.5 * f(t, y, v_prev, b))
    else:
        # For the first step, fall back to Euler method as we don't have v_prev
        v_correct = v_predict
    y_correct = y + 0.5 * h * (v + v_correct)

    return y_correct, v_correct, v_predict

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

# Predictor-Corrector Method

# Main loop
for i in range(10):
    start_time = time.time()
    v_prev = None
    for t in t_values:
        y, v, v_predict = predictor_corrector_step(t, y, v, v_prev, h, b)
        y_values.append(y)
        v_values.append(v)

        v_prev = v  # Update v_prev  for the next iteration

    print((time.time() - start_time))
