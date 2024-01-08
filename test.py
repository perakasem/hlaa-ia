import numpy as np
from scipy.optimize import fsolve

def f(t, y, omega):
    return -omega**2 * y

def backward_euler_first_iteration(y0, v0, h, omega):
    def equations(next_vars):
        y_next, v_next = next_vars
        return [y_next - y0 - h * v_next, v_next - v0 - h * f(h, y_next, omega)]

    y_next, v_next = fsolve(equations, [y0, v0])
    return y_next, v_next

# Parameters and initial conditions
omega = np.sqrt(10)  # Angular frequency
y0 = 1  # Initial position
v0 = 0  # Initial velocity
h = 0.01  # Time step

# Solve the first iteration
y_next, v_next = backward_euler_first_iteration(y0, v0, h, omega)

print("y_next:", y_next)
print("v_next:", v_next)
