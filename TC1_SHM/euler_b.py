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

def analytical_solution(t, y0, v0, omega):
    A = y0
    B = v0 / omega
    return A * np.cos(omega * t) + B * np.sin(omega * t)

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

# Backward Euler Method
start_time = time.time()
for t in t_values:
    y, v = backward_euler_step(t, y, v, h, omega)
    y_values.append(y)
    v_values.append(v)

    y_analytical = analytical_solution(t, y0, v0, omega)
    analytical_values.append(y_analytical)
    errors.append(abs(y - y_analytical))

print("--- %s seconds ---" % (time.time() - start_time))

# Calculate RMSE
rmse = np.sqrt(np.mean(np.array(errors)**2))
print("RMSE:", rmse)

# Plotting the results and error
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(t_values, y_values, label='Backward Euler Solution')
plt.plot(t_values, analytical_values, label='Analytical Solution', linestyle='dashed')
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Comparison of Numerical and Analytical Solutions')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t_values, errors, label='Error')
plt.xlabel('Time')
plt.ylabel('Error')
plt.title('Error Over Time')
plt.legend()

plt.tight_layout()
plt.show()
