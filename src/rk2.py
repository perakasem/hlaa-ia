import numpy as np
import matplotlib.pyplot as plt
import time

def f(t, y, omega):
    return -omega**2 * y

def rk_midpoint_step(t, y, v, h, omega):
    k1_v = f(t, y, omega)
    k1_y = v
    midpoint_v = v + 0.5 * h * k1_v
    midpoint_y = y + 0.5 * h * k1_y
    k2_v = f(t + 0.5 * h, midpoint_y, omega)
    k2_y = midpoint_v
    y_next = y + h * k2_y
    v_next = v + h * k2_v
    return y_next, v_next

def analytical_solution(t, y0, v0, omega):
    A = y0
    B = v0 / omega
    return A * np.cos(omega * t) + B * np.sin(omega * t)

omega = np.sqrt(10)  # Angular frequency
y0 = 1
v0 = 0
t0 = 0
tf = 40
h = 0.01
N = int((tf - t0) / h)

t_values = np.linspace(t0, tf, N)
y_values = []
v_values = []
analytical_values = []
errors = []

y = y0
v = v0

start_time = time.time()
for t in t_values:
    y, v = rk_midpoint_step(t, y, v, h, omega)
    y_values.append(y)
    v_values.append(v)
    y_analytical = analytical_solution(t, y0, v0, omega)
    analytical_values.append(y_analytical)
    errors.append(abs(y - y_analytical))

print("--- %s seconds ---" % (time.time() - start_time))

rmse = np.sqrt(np.mean(np.array(errors)**2))
print("RMSE:", rmse)

plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(t_values, y_values, label='Runge-Kutta Midpoint (RK2) Solution')
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
