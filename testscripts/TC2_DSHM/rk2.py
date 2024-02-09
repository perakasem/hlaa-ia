import numpy as np
import matplotlib.pyplot as plt

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

def analytical_solution(t, y0, v0, b, k):
    discriminant = b**2 - 4*k
    if discriminant >= 0:  # Overdamped or critically damped
        r1 = (-b + np.sqrt(discriminant)) / 2
        r2 = (-b - np.sqrt(discriminant)) / 2
        A = (v0 - r2 * y0) / (r1 - r2)
        B = y0 - A
        return A * np.exp(r1 * t) + B * np.exp(r2 * t)
    else:  # Underdamped
        alpha = -b / 2
        omega = np.sqrt(-discriminant) / 2
        A = y0
        B = (v0 - alpha * y0) / omega
        return np.exp(alpha * t) * (A * np.cos(omega * t) + B * np.sin(omega * t))

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

# RK2 Midpoint Method
for t in t_values:
    y, v = rk_midpoint_step(t, y, v, h, b, k)
    y_values.append(y)
    v_values.append(v)

    y_analytical = analytical_solution(t, y0, v0, b, k)
    analytical_values.append(y_analytical)
    errors.append(abs(y - y_analytical))

# Calculate RMSE
rmse = np.sqrt(np.mean(np.array(errors)**2))
print("RMSE:", rmse)

# Plotting the results and error
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
