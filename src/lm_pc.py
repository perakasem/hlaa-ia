import numpy as np
import matplotlib.pyplot as plt
import time

def f(t, y, omega):
    return -omega**2 * y

def predictor_corrector_step(t, y, v, v_prev, h, omega):
    # Predictor (Euler method)
    v_predict = v + h * f(t, y, omega)
    y_predict = y + h * v

    # Corrector (Adams-Bashforth method)
    if v_prev is not None:
        v_correct = v + h * (1.5 * f(t + h, y_predict, omega) - 0.5 * f(t, y, omega))
    else:
        v_correct = v_predict
    y_correct = y + 0.5 * h * (v + v_correct)

    return y_correct, v_correct, v_predict


def analytical_solution(t, y0, v0, omega):
    A = y0
    B = v0 / omega
    return A * np.cos(omega * t) + B * np.sin(omega * t)


# Damping and stiffness parameters
b = 0.5  # Damping coefficient
k = 10  # Stiffness coefficient

# Initial conditions
y0 = 1  # Initial position
v0 = 0  # Initial velocity

# Time parameters
t0 = 0
tf = 100
h = 0.01  # Step size
N = int((tf - t0) / h)

# Arrays for storing values
t_values = np.linspace(t0, tf, N)
y_values = []
v_values = []
analytical_values = []
errors = []

# Initial conditions
y0 = 1  # Initial position
v0 = 0  # Initial velocity
omega = np.sqrt(k)  # Angular frequency

# Arrays for storing values
t_values = np.linspace(t0, tf, N)
y_values = []
v_values = []
analytical_values = []
errors = []

# Initial values for the loop
y = y0
v = v0
v_prev = None

# Predictor-Corrector Method
start_time = time.time()

# Main loop
for t in t_values:
    y, v, v_predict = predictor_corrector_step(t, y, v, v_prev, h, omega)
    y_values.append(y)
    v_values.append(v)

    v_prev = v  # Update v_prev for the next iteration

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
plt.plot(t_values, y_values, label='Predictor-Corrector (P-C) Solution')
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