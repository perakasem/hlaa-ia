import numpy as np
import matplotlib.pyplot as plt

def f(y, omega):
    return -omega**2 * y

def euler_step(y, v, h, omega):
    v_next = v + h * f(y, omega)
    y_next = y + h * v

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
tf = 5
h = 1/(2**6)
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

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.figure(figsize=(8, 4))
ax = plt.gca()

# Euler Method
for t in t_values:
    y, v = euler_step(y, v, h, omega)
    y_values.append(y)
    v_values.append(v)

    y_analytical = analytical_solution(t, y0, v0, omega)
    analytical_values.append(y_analytical)
    abserr = abs(y - y_analytical)
    errors.append(abserr)

plt.plot(t_values, errors, label='absolute error', linestyle="dashed", color="black")
plt.plot(t_values, y_values, label='numerical solution', color="red")
plt.plot(t_values, analytical_values, label='analytical solution', color="blue")

# Finalize the plot
plt.xlabel('time / s', loc='right')
plt.ylabel('$x$')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()

# Move left spine and bottom spine to the center
ax.spines['bottom'].set_position('zero')

# Hide the top and right spines
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# Adjust ticks after moving spines
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')


plt.show()

# Calculate RMSE
rmse = np.sqrt(np.mean(np.array(errors)**2))
print("RMSE:", rmse)