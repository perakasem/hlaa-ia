import numpy as np
import matplotlib.pyplot as plt

def f(y, omega):
    return -omega**2 * y

def rk2(y, v, h, omega, v_prev):
    k1_v = f(y, omega)
    k1_y = v
    midpoint_v = v + 0.5 * h * k1_v
    midpoint_y = y + 0.5 * h * k1_y
    k2_v = f(midpoint_y, omega)
    k2_y = midpoint_v
    y_next = y + h * k2_y
    v_next = v + h * k2_v
    return y_next, v_next

def euler(y, v, h, omega, v_prev):
    v_next = v + h * f(y, omega)
    y_next = y + h * v
    return y_next, v_next

def pc(y, v, h, omega, v_prev):
    # Predictor (Euler method)
    v_predict = v + h * f(y, omega)
    y_predict = y + h * v

    # Corrector (Adams-Bashforth method)
    if v_prev is not None:
        v_correct = v + h * (1.5 * f(y_predict, omega) - 0.5 * f(y, omega))
    else:
        v_correct = v_predict
    y_correct = y + 0.5 * h * (v + v_correct)

    return y_correct, v_correct, v_predict

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
analytical_values = []

euler_y_values = []
euler_v_values = []
euler_errors = []

rk2_y_values = []
rk2_v_values = []
rk2_errors = []

pc_y_values = []
pc_v_values = []
pc_errors = []

# Initial values
euler_y = y0
euler_v = v0

rk2_y = y0
rk2_v = v0

pc_y = y0
pc_v = v0

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.figure(figsize=(8, 4))
ax = plt.gca()

v_prev = None
# Euler Method
for t in t_values:
    y_analytical = analytical_solution(t, y0, v0, omega)
    analytical_values.append(y_analytical)

    euler_y, euler_v = euler(euler_y, euler_v, h, omega, v_prev)
    euler_y_values.append(euler_y)
    euler_v_values.append(euler_v)
    abserr = abs(euler_y - y_analytical)
    euler_errors.append(abserr)

    rk2_y, rk2_v = rk2(rk2_y, rk2_v, h, omega, v_prev)
    rk2_y_values.append(rk2_y)
    rk2_v_values.append(rk2_v)
    abserr = abs(rk2_y - y_analytical)
    rk2_errors.append(abserr)

    pc_y, pc_v, v_predict = pc(pc_y, pc_v, h, omega, v_prev)
    pc_y_values.append(pc_y)
    pc_v_values.append(pc_v)

    abserr = abs(pc_y - y_analytical)
    pc_errors.append(abserr)

    v_prev = pc_v

plt.plot(t_values, euler_errors, label='Euler', color="red")
plt.plot(t_values, pc_errors, label='P-C', color="blue")
plt.plot(t_values, rk2_errors, label='RK2', color="purple")

# Finalize the plot
plt.xlabel('time / s')
plt.ylabel('$\Delta x$')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()

plt.show()

# Calculate RMSE
euler_rmse = np.sqrt(np.mean(np.array(euler_errors)**2))
rk2_rmse = np.sqrt(np.mean(np.array(rk2_errors)**2))
pc_rmse = np.sqrt(np.mean(np.array(pc_errors)**2))
print("EULER RMSE:", euler_rmse)
print("  RK2 RMSE:", rk2_rmse)
print("   PC RMSE:", pc_rmse)