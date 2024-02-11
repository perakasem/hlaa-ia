import numpy as np
import matplotlib.pyplot as plt
import time

class IVP1: # SHM
    def __init__(self, omega, y0, v0, t0, tf, rmsef) -> None:
        self.omega = omega # Angular frequency
        self.y0 = y0 # Initial position
        self.v0 = v0 # Initial velocity
        self.t0 = t0 # Initial time
        self.tf = tf # Final time
        self.rmsef = rmsef # Target L2 norm

    def function(self, y, omega):
        return -omega**2 * y
    
    def analytical_solution(self, t, y0, v0, omega):
        A = y0
        B = v0 / omega
        return A * np.cos(omega * t) + B * np.sin(omega * t)

# Configure test case
TC1 = IVP1(np.sqrt(10), 1, 0, 0, 40, 0.01)

def predictor_corrector_step(t, y, v, v_prev, h, omega):
    # Predictor (Euler method)
    v_predict = v + h * TC1.function(y, omega)
    y_predict = y + h * v

    # Corrector (Adams-Bashforth method)
    if v_prev is not None:
        v_correct = v + h * (1.5 * TC1.function(y_predict, omega) - 0.5 * TC1.function(y, omega))
    else:
        v_correct = v_predict
    y_correct = y + 0.5 * h * (v + v_correct)

    return y_correct, v_correct, v_predict

h0max = 1

l2norm = {}

target_rmsef = TC1.rmsef
tf = TC1.tf
t0 = TC1.t0
omega = TC1.omega
y0 = TC1.y0
v0 = TC1.v0

print(TC1.tf)
print(TC1.rmsef)

h = h0max
maxrele = 1
rmse = 1
flag = 0

while rmse > target_rmsef:
    if flag > 15:
        break
    N = int((tf - t0) / h)
    t_values = np.linspace(t0, tf, N)
    flag += 1
    v_prev = None
    # Initial values
    y = y0
    v = v0
    runtime = 0
    maxrele = 0
    
    errors = []
    y_values = []
    v_values = []
    analytical_values = []

    # Euler Method
    for t in t_values:
        start_time = time.time()
        y, v, v_predict = predictor_corrector_step(t, y, v, v_prev, h, omega)
        v_prev = v  # Update v_prev for the next iteration
        elapsed_time = (time.time() - start_time)

        y_values.append(y)
        v_values.append(v)
        runtime += elapsed_time
        y_analytical = TC1.analytical_solution(t, y0, v0, omega)

        analytical_values.append(y_analytical)
        error = abs(y - y_analytical)
        relerror = error / y_analytical
        errors.append(error)

        if relerror > maxrele:
            maxrele = relerror

        # print(f"STEPSIZE: {h}")
        # print(f" ELAPSED: {elapsed_time}")
        # print(f" RUNTIME: {runtime}")
        # print(f"   ERROR: {error}")
        # print(f"RELERROR: {relerror}")
        # print(f" MAXRELE: {maxrele}")
        # print(f"       y: {y}")
        # print(f"  y_anal: {y_analytical}")
        # print(f"       N: {N}")
        # print(f"       h: {h}")
        # print(f"    RMSE: {rmse}")
        # print("----------------------------")
    
    rmse = np.sqrt(np.mean(np.array(errors)**2))
    l2norm[h] = rmse

    h = h / 2 

xval, yval = zip(*sorted(l2norm.items()))

plt.figure(figsize=(12, 8))
plt.scatter(xval, yval)
plt.xlabel('h')
plt.ylabel('RMSE')
plt.title('Convergence Rate')
plt.legend()

m, c = np.polyfit(np.log(xval), np.log(yval), 1)
y_fit = np.exp(m*np.log(xval)+ c)
plt.plot(xval, y_fit)
plt.tight_layout()
plt.loglog()

print(f"y = e^({m}*log(x) + {c}")
print(f"runtime: {runtime}s")

plt.show()