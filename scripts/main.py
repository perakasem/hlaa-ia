import numpy as np
import matplotlib.pyplot as plt
import methods
import time

# Damped SHM
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
IVP1TC1 = IVP1(np.sqrt(10), 1, 0, 0, 10, 0.005)
tc = IVP1TC1

# Configure starting step size
h0max = 1

l2norm = {}
data = {}

target_rmsef = tc.rmsef
tf = tc.tf
t0 = tc.t0
omega = tc.omega
y0 = tc.y0
v0 = tc.v0

print(tc.tf)
print(tc.rmsef)

h = h0max
rmse = 1

# Convergence Step
while rmse > target_rmsef:
    N = int((tf - t0) / h)
    t_values = np.linspace(t0, tf, N)
    v_prev = None
    # Initial values
    y = y0
    v = v0
    
    errors = []
    y_values = []
    v_values = []
    analytical_values = []

    # Euler Method
    for t in t_values:
        y, v, v_predict = methods.pc(tc, y, v, h, omega, v_prev) # Configure Method
        v_prev = v

        y_values.append(y)
        v_values.append(v)
        y_analytical = tc.analytical_solution(t, y0, v0, omega)

        analytical_values.append(y_analytical)
        error = abs(y - y_analytical)
        errors.append(error)
    
    rmse = np.sqrt(np.mean(np.array(errors)**2))
    l2norm[h] = rmse
    print(h)
    print(rmse)
    print("--------")
    h = h / 2

# Plot Convergence Rate Graph
xval, yval = zip(*sorted(l2norm.items()))

plt.figure(figsize=(12, 8))
plt.scatter(xval, yval)
plt.xlabel('h')
plt.ylabel('RMSE')
plt.title('Convergence Rate')
plt.gca().invert_xaxis()
plt.legend()

m, c = np.polyfit(np.log(xval), np.log(yval), 1)
y_fit = np.exp(m*np.log(xval)+ c)
plt.plot(xval, y_fit)
plt.tight_layout()
plt.loglog()

print(f"y = e^({m}*log(x) + {c}")

plt.savefig('convergence.png', bbox_inches='tight')

h0 = h0max
iteration = 0

for i in range(17): # Configure iterations
    h = h0
    rmse = 1
    operations = 0
    iteration += 1

    while rmse > target_rmsef:
        N = int((tf - t0) / h)
        t_values = np.linspace(t0, tf, N)
        v_prev = None
        y = y0
        v = v0
        runtime = 0
        
        errors = []
        y_values = []
        v_values = []
        analytical_values = []

        for t in t_values:
            start_time = time.time()
            y, v, v_predict = methods.pc(tc, y, v, h, omega, v_prev) # Configure Method
            v_prev = v
            elapsed_time = (time.time() - start_time)

            y_values.append(y)
            v_values.append(v)
            runtime += elapsed_time
            operations += 4 # CONFIGURE STEPS: RK2=8, PC=4, EULER=2
            y_analytical = tc.analytical_solution(t, y0, v0, omega)

            analytical_values.append(y_analytical)
            error = abs(y - y_analytical)
            errors.append(error)
        
        rmse = np.sqrt(np.mean(np.array(errors)**2))
        h = h / 2
    print(f" Iteration: {iteration}")
    print(f"        h0: {h0}")
    print(f"Final RMSE: {rmse}")
    print(f"Operations: {operations}")
    print(f"-------------------------------")
    data[(iteration - 1)] = [rmse, operations]
    h0 = h0 / 2

# Assuming 'data' is already defined
xval, yval = zip(*sorted(data.items()))
rmseplot = []
rtplot = []
ratioplot = []

for item in yval:
    rmseplot.append(item[0])  # Root Mean Square Error values
    rtplot.append(item[1])  # Runtime values
    ratioplot.append(item[0] / item[1])

# For the runtime plot
min_rt = min(rtplot)
max_rt = max(rtplot)
buffer_rt = (max_rt - min_rt) * 0.05

# For the RMSE plot
min_rmse = min(rmseplot)
max_rmse = max(rmseplot)
buffer_rmse = (max_rmse - min_rmse) * 0.05

# For the ratio plot
min_ratio = min(ratioplot)
max_ratio = max(ratioplot)
buffer_ratio = (max_ratio - min_ratio) * 0.05

fig, ax1 = plt.subplots(figsize=(10, 8), dpi=100)

X = np.arange(len(data))
ax1.bar(X, rtplot, color='b', label='Operations', width=0.4)

ax1.set_ylabel('Operations', color='b')
ax1.set_xlabel('h = h0 / 2^x')
ax1.set_title('Operations vs h0')
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_ylim([min_rt - buffer_rt, max_rt + buffer_rt])
ax1.set_xticks(X)
ax1.set_xticklabels(xval, rotation=90)
ax1.grid(True)

# Create a secondary y-axis for the RMSE values
ax2 = ax1.twinx()
ax2.plot(X, rmseplot, color='r', label='RMSE', linestyle='-', marker='o')
ax2.set_ylabel('RMSE', color='r')
ax2.tick_params(axis='y', labelcolor='r')
ax2.set_ylim([min_rmse - buffer_rmse, max_rmse + buffer_rmse])

# Optional: Adding a legend to the plot
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Ensure the display is not too tight
plt.tight_layout()

# Save the figure
plt.savefig('operations_plot.png', bbox_inches='tight')


fig, ax1 = plt.subplots(figsize=(10, 8), dpi=100)

X = np.arange(len(data))
ax1.bar(X, rtplot, color='b', label='Operations', width=0.4)

ax1.set_ylabel('Operations', color='b')
ax1.set_xlabel('h = h0 / 2^x')
ax1.set_title('Operations vs h0')
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_ylim([min_rt - buffer_rt, max_rt + buffer_rt])
ax1.set_xticks(X)
ax1.set_xticklabels(xval, rotation=90)
ax1.grid(True)

# Create a secondary y-axis for the Ratio values
ax2 = ax1.twinx()
ax2.plot(X, ratioplot, color='r', label='RMSE/Steps', linestyle='-', marker='o')
ax2.set_ylabel('RMSE/Steps', color='r')
ax2.tick_params(axis='y', labelcolor='r')
ax2.set_ylim([min_ratio - buffer_ratio, max_ratio + buffer_ratio])

# Optional: Adding a legend to the plot
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Ensure the display is not too tight
plt.tight_layout()

# Save the figure
plt.savefig('ratio_plot.png', bbox_inches='tight')


# Show the plot
plt.show()
