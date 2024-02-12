import matplotlib.pyplot as plt
import numpy as np

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.figure(figsize=(6, 4))
plt.loglog()
plt.xlabel('d$t$')
plt.ylabel('RMSE')
plt.gca().invert_xaxis()
plt.tick_params(axis='x', which='minor', bottom=False)
plt.tick_params(axis='y', which='minor', left=False)

# Ensuring x is within a range that makes sense for dt, and strictly positive
x = np.logspace(-4.5, 0, 100)  # For dt ranging from 0.0001 to 1

euler1_y_fit = np.exp(1.0662*np.log(x) + 2.8698)
euler2_y_fit = np.exp(1.0592*np.log(x) + 0.7415)
rk21_y_fit = np.exp(1.0511*np.log(x) + 0.5907)
rk22_y_fit = np.exp(1.0299*np.log(x) + 0.1352)
pc1_y_fit = np.exp(0.9457*np.log(x) + 1.9084)
pc2_y_fit = np.exp(1.2317*np.log(x) + 1.0899)

plt.plot(x, euler1_y_fit, color="k", label="Euler (Test Case 1)")
plt.plot(x, euler2_y_fit, linestyle="dashed", color="k", label="Euler Test Case 2")
plt.plot(x, rk21_y_fit, color="red", label="RK2 Test Case 1")
plt.plot(x, rk22_y_fit, linestyle="dashed", color="red", label="RK2 Test Case 2")
plt.plot(x, pc1_y_fit, color="blue", label="P-C Test Case 1")
plt.plot(x, pc2_y_fit, linestyle="dashed", color="blue", label="P-C Test Case 2")

# Draw horizontal line at RMSE = 0.001
target_rmse = 0.001
plt.axhline(y=target_rmse, color='grey', linestyle='--', linewidth=1)

# Find the intersection points and plot them
methods_y_fits = [euler1_y_fit, euler2_y_fit, rk21_y_fit, rk22_y_fit, pc1_y_fit, pc2_y_fit]
colors = ['k', 'k', 'red', 'red', 'blue', 'blue']
linestyles = ['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed']

for y_fit, color, linestyle in zip(methods_y_fits, colors, linestyles):
    intersect_dt = x[np.argmin(np.abs(y_fit - target_rmse))]
    intersect_rmse = y_fit[np.argmin(np.abs(y_fit - target_rmse))]
    plt.plot(intersect_dt, intersect_rmse, 'o', color=color)

plt.tight_layout()
plt.ylim(0.0005, 50)
plt.legend()

plt.savefig('allconvergencerates.png', bbox_inches='tight')

# Show the plot
plt.show()
