import numpy as np
import matplotlib.pyplot as plt
import scripts.tc1methods as tc1methods

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

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
IVP1TC1 = IVP1(np.sqrt(10), 1, 0, 0, 5, 0.001)
tc = IVP1TC1

# Configure starting step size
H0MAX = 1

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

h = H0MAX
rmse = 1

step = 0
plt.figure(figsize=(8, 4))

# Convergence Step
while rmse > target_rmsef:
    N = int((tf - t0) / h)
    t_values = np.linspace(t0, tf, N)
    v_prev = None
    step += 1
    # Initial values
    y = y0
    v = v0
    
    errors = []
    y_values = []
    v_values = []
    analytical_values = []

    # Euler Method
    for t in t_values:
        y, v, v_predict = tc1methods.euler(tc, y, v, h, omega, v_prev) # Configure Method
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

    # Plot the errors for the current step size
    plt.plot(t_values, errors, label=f'$dt=2^{{-{step}}}$')

# Finalize the plot
plt.xlabel('time / s')
plt.ylabel('$\Delta x$')
plt.yscale('log')
plt.legend()
plt.legend(loc=' leftupper', bbox_to_anchor=(1, 1))
plt.tight_layout()

plt.savefig('convergence.png', bbox_inches='tight')



# Plot Convergence Rate Graph
xval, yval = zip(*sorted(l2norm.items()))

plt.figure(figsize=(4, 4))
plt.loglog()
plt.scatter(xval, yval)
plt.xlabel('d$t$')
plt.ylabel('RMSE')
plt.title('Euler') # Configure Method Name
plt.gca().invert_xaxis()
plt.tick_params(axis='x', which='minor', bottom=False)
plt.tick_params(axis='y', which='minor', left=False)

h_threshold = 10**(-1)

stable_xval = [x for x in xval if x < h_threshold]
stable_yval = [y for x, y in zip(xval, yval) if x < h_threshold]

m, c = np.polyfit(np.log(stable_xval), np.log(stable_yval), 1)
y_fit = np.exp(m*np.log(xval) + c)

plt.plot(xval, y_fit, label=f"$y = e^{{{m:.4f}\\log(x) + {c:.4f}}}$")
plt.tight_layout()
plt.legend()

print(f"y = e^({m:.4f}*log(x) + {c:.4f}")

plt.savefig('convergencerate.png', bbox_inches='tight')

# Show the plot
plt.show()