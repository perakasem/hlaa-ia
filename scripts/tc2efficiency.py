import numpy as np
import matplotlib.pyplot as plt
import tc2methods

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Damped SHM
class IVP2:
    def __init__(self, b, k, y0, v0, t0, tf, rmsef) -> None:
        self.b = b # Damping coefficient
        self.k = k # Stiffness coefficient
        self.y0 = y0 # Initial position
        self.v0 = v0 # Initial velocity
        self.t0 = t0 # Initial time
        self.tf = tf # Final time
        self.rmsef = rmsef # Target L2 norm

    def function(self, y, v, b, k):
        return -b * v - k * y
    
    def analytical_solution(self, t, y0, v0, b, k):
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

IVP2TC1 = IVP2(10, 100, 1, 0, 0, 5, 0.001)
tc = IVP2TC1

# Configure starting step size
H0MAX = 1

l2norm = {}
euler_data = {}
rk2_data = {}
pc_data = {}

target_rmsef = tc.rmsef
tf = tc.tf
t0 = tc.t0
b = tc.b
k = tc.k
y0 = tc.y0
v0 = tc.v0
h0 = H0MAX
iteration = 0

for i in range(14): # Configure iterations
    h = h0
    euler_rmse = 1
    rk2_rmse = 1
    pc_rmse = 1
    euler_operations = 0
    rk2_operations = 0
    pc_operations = 0
    iteration += 1

    while euler_rmse > target_rmsef:
        N = int((tf - t0) / h)
        t_values = np.linspace(t0, tf, N)
        v_prev = None
        euler_y = y0
        euler_v = v0
        
        euler_errors = []
        euler_y_values = []
        euler_v_values = []
        analytical_values = []

        for t in t_values:
            y_analytical = tc.analytical_solution(t, y0, v0, b, k)
            analytical_values.append(y_analytical)

            euler_y, euler_v, v_predict = tc2methods.euler(tc, euler_y, euler_v, h, b, k, v_prev) 
            euler_y_values.append(euler_y)
            euler_v_values.append(euler_v)
            euler_operations += 2
            euler_error = abs(euler_y - y_analytical)
            euler_errors.append(euler_error)
            print(i, h, t)

        euler_rmse = np.sqrt(np.mean(np.array(euler_errors)**2))

        h = h / 2
    
    h = h0
    
    while rk2_rmse > target_rmsef:
        N = int((tf - t0) / h)
        t_values = np.linspace(t0, tf, N)
        v_prev = None
        rk2_y = y0
        rk2_v = v0
        pc_v = v0
        
        rk2_errors = []
        rk2_y_values = []
        rk2_v_values = []    
        analytical_values = []

        for t in t_values:
            y_analytical = tc.analytical_solution(t, y0, v0, b, k)
            analytical_values.append(y_analytical)

            rk2_y, rk2_v, v_predict = tc2methods.rk2(tc, rk2_y, rk2_v, h, b, k, v_prev) 
            rk2_y_values.append(rk2_y)
            rk2_v_values.append(rk2_v)
            rk2_operations += 8
            rk2_error = abs(rk2_y - y_analytical)
            rk2_errors.append(rk2_error)
            print(i, h, t)

        rk2_rmse = np.sqrt(np.mean(np.array(rk2_errors)**2))

        h = h / 2
    
    h = h0

    while pc_rmse > target_rmsef:
        N = int((tf - t0) / h)
        t_values = np.linspace(t0, tf, N)
        v_prev = None
        pc_y = y0
        pc_v = v0
        
        pc_errors = []
        pc_y_values = []
        pc_v_values = []
        analytical_values = []

        for t in t_values:
            y_analytical = tc.analytical_solution(t, y0, v0, b, k)
            analytical_values.append(y_analytical)

            pc_y, pc_v, v_predict = tc2methods.pc(tc, pc_y, pc_v, h, b, k, v_prev) 
            v_prev = pc_v
            pc_y_values.append(pc_y)
            pc_v_values.append(pc_v)
            pc_operations += 4
            pc_error = abs(pc_y - y_analytical)
            pc_errors.append(pc_error)
            print(i, h, t)

        pc_rmse = np.sqrt(np.mean(np.array(pc_errors)**2))

        h = h / 2

    euler_data[(iteration - 1)] = [euler_rmse, euler_operations]
    rk2_data[(iteration - 1)] = [rk2_rmse, rk2_operations]
    pc_data[(iteration - 1)] = [pc_rmse, pc_operations]
    h0 = h0 / 2

print("done.")

def plotops(method_data):
    # Assuming 'data' is already defined
    xval, yval = zip(*sorted(method_data.items()))
    rmseplot = []
    opsplot = []
    ratioplot = []

    for item in yval:
        rmseplot.append(item[0])  # Root Mean Square Error values
        opsplot.append(item[1])  # Runtime values
        ratioplot.append(item[0] * item[1])

    X = np.arange(len(method_data))

    # Standardize font sizes
    plt.rcParams.update({'font.size': 12})

    # Define consistent limits and buffers for the plots
    min_operations, max_operations = 0, max(opsplot) * 1.1
    min_rmse, max_rmse = 0, max(rmseplot) * 1.1
    min_ratio, max_ratio = 0, max(ratioplot) * 1.1

    # Define the number of x ticks and create a consistent set of labels
    x_ticks = np.arange(len(method_data))

    # Set a consistent color scheme
    color_operations = 'blue'
    color_ratio = 'red'


    # Create the plot
    fig, ax1 = plt.subplots(figsize=(8, 4))

    # Operations plot
    ax1.bar(x_ticks, opsplot, color=color_operations, label='Operations', width=0.4)
    ax1.set_ylabel('Operations', color=color_operations)
    ax1.tick_params(axis='y', labelcolor=color_operations)
    ax1.set_ylim([min_operations, max_operations])
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(X, rotation=90)
    ax1.set_xlabel('$h_0$ order')

    # Ratio plot
    ax2 = ax1.twinx()
    ax2.plot(x_ticks, rmseplot, color=color_ratio, label='RMSE', linestyle='-', marker='o')
    ax2.set_ylabel('RMSE', color=color_ratio)
    ax2.tick_params(axis='y', labelcolor=color_ratio)
    ax2.set_ylim([min_rmse, max_rmse])

    plt.xlabel('$h_0$ Order')

    # Ensure the display is not too tight
    plt.tight_layout()

    # Save the figure
    plt.savefig('eff_plot.png', bbox_inches='tight')


    # Create the plot
    fig, ax1 = plt.subplots(figsize=(8, 4))

    # Operations plot
    ax1.bar(x_ticks, opsplot, color=color_operations, label='Operations', width=0.4)
    ax1.set_ylabel('Operations', color=color_operations)
    ax1.tick_params(axis='y', labelcolor=color_operations)
    ax1.set_ylim([min_operations, max_operations])
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(X, rotation=90)
    ax1.set_xlabel('$h_0$ order')

    # Ratio plot
    ax2 = ax1.twinx()
    ax2.plot(x_ticks, ratioplot, color=color_ratio, label='Accuracy', linestyle='-', marker='o')
    ax2.set_ylabel('Accuracy', color=color_ratio)
    ax2.tick_params(axis='y', labelcolor=color_ratio)
    ax2.set_ylim([min_ratio, max_ratio])

    plt.xlabel('$h_0$ Order')

    # Ensure the display is not too tight
    plt.tight_layout()

    # Save the figure
    plt.savefig('eff_plot.png', bbox_inches='tight')

def combined(d1, d2, d3):
    x1val, y1val = zip(*sorted(d1.items()))
    x2val, y2val = zip(*sorted(d2.items()))
    x3val, y3val = zip(*sorted(d3.items()))

    opsplot1 = []
    opsplot2 = []
    opsplot3 = []

    accplot1 = []
    accplot2 = []
    accplot3 = []

    for item in y1val:
        opsplot1.append(item[1])  # Runtime values
        accplot1.append(item[0] * item[1])
    for item in y2val:
        opsplot2.append(item[1])  # Runtime values
        accplot2.append(item[0] * item[1])
    for item in y3val:
        opsplot3.append(item[1])  # Runtime values
        accplot3.append(item[0] * item[1])

    # Standardize font sizes
    plt.rcParams.update({'font.size': 12})

    # Define consistent limits and buffers for the plots
    min_operations, max_operations = 0, max(opsplot1 + opsplot2 + opsplot3) * 1.1
    min_acc, max_acc = 0, max(accplot1 + accplot2 + accplot3) * 1.1

    # Define the number of x ticks and create a consistent set of labels
    x_ticks = np.arange(len(d1))

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(x_ticks, opsplot1, color='k', label='Euler', linestyle='-')
    ax.plot(x_ticks, opsplot2, color='red', label='RK2', linestyle='-')
    ax.plot(x_ticks, opsplot3, color='blue', label='P-C', linestyle='-')
    ax.set_ylabel('Operations')
    ax.tick_params(axis='y')
    ax.set_ylim([min_operations, max_operations])

    # Add combined legend
    fig.legend()

    plt.xlabel('$h_0$ Order')

    # Ensure the display is not too tight
    plt.tight_layout()

    # Save the figure
    plt.savefig('eff_plot.png')

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(x_ticks, accplot1, color='k', label='Euler', linestyle='-')
    ax.plot(x_ticks, accplot2, color='red', label='RK2', linestyle='-')
    ax.plot(x_ticks, accplot3, color='blue', label='P-C', linestyle='-')
    ax.set_ylabel('Accuracy')
    ax.tick_params(axis='y')
    ax.set_ylim([min_acc, max_acc])

    # Add combined legend
    fig.legend()

    plt.xlabel('$h_0$ Order')

    # Ensure the display is not too tight
    plt.tight_layout()

    # Save the figure
    plt.savefig('e_plot.png')

plotops(euler_data)
plotops(rk2_data)
plotops(pc_data)

combined(euler_data, rk2_data, pc_data)

# Show the plot
plt.show()