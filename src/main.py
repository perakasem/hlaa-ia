import numpy as np
import matplotlib.pyplot as plt
import time

# Damped SHM
class IVP1: # SHM
    def __init__(self, omega, y0, v0, t0, tf, rmsef, maxrele) -> None:
        self.omega = omega # Angular frequency
        self.y0 = y0 # Initial position
        self.v0 = v0 # Initial velocity
        self.t0 = t0 # Initial time
        self.tf = tf # Final time
        self.rmsef = rmsef # Target L2 norm
        self.maxrele = maxrele # Target maximum relative error

    def function(self, y, omega):
        return -omega**2 * y
    
    def analytical_solution(self, t, y0, v0, omega):
        A = y0
        B = v0 / omega
        return A * np.cos(omega * t) + B * np.sin(omega * t)

# Damped SHM
class IVP2:
    def __init__(self, b, k, y0, v0, t0, tf, rmsef, maxrele) -> None:
        self.b = b # Damping coefficient
        self.k = k # Stiffness coefficient
        self.y0 = y0 # Initial position
        self.v0 = v0 # Initial velocity
        self.t0 = t0 # Initial time
        self.tf = tf # Final time
        self.rmsef = rmsef # Target L2 norm
        self.maxrele = maxrele # Target maximum relative error

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

TC1 = IVP1(np.sqrt(10), 1, 0, 0, 40, 0.3, 0.01)
TC2 = IVP2(0.5, 0.1, 1, 0, 0, 40, 0.3, 0.01)
