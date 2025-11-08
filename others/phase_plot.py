import numpy as np
import matplotlib.pyplot as plt


def func_exact(t, eps):
    return np.exp(-eps*t)/np.sqrt(1-eps) * np.sin(t*np.sqrt(1-eps))

def func_approx(t, eps):
    return np.exp(-eps*t) * np.sin(t)


def y_eps(t,eps):
    return np.exp(-eps*t) * ((eps-1)*np.cos(np.sqrt(1-eps)*t) - 2*eps/(np.sqrt(1-eps)) * np.sin(np.sqrt(1-eps)*t))

def x_eps(t, eps):
    return np.exp(-eps*t)/(np.sqrt(1-eps)) * np.sin(np.sqrt(1-eps)*t)

t_values = np.linspace(0, 75, 1000)



############################################################### Plotting ###############################################################

plt.figure(figsize=(12, 8))
plt.plot(t_values, func_exact(t_values, 0.1), label='Exact')
plt.plot(t_values, func_approx(t_values, 0.1), '--', label='Approximation')
plt.xlabel('Time')
plt.legend()
plt.grid()
plt.tight_layout()

plt.show()

if __name__ == "__main__":
    pass
