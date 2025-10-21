import numpy as np
import matplotlib.pyplot as plt

r = 1/4
s_points = np.linspace(0, 1/8, 1000)

x_real = []   # raíces reales
s_real = []   # valores de s correspondientes

for s in s_points:
    coeffs = [r, -(1 + s), r, -s]     # ecuación cúbica: r*x^3 - (1+s)*x^2 + r*x - s = 0
    roots = np.roots(coeffs)
    # Filtramos solo las raíces estrictamente reales
    roots_real = roots[np.isreal(roots)].real
    for x in roots_real:
        x_real.append(x)
        s_real.append(s)

plt.plot(s_real, x_real, '.', markersize=1)
plt.xlabel(f'$s$')
plt.ylabel(f'$x^*$')
plt.title(f'Bifurcation plot for $r=1/4$')
plt.grid()
plt.show()



