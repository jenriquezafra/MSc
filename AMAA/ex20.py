import numpy as np
import matplotlib.pyplot as plt
from math import comb

def BernsteinPoly(a, b, n, h):
    """
    Compute Bernstein approximants on x \in [-1, 1] for f(x) = exp(-1/x^2) for x!=0 and f(0)=0.
    
    Params:
    a: float
        left boundary
    b: float
        right boundary
    n: int
        degree of the polynomial
    h: int
        step of the computations
    """

    x = np.linspace(a, b, h)
    t = (x - a) / (b - a)  # map x in [a,b] to t in [0,1]

    Bn = 0.0
    for k in range(0, n+1):
        if (n % 2 == 0) and (k == n // 2):
            continue

        Bn += np.exp(-1.0 / ((2.0*k/n - 1.0)**2)) * comb(n, k) * (t**k) * ((1 - t)**(n - k))
    return x, Bn
    

def func(x):
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    m = (x != 0)
    y[m] = np.exp(-1.0 / (x[m]**2))
    return y

colors = plt.cm.coolwarm(np.linspace(0, 1, 10))  # o 'plasma', 'cividis', 'coolwarm', etc.

plt.figure(figsize=(10, 6))
for i, color in enumerate(colors, start=1):
    xp, Bn = BernsteinPoly(-1, 1, i, 1000)
    plt.plot(xp, Bn, lw=1 + 0.15*i, color=color, alpha=0.9 - i*0.05,
             label=fr'$B_{{{i}}}f(x)$')

plt.plot(xp, func(xp), 'k', lw=2.5, label=r"$f(x)=e^{-1/x^2}$")

plt.xlabel('x')
plt.xlim([-1, 1])
#plt.ylim([-2, 2])
plt.title('First 10 Bernstein polynomials')
plt.legend()
plt.grid(True, which='both')
plt.show()
