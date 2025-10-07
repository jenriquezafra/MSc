import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def solve_polar_poisson_5pt(
    R1=0.0,
    R2=1.0,
    m_r=50,            # nº de anillos interiores (sin contar bordes)
    n_theta=64,        # nº de nodos angulares (periódicos)
    f=None,            # f(r,theta); por defecto 0
    g_outer=None,      # Dirichlet en r=R2: u(R2,θ)=g_outer(θ)
    g_inner=None,      # Dirichlet en r=R1 si R1>0: u(R1,θ)=g_inner(θ)
    theta0=0.0,
    thetamax=2*np.pi,
):
    """
    Resuelve (1/r) ∂_r ( r ∂_r u ) + (1/r^2) ∂^2_{θ} u = f(r,θ)
    con malla uniforme en (r,θ), condiciones periódicas en θ y:
      - si R1==0 (disco): U0 = mean(U_{1j}) - f(0)*(Δr/2)^2
      - si R1>0 (corona): Dirichlet en r=R1 con g_inner(θ)

    Devuelve:
        RHO, THETA, U  con shape ((m_r+2), n_theta)
        (incluye los bordes r=R1 y r=R2).
    """

    m = int(m_r)
    n = int(n_theta)

    dr = (R2 - R1) / (m + 1)
    dth = (thetamax - theta0) / n  # periódico: n nodos, sin repetir 2π

    r_nodes = R1 + np.arange(0, m + 2) * dr      # i = 0..m+1
    th_nodes = theta0 + np.arange(n) * dth       # j = 0..n-1. CHANGE

    def idx(i, j):  # i=1..m (solo interior), j=0..n-1
        return (i - 1) * n + (j % n)

    N = m * n
    A = lil_matrix((N, N), dtype=float)
    b = np.zeros(N, dtype=float)

    g_out = np.array([g_outer(th) for th in th_nodes])
    if R1 > 0:
        g_in = np.array([g_inner(th) for th in th_nodes])
    else:
        f0 = f(0.0, th_nodes[0])  # valor de f en el origen

    for i in range(1, m + 1):
        ri   = r_nodes[i]
        rimh = ri - dr/2          # r_{i-1/2}
        riph = ri + dr/2          # r_{i+1/2}
        ar_minus = rimh / (ri * dr * dr)
        ar_plus  = riph / (ri * dr * dr)
        ath = 1.0 / (ri * ri * dth * dth)
        a_center = -((rimh + riph) / (ri * dr * dr) + 2.0 * ath)

        for j in range(n):
            row = idx(i, j)
            # centro
            A[row, row] += a_center
            # términos angulares (periódicos)
            A[row, idx(i, j-1)] += ath
            A[row, idx(i, j+1)] += ath
            # radial hacia fuera
            if i < m:
                A[row, idx(i+1, j)] += ar_plus
            else:
                b[row] -= ar_plus * g_out[j]  # Dirichlet en r=R2 
            # radial hacia dentro
            if i > 1:
                A[row, idx(i-1, j)] += ar_minus
            else:
                if R1 == 0.0:
                    # U0 = (1/n) Σ_k U_{1k} - f(0)*(dr/2)^2
                    coeff = ar_minus / n
                    for k in range(n):
                        A[row, idx(1, k)] += coeff 
                    b[row] -= ar_minus * (f0 * (dr/2.0)**2) 
                else:
                    b[row] -= ar_minus * g_in[j]  # Dirichlet en r=R1
            # término fuente
            b[row] -= f(ri, th_nodes[j]) 

    U_int = spsolve(A.tocsr(), b).reshape(m, n)

    # Construir solución incluyendo bordes
    U = np.zeros((m + 2, n), dtype=float)
    U[1:m+1, :] = U_int
    U[m+1, :] = g_out
    if R1 == 0.0:
        U0 = U[1, :].mean() - f0 * (dr/2.0)**2
        U[0, :] = U0
    else:
        U[0, :] = g_in

    THETA, RHO = np.meshgrid(th_nodes, r_nodes)
    return RHO, THETA, U

# --- Ejemplo mínimo (comentado) ---
# f = lambda r,th: 0.0
# gR2 = lambda th: np.cos(3*th)
# RHO, THETA, U = solve_polar_poisson_5pt(R1=0.0, R2=1.0, m_r=64, n_theta=128,
#                                          f=f, g_outer=gR2)
# # RHO, THETA, U ya listos para contourf en coordenadas polares/cartesianas.