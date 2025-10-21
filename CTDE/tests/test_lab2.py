import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def poisson_rect(ax, bx, ay, by, m_x, m_y, u_exact, f_rhs, Lf_rhs, g):
    h_x = (bx-ax)/(m_x+1)
    h = h_x

    # --- mallas
    x = np.linspace(ax, bx, m_x + 2)            
    y = np.linspace(ay, by, m_y + 2)           
    X, Y = np.meshgrid(x, y, indexing="xy")

    # Dirichlet en y
    bottom = u_exact(x, np.full_like(x, ay))    
    top    = u_exact(x, np.full_like(x, by))    

    # Neumann en x
    gL_fun, gR_fun = g
    g_left  = lambda yy: gL_fun(yy)
    g_right = lambda yy: gR_fun(yy)

    nx = m_x + 2            
    ny_int = m_y            
    n_unknowns = nx * ny_int

    def k_index(i, j):
        return (j - 1) * nx + i

    # 9pts coefficients
    s = 1.0 / (6.0 * h * h)
    c_center = -20.0 * s
    c_cross  =  4.0 * s
    c_diag   =  1.0 * s

    A = lil_matrix((n_unknowns, n_unknowns), dtype=float)
    F = np.zeros(n_unknowns, dtype=float)

    # interior points
    for j in range(1, m_y + 1):
        yj = y[j]
        for i in range(1, m_x + 1):
            xi = x[i]
            k = k_index(i, j)

            A[k, k] = c_center

            A[k, k_index(i - 1, j)] = c_cross
            A[k, k_index(i + 1, j)] = c_cross

            if j - 1 >= 1:
                A[k, k_index(i, j - 1)] = c_cross
            else:
                F[k] -= c_cross * bottom[i]

            if j + 1 <= m_y:
                A[k, k_index(i, j + 1)] = c_cross
            else:
                F[k] -= c_cross * top[i]

            # diags
            # (i-1, j-1)
            if j - 1 >= 1:
                A[k, k_index(i - 1, j - 1)] = c_diag
            else:
                F[k] -= c_diag * u_exact(x[i - 1], ay)

            # (i+1, j-1)
            if j - 1 >= 1:
                A[k, k_index(i + 1, j - 1)] = c_diag
            else:
                F[k] -= c_diag * u_exact(x[i + 1], ay)

            # (i-1, j+1)
            if j + 1 <= m_y:
                A[k, k_index(i - 1, j + 1)] = c_diag
            else:
                F[k] -= c_diag * u_exact(x[i - 1], by)

            # (i+1, j+1)
            if j + 1 <= m_y:
                A[k, k_index(i + 1, j + 1)] = c_diag
            else:
                F[k] -= c_diag * u_exact(x[i + 1], by)

            # RHS corrected
            F[k] += f_rhs(xi, yj) + (h * h / 12.0) * Lf_rhs(xi, yj)

    # rows
    inv12h = 1.0 / (12.0 * h)
    for j in range(1, m_y + 1):
        # left (i=0)
        kL = k_index(0, j)
        A[kL, k_index(0, j)] += -25.0 * inv12h
        A[kL, k_index(1, j)] +=  48.0 * inv12h
        A[kL, k_index(2, j)] += -36.0 * inv12h
        A[kL, k_index(3, j)] +=  16.0 * inv12h
        A[kL, k_index(4, j)] +=  -3.0 * inv12h
        F[kL] += g_left(y[j])

        # right (i=m_x+1)
        kR = k_index(m_x + 1, j)
        A[kR, k_index(m_x + 1, j)] +=  25.0 * inv12h
        A[kR, k_index(m_x,     j)] += -48.0 * inv12h
        A[kR, k_index(m_x - 1, j)] +=  36.0 * inv12h
        A[kR, k_index(m_x - 2, j)] += -16.0 * inv12h
        A[kR, k_index(m_x - 3, j)] +=   3.0 * inv12h
        F[kR] += g_right(y[j])

    # Step 3: Solve the system
    U_strip = spsolve(A.tocsr(), F).reshape((m_y, m_x + 2))  # j=1..m_y, i=0..m_x+1

    # Step 4;
    U = np.zeros((m_y + 2, m_x + 2), dtype=float)
    U[0, :] = bottom
    U[-1, :] = top
    U[1:-1, :] = U_strip

    return X, Y, U