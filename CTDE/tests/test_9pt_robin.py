import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def poisson(a, b, m, u_exact, f_rhs, Lf_rhs, g):
    """
    Solve the 2D Poisson problem on [a,b]x[a,b] with the 9-point (4th-order) FD scheme.
    BCs: Dirichlet on y=a and y=b via u_exact; Neumann on x=a and x=b via g.
    
    Parameters
    ----------
    a, b : float
        Square domain [a,b] x [a,b].
    m : int
        Number of interior points per dimension (requires m >= 4).
    u_exact : callable
        u_exact(x, y). Used for Dirichlet boundaries y=a and y=b.
    f_rhs : callable
        f(x,y) in Δu = f.
    Lf_rhs : callable
        Laplacian(f)(x,y) for the 9-point RHS correction: f + (h^2/12) * Lap(f).
    g : tuple or callable
        Neumann data for x=a and x=b. If tuple, g=(g_left, g_right) with g_left(y), g_right(y).
        If single callable, it must return a pair (g_left(y), g_right(y)).

    Returns
    -------
    X, Y : 2D ndarrays
        Grid coordinates including boundaries.
    U : 2D ndarray
        Numerical solution on all grid points (boundaries included).
    """

    # Step 1: Discretize the domain.
    h = (b - a) / (m + 1)
    x = np.linspace(a, b, m + 2)                 # i = 0..m+1
    y = np.linspace(a, b, m + 2)                 # j = 0..m+1
    X, Y = np.meshgrid(x, y, indexing="xy")

    # Step 2: Build the sparse matrix A using:
    bottom = u_exact(x, np.full_like(x, a))      # j=0
    top    = u_exact(x, np.full_like(x, b))      # j=m+1

    gL_fun, gR_fun = g
    g_left  = lambda yy: gL_fun(yy)
    g_right = lambda yy: gR_fun(yy)


    nx = m + 2
    ny_int = m
    n_unknowns = nx * ny_int

    def k_index(i, j):
        return (j - 1) * nx + i

    s = 1.0 / (6.0 * h * h)
    c_center = -20.0 * s
    c_cross  =  4.0 * s
    c_diag   =  1.0 * s

    A = lil_matrix((n_unknowns, n_unknowns), dtype=float)
    F = np.zeros(n_unknowns, dtype=float)

    # --- Interior PDE eqs for i=1..m, j=1..m
    for j in range(1, m + 1):
        yj = y[j]
        for i in range(1, m + 1):
            xi = x[i]
            k = k_index(i, j)

            # 9-point stencil at (i,j)
            A[k, k] = c_center

            # cross neighbors
            A[k, k_index(i - 1, j)] = c_cross
            A[k, k_index(i + 1, j)] = c_cross

            # (i,j-1) bottom Dirichlet si j-1 == 0
            if j - 1 >= 1:
                A[k, k_index(i, j - 1)] = c_cross
            else:
                F[k] -= c_cross * bottom[i]

            # (i,j+1) top Dirichlet if j+1 == m+1
            if j + 1 <= m:
                A[k, k_index(i, j + 1)] = c_cross
            else:
                F[k] -= c_cross * top[i]

            # diags (i-1,j-1)
            if j - 1 >= 1:
                A[k, k_index(i - 1, j - 1)] = c_diag
            else:
                F[k] -= c_diag * u_exact(x[i - 1], a)  

            # (i+1,j-1)
            if j - 1 >= 1:
                A[k, k_index(i + 1, j - 1)] = c_diag
            else:
                F[k] -= c_diag * u_exact(x[i + 1], a)

            # (i-1,j+1)
            if j + 1 <= m:
                A[k, k_index(i - 1, j + 1)] = c_diag
            else:
                F[k] -= c_diag * u_exact(x[i - 1], b)  

            # (i+1,j+1)
            if j + 1 <= m:
                A[k, k_index(i + 1, j + 1)] = c_diag
            else:
                F[k] -= c_diag * u_exact(x[i + 1], b)

            # RHS 
            F[k] += f_rhs(xi, yj) + (h * h / 12.0) * Lf_rhs(xi, yj)

    # neumann 4ht order
    inv12h = 1.0 / (12.0 * h)
    for j in range(1, m + 1):
        # left boundary row
        kL = k_index(0, j)
        A[kL, k_index(0, j)] += -25.0 * inv12h
        A[kL, k_index(1, j)] +=  48.0 * inv12h
        A[kL, k_index(2, j)] += -36.0 * inv12h
        A[kL, k_index(3, j)] +=  16.0 * inv12h
        A[kL, k_index(4, j)] +=  -3.0 * inv12h
        F[kL] += g_left(y[j])

        # right boundary row
        kR = k_index(m + 1, j)
        A[kR, k_index(m + 1, j)] +=  25.0 * inv12h
        A[kR, k_index(m,     j)] += -48.0 * inv12h
        A[kR, k_index(m - 1, j)] +=  36.0 * inv12h
        A[kR, k_index(m - 2, j)] += -16.0 * inv12h
        A[kR, k_index(m - 3, j)] +=   3.0 * inv12h
        F[kR] += g_right(y[j])

    # Step 3: Solve the linear system A U = F.
    U_strip = spsolve(A.tocsr(), F).reshape((m, m + 2))  # j=1..m, i=0..m+1

    # Step 4: Reshape the full 2D solution including boundaries.
    U = np.zeros((m + 2, m + 2), dtype=float)
    U[0, :] = bottom
    U[-1, :] = top
    U[1:-1, :] = U_strip

    return X, Y, U