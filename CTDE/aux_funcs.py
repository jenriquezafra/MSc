import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import spsolve

def test_convergence(m_list, a, b, u_a, u_b, f, u_true, solve_func):
    """
    To test the convergence of FD methods and plot the error vs. step size h in a log-log plot.
    parameters:
        a, b: interval endpoints
        u_a, u_b: boundary values
        u_true: function for the true solution
        solve_func: function that takes (a, b, u_a, u_b, f, m) and returns (points, U_it)
        m_list: list of m values to test
    returns:
        None (plots the error vs. h)    
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    errors = []
    hs = []

    for m in m_list:
        points, U_it = solve_func(a, b, u_a, u_b, f, m)
        error = np.max(np.abs(U_it - u_true(points)))
        errors.append(error)
        hs.append((b - a) / (m + 1))

    hvec = 1.0 / 2**np.arange(30)

    plt.figure(figsize=(12, 8))
    plt.loglog(hs, errors, marker='o', label="Error")
    plt.loglog(hvec, 0.1*hvec**1, 'k:', label="Slope 1")
    plt.loglog(hvec, 0.01*hvec**2, 'g:', label="Slope 2")
    plt.loglog(hvec, 0.01*hvec**3, 'm:', label="Slope 3")
    plt.loglog(hvec, 0.001*hvec**4, 'c:', label="Slope 4")
    plt.xlim([min(hs), max(hs)])
    plt.ylim([min(errors), max(errors)])
    plt.xlabel('h')
    plt.ylabel('Max error')
    plt.title('Error vs. Step Size')
    plt.legend()
    plt.grid(True, which='both')

    return None





def poisson_general(mx, my):
    """
    Solve the 2D Poisson equation:
        ∇²u = -2 sin(x) sin(y),   (x,y) ∈ [0,2π] × [0,2π],
        u = 0 on the boundary.

    Parameters
    ----------
    m : int
        Number of interior grid points in each direction.

    Returns
    -------
    X, Y : 2D ndarrays
        Meshgrid of all grid points including boundaries.
    U : 2D ndarray
        Numerical solution at all grid points.
    """
    # Step 1: Discretize domain [0, 2π] × [0, 2π]
    h = (2*np.pi)/(m+1) 

    # Step 2: Build sparse matrix A for the 5-point Laplacian
    B = diags([np.ones(m-1), -4*np.ones(m), np.ones(m-1)],
        offsets=[-1, 0, 1], format='csr')

    T = diags([np.ones(m-1), np.zeros(m), np.ones(m-1)],
        offsets=[-1, 0, 1], format='csr')

    I = eye(m, format='csr')

    ## reconstruct A using Kronecker products
    A = 1/h**2 * (kron(I,B) + kron(T, I))

    # Step 3: Assemble RHS vector with f(x,y) = -2 sin(x) sin(y)
    x = np.linspace(0, 2*np.pi, m+2)
    y = np.linspace(0, 2*np.pi, m+2)
    X, Y = np.meshgrid(x, y)
    F = -2*np.sin(X[1:-1, 1:-1])*np.sin(Y[1:-1, 1:-1]) # only interior points
    F = F.reshape(m**2)  # flatten to 1D array



    # Step 4: Solve linear system AU = F
    U = spsolve(A, F)
    U = U.reshape((m, m))  # reshape back to 2D array


    # Step 5: Reconstruct solution including boundary values
    U_full = np.zeros((m+2, m+2))
    U_full[1:-1, 1:-1] = U  # interior points
    U = U_full

    return X, Y, U
