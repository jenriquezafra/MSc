import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import spsolve

def plot_convergence(m_list, a, b, u_a, u_b, f, u_true, solve_func):
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


##################################### de Adrian ############################1

def solve_poisson_2d_9point(a, b, c, d,
                            m_x, m_y,
                            g_a, g_b, g_c, g_d,
                            f):
    """
    Resuelve el problema de Poisson 2D con condiciones de Dirichlet
    usando el esquema de 9 puntos (orden 4).

        u_xx + u_yy = f(x,y)

    Parámetros
    ----------
    a, b, c, d : float
        Intervalo en x e y: dominio [a,b]x[c,d].
    m_x, m_y : int
        Número de puntos interiores en x e y.
    g_a, g_b, g_c, g_d : functions
        Condiciones de contorno:
            g_a(y) = u(a,y), g_b(y) = u(b,y),
            g_c(x) = u(x,c), g_d(x) = u(x,d).
    f : function
        Función f(x,y).

    Returns
    -------
    X, Y : 2D ndarrays
        Malla de puntos.
    U : 2D ndarray
        Solución aproximada en la malla.
    A, F : ndarray
        Matriz del sistema y vector RHS.
    """

    # Discretización (general en x e y)
    h_x = (b - a) / (m_x + 1)
    h_y = (d - c) / (m_y + 1)

    x = np.linspace(a, b, m_x + 2)
    y = np.linspace(c, d, m_y + 2)

    # Tamaño del sistema
    N = m_x * m_y
    A = np.zeros((N, N))
    F = np.zeros(N)

    # Ensamblado
    for j in range(1, m_y + 1):
        for i in range(1, m_x + 1):
            k = (j - 1) * m_x + (i - 1)

            xi = x[i]
            yj = y[j]

            # ---- Lado derecho con corrección de truncamiento ----
            f0 = f(xi, yj)

            # vecinos en x e y
            f_xm = f(x[i-1], yj) if i > 0 else f0
            f_xp = f(x[i+1], yj) if i < m_x+1 else f0
            f_ym = f(xi, y[j-1]) if j > 0 else f0
            f_yp = f(xi, y[j+1]) if j < m_y+1 else f0

            # diagonales
            f_xmym = f(x[i-1], y[j-1]) if i > 0 and j > 0 else f0
            f_xpym = f(x[i+1], y[j-1]) if i < m_x+1 and j > 0 else f0
            f_xmyp = f(x[i-1], y[j+1]) if i > 0 and j < m_y+1 else f0
            f_xpyp = f(x[i+1], y[j+1]) if i < m_x+1 and j < m_y+1 else f0

            # Laplaciano de 9 puntos para f (con pasos hx, hy)
            #lap_f = (-20*f0 +
             #         4*(f_xm + f_xp + f_ym + f_yp) +
              #        (f_xmym + f_xpym + f_xmyp + f_xpyp)) / (6*h_x*h_y)
            lap_f = (f_xm+f_xp-2*f0)/h_x**2+(f_ym+f_yp-2*f0)/h_y**2
              
            F[k] = f0 + (h_x**2 + h_y**2)/24 * lap_f

            # ---- Matriz A ----
            # Coeficiente central
            A[k, k] = -20.0 / (6*h_x*h_y)

            # Vecinos directos
            if i > 1:
                A[k, k-1] = 4.0 / (6*h_x*h_y)
            else:
                F[k] -= 4.0/(6*h_x*h_y) * g_a(yj)

            if i < m_x:
                A[k, k+1] = 4.0 / (6*h_x*h_y)
            else:
                F[k] -= 4.0/(6*h_x*h_y) * g_b(yj)

            if j > 1:
                A[k, k-m_x] = 4.0 / (6*h_x*h_y)
            else:
                F[k] -= 4.0/(6*h_x*h_y) * g_c(xi)

            if j < m_y:
                A[k, k+m_x] = 4.0 / (6*h_x*h_y)
            else:
                F[k] -= 4.0/(6*h_x*h_y) * g_d(xi)

            # Vecinos diagonales
            coeff_diag = 1.0 / (6*h_x*h_y)

            # abajo-izquierda
            if i > 1 and j > 1:
                A[k, k-m_x-1] = coeff_diag
            else:
                bc = g_a(y[j-1]) if i==1 else g_c(x[i-1])
                F[k] -= coeff_diag * bc

            # abajo-derecha
            if i < m_x and j > 1:
                A[k, k-m_x+1] = coeff_diag
            else:
                bc = g_b(y[j-1]) if i==m_x else g_c(x[i+1])
                F[k] -= coeff_diag * bc

            # arriba-izquierda
            if i > 1 and j < m_y:
                A[k, k+m_x-1] = coeff_diag
            else:
                bc = g_a(y[j+1]) if i==1 else g_d(x[i-1])
                F[k] -= coeff_diag * bc

            # arriba-derecha
            if i < m_x and j < m_y:
                A[k, k+m_x+1] = coeff_diag
            else:
                bc = g_b(y[j+1]) if i==m_x else g_d(x[i+1])
                F[k] -= coeff_diag * bc

    # Resolver sistema
    U_vec = np.linalg.solve(A, F)

    # Reconstruir solución
    U = np.zeros((m_y+2, m_x+2))
    U[0, :] = [g_c(xi) for xi in x]
    U[-1,:] = [g_d(xi) for xi in x]
    U[:, 0] = [g_a(yj) for yj in y]
    U[:, -1] = [g_b(yj) for yj in y]

    for j in range(1, m_y+1):
        for i in range(1, m_x+1):
            k = (j-1)*m_x + (i-1)
            U[j,i] = U_vec[k]

    X, Y = np.meshgrid(x, y)
    return X, Y, U, A, F
