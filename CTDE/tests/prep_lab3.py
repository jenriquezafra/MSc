import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def solve_crank_nicolson_2d(mx, my, T_final, *, diffusivity=1.0):
    """
    Crank–Nicolson para la ecuación de calor 2D en [-1,1]x[-1,1] con BC Dirichlet.

    Requiere funciones globales:
      - u0(x, y): condición inicial en la malla completa (incluye bordes)
      - g_left(y, t), g_right(y, t), g_bottom(x, t), g_top(x, t): contornos Dirichlet

    Parámetros
    ----------
    mx, my : int
        Número total de nodos en x e y (incluyendo bordes). Requiere mx>=3, my>=3.
    T_final : float
        Tiempo final de integración.
    diffusivity : float, opcional
        Coeficiente de difusión kappa.

    Devuelve
    --------
    x, y : ndarray
        Vectores de malla incluyendo bordes. x.shape=(mx,), y.shape=(my,)
    t : ndarray
        Instantes de tiempo, shape (N+1,)
    U : ndarray
        Solución con shape (N+1, my, mx)
    """
    if mx < 3 or my < 3:
        raise ValueError("mx y my deben ser >= 3 para tener puntos interiores")
    if T_final <= 0.0:
        raise ValueError("T_final debe ser positivo")
    if diffusivity <= 0.0:
        raise ValueError("diffusivity debe ser positivo")

    # 1) Mallado espacial fijo en [-1,1]^2
    ax = ay = -1.0
    bx = by =  1.0
    x = np.linspace(ax, bx, mx, dtype=float)
    y = np.linspace(ay, by, my, dtype=float)
    hx = x[1] - x[0]
    hy = y[1] - y[0]

    # 2) Paso temporal: proporcional a la escala espacial más fina
    delta_t = min(hx, hy)
    N = int(np.ceil(T_final / delta_t))
    delta_t = T_final / max(N, 1)
    t = np.linspace(0.0, T_final, N + 1)

    # 3) Array solución: capas de tiempo completas (incluye bordes)
    U = np.zeros((N + 1, my, mx), dtype=float)

    # Condición inicial (incluye interior y bordes)
    # Se asume u0 devuelve shape (my, mx) al evaluar sobre malla 2D
    X, Y = np.meshgrid(x, y, indexing="xy")
    U[0] = u0(X, Y)

    # --- Matrices en puntos interiores ---
    nx_int = mx - 2
    ny_int = my - 2
    M = nx_int * ny_int  # número de incógnitas interiores

    # --- Construcción del Laplaciano con sparse ---
    ex = np.ones(nx_int)
    ey = np.ones(ny_int)
    Lx = diags([ex, -2*ex, ex], [-1,0,1], shape=(nx_int,nx_int)) / hx**2
    Ly = diags([ey, -2*ey, ey], [-1,0,1], shape=(ny_int,ny_int)) / hy**2

    Ix = eye(nx_int, format='csc')
    Iy = eye(ny_int, format='csc')

    L = kron(Iy, Lx, format='csc') + kron(Ly, Ix, format='csc')

    factor = 0.5 * diffusivity * delta_t
    A = eye(M, format='csc') - factor * L
    B = eye(M, format='csc') + factor * L

    # Prefactorización LU
    A_lu = splu(A)

    def solve_linear(rhs):
        return A_lu.solve(rhs)

    # Helpers de contorno
    def boundary_arrays(time):
        gB = np.broadcast_to(np.asarray(g_bottom(x, time), dtype=float), (mx,))
        gT = np.broadcast_to(np.asarray(g_top(x, time), dtype=float), (mx,))
        gL = np.broadcast_to(np.asarray(g_left(y, time), dtype=float), (my,))
        gR = np.broadcast_to(np.asarray(g_right(y, time), dtype=float), (my,))
        return gL, gR, gB, gT

    def boundary_contrib(gL, gR, gB, gT):
        # contribución en vector interior de tamaño (ny_int, nx_int) -> ravel C
        contrib = np.zeros((ny_int, nx_int), dtype=float)
        # columnas interiores adyacentes a izquierda/derecha
        contrib[:, 0]     += gL[1:-1] / hx**2
        contrib[:, -1]    += gR[1:-1] / hx**2
        # filas interiores adyacentes a abajo/arriba
        contrib[0, :]     += gB[1:-1] / hy**2
        contrib[-1, :]    += gT[1:-1] / hy**2
        return contrib.ravel(order="C")

    # Vector interior en t0
    U_int = U[0, 1:-1, 1:-1].ravel(order="C")

    # --- Bucle temporal ---
    for n in range(N):
        tn, tnp1 = t[n], t[n + 1]

        gL_n, gR_n, gB_n, gT_n       = boundary_arrays(tn)
        gL_np1, gR_np1, gB_np1, gT_np1 = boundary_arrays(tnp1)

        # Impone contorno en la capa n para coherencia
        U[n, :, 0]  = gL_n
        U[n, :, -1] = gR_n
        U[n, 0, :]  = gB_n
        U[n, -1, :] = gT_n

        # RHS de CN en interior
        bc = factor * (boundary_contrib(gL_n, gR_n, gB_n, gT_n)
                       + boundary_contrib(gL_np1, gR_np1, gB_np1, gT_np1))
        b = B @ U_int + bc

        # Resolver sistema lineal (sin inversa explícita)
        U_int = solve_linear(b)

        # Escribir capa n+1
        U[n + 1, 1:-1, 1:-1] = U_int.reshape((ny_int, nx_int), order="C")
        U[n + 1, :, 0]  = gL_np1
        U[n + 1, :, -1] = gR_np1
        U[n + 1, 0, :]  = gB_np1
        U[n + 1, -1, :] = gT_np1

    return x, y, t, U
