import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import spsolve


def _evaluate_boundary(data, points):
    """Return boundary values evaluated at the provided 1D grid."""
    if callable(data):
        return np.asarray([data(p) for p in points], dtype=float)
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 0:
        return np.full(points.shape, arr, dtype=float)
    if arr.shape != points.shape:
        raise ValueError("Boundary data array has incompatible shape.")
    return arr


def _evaluate_rhs(rhs, x_grid, y_grid):
    """Return the interior right-hand side evaluated on the mesh."""
    if rhs is None:
        return np.zeros_like(x_grid, dtype=float)
    if callable(rhs):
        return np.asarray([[rhs(x, y) for x in row] for row, y in zip(x_grid, y_grid[:, 0])], dtype=float)
    arr = np.asarray(rhs, dtype=float)
    if arr.shape != x_grid.shape:
        raise ValueError("RHS array must match the interior grid shape.")
    return arr


def solve_dirichlet_laplacian_5pt(a, b, c, d,
                                  m_x, m_y,
                                  g_left, g_right,
                                  g_bottom, g_top,
                                  rhs=None):
    """Solve a 2D Poisson problem with Dirichlet data using the 5-point Laplacian.

    Parameters
    ----------
    a, b, c, d : float
        Domain limits such that x in [a, b] and y in [c, d].
    m_x, m_y : int
        Number of interior grid points in x and y direction respectively.
    g_left, g_right : callable or array-like or float
        Boundary data for x = a and x = b as functions of y, arrays or constants.
    g_bottom, g_top : callable or array-like or float
        Boundary data for y = c and y = d as functions of x, arrays or constants.
    rhs : callable or array-like or None
        Right-hand side f(x, y). When None the Laplace equation is solved.

    Returns
    -------
    X, Y : 2D ndarrays
        Meshgrid containing all grid points including boundaries.
    U : 2D ndarray
        Numerical solution including Dirichlet boundary values.
    A : scipy.sparse.csr_matrix
        Sparse system matrix corresponding to the 5-point stencil.
    F : ndarray
        Right-hand side vector used in the linear system.
    """
    if m_x <= 0 or m_y <= 0:
        raise ValueError("m_x and m_y must be positive integers.")

    h_x = (b - a) / (m_x + 1)
    h_y = (d - c) / (m_y + 1)

    x = np.linspace(a, b, m_x + 2)
    y = np.linspace(c, d, m_y + 2)
    X, Y = np.meshgrid(x, y, indexing="xy")

    x_int = x[1:-1]
    y_int = y[1:-1]
    XI, YI = np.meshgrid(x_int, y_int, indexing="xy")

    F_mat = _evaluate_rhs(rhs, XI, YI)

    left_vals = _evaluate_boundary(g_left, y_int)
    right_vals = _evaluate_boundary(g_right, y_int)
    bottom_vals = _evaluate_boundary(g_bottom, x_int)
    top_vals = _evaluate_boundary(g_top, x_int)

    if m_x > 0:
        F_mat[:, 0] += left_vals / h_x**2
        F_mat[:, -1] += right_vals / h_x**2
    if m_y > 0:
        F_mat[0, :] += bottom_vals / h_y**2
        F_mat[-1, :] += top_vals / h_y**2

    T_x = diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(m_x, m_x), format="csr") / h_x**2
    T_y = diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(m_y, m_y), format="csr") / h_y**2
    I_x = eye(m_x, format="csr")
    I_y = eye(m_y, format="csr")
    A = kron(I_y, T_x) + kron(T_y, I_x)
    A = A.tocsr()

    F_vec = F_mat.reshape(-1)
    U_int = spsolve(A, F_vec)
    U_int = U_int.reshape((m_y, m_x))

    U = np.zeros((m_y + 2, m_x + 2), dtype=float)
    U[0, :] = _evaluate_boundary(g_bottom, x)
    U[-1, :] = _evaluate_boundary(g_top, x)
    U[:, 0] = _evaluate_boundary(g_left, y)
    U[:, -1] = _evaluate_boundary(g_right, y)
    U[1:-1, 1:-1] = U_int

    return X, Y, U, A, F_vec


def demo_laplacian_contourf(m_x=60, m_y=60):
    """Illustrate the solver on a harmonic function and plot with contourf."""
    import matplotlib.pyplot as plt

    a, b = 0.0, 1.0
    c, d = 0.0, 1.0

    def exact_solution(x, y):
        return np.sin(np.pi * x) * np.sinh(np.pi * y) / np.sinh(np.pi)

    g_left = 0.0
    g_right = 0.0
    g_bottom = 0.0
    g_top = lambda x: np.sin(np.pi * x)

    X, Y, U, _, _ = solve_dirichlet_laplacian_5pt(a, b, c, d,
                                                  m_x, m_y,
                                                  g_left, g_right,
                                                  g_bottom, g_top,
                                                  rhs=None)

    U_exact = exact_solution(X, Y)
    max_error = float(np.max(np.abs(U - U_exact)))

    fig, ax = plt.subplots(figsize=(6, 4))
    contour = ax.contourf(X, Y, U, levels=40, cmap="coolwarm")
    ax.contour(X, Y, U_exact, levels=8, colors="white", linewidths=0.7)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"5-point Laplacian solution (max error {max_error:.2e})")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(contour, ax=ax, label="u(x, y)")
    fig.tight_layout()

    return X, Y, U, fig, ax, max_error


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    demo_laplacian_contourf()
    plt.show()
