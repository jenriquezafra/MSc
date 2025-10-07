import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


def solve_dirichlet_laplacian_9pt(a, b, c, d,
                                  m_x, m_y,
                                  g_left, g_right,
                                  g_bottom, g_top,
                                  rhs=None):
    """Resolver Poisson 2D con Dirichlet empleando el esquema de 9 puntos."""
    if m_x <= 0 or m_y <= 0:
        raise ValueError("m_x y m_y deben ser enteros positivos.")

    h_x = (b - a) / (m_x + 1)
    h_y = (d - c) / (m_y + 1)
    if not np.isclose(h_x, h_y):
        raise ValueError("El esquema de 9 puntos requiere malla uniforme: h_x = h_y.")
    h = h_x

    x = np.linspace(a, b, m_x + 2)
    y = np.linspace(c, d, m_y + 2)
    X, Y = np.meshgrid(x, y, indexing="xy")

    x_int = x[1:-1]
    y_int = y[1:-1]
    XI, YI = np.meshgrid(x_int, y_int, indexing="xy")

    if rhs is None:
        F_mat = np.zeros((m_y, m_x), dtype=float)
        rhs_callable = None
    elif callable(rhs):
        F_mat = np.empty((m_y, m_x), dtype=float)
        for jj in range(m_y):
            for ii in range(m_x):
                F_mat[jj, ii] = rhs(XI[jj, ii], YI[jj, ii])
        rhs_callable = rhs
    else:
        rhs_arr = np.asarray(rhs, dtype=float)
        if rhs_arr.shape != (m_y, m_x):
            raise ValueError("El término derecho debe tener la forma (m_y, m_x).")
        F_mat = rhs_arr
        rhs_callable = None

    def boundary_values(data, points):
        if callable(data):
            return np.asarray([data(p) for p in points], dtype=float)
        arr = np.asarray(data, dtype=float)
        if arr.ndim == 0:
            return np.full(points.shape, arr, dtype=float)
        if arr.shape != points.shape:
            raise ValueError("Los datos de contorno no coinciden con la malla.")
        return arr

    left_boundary = boundary_values(g_left, y)
    right_boundary = boundary_values(g_right, y)
    bottom_boundary = boundary_values(g_bottom, x)
    top_boundary = boundary_values(g_top, x)

    scale = 1.0 / (6.0 * h * h)
    main_coeff = -20.0 * scale
    cross_coeff = 4.0 * scale
    diag_coeff = 1.0 * scale

    n_unknowns = m_x * m_y
    A = lil_matrix((n_unknowns, n_unknowns), dtype=float)
    F_vec = np.zeros(n_unknowns, dtype=float)

    for j in range(1, m_y + 1):
        for i in range(1, m_x + 1):
            k = (j - 1) * m_x + (i - 1)

            rhs_val = F_mat[j - 1, i - 1]
            if rhs_callable is not None:
                xi = x[i]
                yj = y[j]
                f0 = rhs_callable(xi, yj)
                f_xm = rhs_callable(x[i - 1], yj)
                f_xp = rhs_callable(x[i + 1], yj)
                f_ym = rhs_callable(xi, y[j - 1])
                f_yp = rhs_callable(xi, y[j + 1])
                lap_f = (f_xm + f_xp - 2.0 * f0) / (h * h)
                lap_f += (f_ym + f_yp - 2.0 * f0) / (h * h)
                rhs_val = f0 + (h * h) / 12.0 * lap_f

            A[k, k] = main_coeff

            if i > 1:
                A[k, k - 1] = cross_coeff
            else:
                rhs_val -= cross_coeff * left_boundary[j]

            if i < m_x:
                A[k, k + 1] = cross_coeff
            else:
                rhs_val -= cross_coeff * right_boundary[j]

            if j > 1:
                A[k, k - m_x] = cross_coeff
            else:
                rhs_val -= cross_coeff * bottom_boundary[i]

            if j < m_y:
                A[k, k + m_x] = cross_coeff
            else:
                rhs_val -= cross_coeff * top_boundary[i]

            if i > 1 and j > 1:
                A[k, k - m_x - 1] = diag_coeff
            else:
                if i == 1 and j == 1:
                    bc_val = 0.5 * (left_boundary[0] + bottom_boundary[0])
                elif i == 1:
                    bc_val = left_boundary[j - 1]
                else:
                    bc_val = bottom_boundary[i - 1]
                rhs_val -= diag_coeff * bc_val

            if i < m_x and j > 1:
                A[k, k - m_x + 1] = diag_coeff
            else:
                if i == m_x and j == 1:
                    bc_val = 0.5 * (right_boundary[0] + bottom_boundary[-1])
                elif i == m_x:
                    bc_val = right_boundary[j - 1]
                else:
                    bc_val = bottom_boundary[i + 1]
                rhs_val -= diag_coeff * bc_val

            if i > 1 and j < m_y:
                A[k, k + m_x - 1] = diag_coeff
            else:
                if i == 1 and j == m_y:
                    bc_val = 0.5 * (left_boundary[-1] + top_boundary[0])
                elif i == 1:
                    bc_val = left_boundary[j + 1]
                else:
                    bc_val = top_boundary[i - 1]
                rhs_val -= diag_coeff * bc_val

            if i < m_x and j < m_y:
                A[k, k + m_x + 1] = diag_coeff
            else:
                if i == m_x and j == m_y:
                    bc_val = 0.5 * (right_boundary[-1] + top_boundary[-1])
                elif i == m_x:
                    bc_val = right_boundary[j + 1]
                else:
                    bc_val = top_boundary[i + 1]
                rhs_val -= diag_coeff * bc_val

            F_vec[k] = rhs_val

    A_csr = A.tocsr()
    U_int = spsolve(A_csr, F_vec).reshape((m_y, m_x))

    U = np.zeros((m_y + 2, m_x + 2), dtype=float)
    U[1:-1, 1:-1] = U_int
    U[0, :] = bottom_boundary
    U[-1, :] = top_boundary
    U[:, 0] = left_boundary
    U[:, -1] = right_boundary

    return X, Y, U, A_csr, F_vec
