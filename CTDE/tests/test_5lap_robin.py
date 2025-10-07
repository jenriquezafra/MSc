import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


def solve_poisson_5pt_mixed(a, b, c, d,
                            m_x, m_y,
                            left_bc,
                            right_bc,
                            bottom_bc,
                            top_bc,
                            rhs=None):
    """
    Resuelve -Δu = rhs en (a,b)×(c,d) con el stencil de 5 puntos.
    Solo admite configuraciones donde dos caras opuestas son Dirichlet
    y las otras dos son Neumann. Cada condición se pasa como:
        ('D', data)  con data = u en la frontera,
        ('N', data)  con data = ∂u/∂n (normal saliente).
    El parámetro data puede ser escalar, función o array con la longitud adecuada.
    """

    if m_x < 2 or m_y < 2:
        raise ValueError("Se requiere m_x, m_y >= 2.")

    def _eval(data, pts):
        pts = np.asarray(pts, dtype=float)
        if callable(data):
            return np.asarray([data(float(p)) for p in pts], dtype=float)
        arr = np.asarray(data, dtype=float)
        if arr.ndim == 0:
            return np.full_like(pts, float(arr), dtype=float)
        if arr.shape != pts.shape:
            raise ValueError("Los datos de contorno no coinciden con la malla.")
        return arr.astype(float)

    x = np.linspace(a, b, m_x + 2)
    y = np.linspace(c, d, m_y + 2)
    X, Y = np.meshgrid(x, y, indexing="xy")
    x_int = x[1:-1]
    y_int = y[1:-1]

    def _rhs_eval(rhs_fun):
        if rhs_fun is None:
            return np.zeros((m_y, m_x), dtype=float)
        if np.isscalar(rhs_fun):
            return np.full((m_y, m_x), float(rhs_fun), dtype=float)
        if callable(rhs_fun):
            return np.asarray([[rhs_fun(float(xi), float(yj)) for xi in x_int]
                               for yj in y_int], dtype=float)
        arr = np.asarray(rhs_fun, dtype=float)
        if arr.shape != (m_y, m_x):
            raise ValueError("rhs debe tener forma (m_y, m_x).")
        return arr

    rhs_vec = _rhs_eval(rhs).reshape(-1)

    bcs = {
        "left": {"raw": left_bc, "pts_full": y, "pts_int": y_int},
        "right": {"raw": right_bc, "pts_full": y, "pts_int": y_int},
        "bottom": {"raw": bottom_bc, "pts_full": x, "pts_int": x_int},
        "top": {"raw": top_bc, "pts_full": x, "pts_int": x_int},
    }

    for side, info in bcs.items():
        tag, data = info["raw"]
        tag = tag.upper()
        if tag not in ("D", "N"):
            raise ValueError("Solo se permiten condiciones 'D' o 'N'.")
        info["type"] = tag
        if tag == "D":
            val_full = _eval(data, info["pts_full"])
            info["values_full"] = val_full
            info["values_int"] = val_full[1:-1]
        else:
            info["flux"] = _eval(data, info["pts_int"])

    if (bcs["left"]["type"], bcs["right"]["type"]).count("D") == 2:
        if bcs["bottom"]["type"] != "N" or bcs["top"]["type"] != "N":
            raise ValueError("Si izquierda/derecha son Dirichlet, arriba/abajo deben ser Neumann.")
    elif (bcs["bottom"]["type"], bcs["top"]["type"]).count("D") == 2:
        if bcs["left"]["type"] != "N" or bcs["right"]["type"] != "N":
            raise ValueError("Si arriba/abajo son Dirichlet, izquierda/derecha deben ser Neumann.")
    else:
        raise ValueError("Debe haber exactamente un par opuesto Dirichlet.")

    h_x = (b - a) / (m_x + 1)
    h_y = (d - c) / (m_y + 1)
    inv_hx2 = 1.0 / (h_x * h_x)
    inv_hy2 = 1.0 / (h_y * h_y)

    n = m_x * m_y
    main = np.full(n, 2.0 * (inv_hx2 + inv_hy2), dtype=float)
    east = np.full(n - 1, -inv_hx2, dtype=float) if n > 1 else np.empty(0, dtype=float)
    west = east.copy()
    north = np.full(n - m_x, -inv_hy2, dtype=float) if m_y > 1 else np.empty(0, dtype=float)
    south = north.copy()

    if m_x > 1:
        east[m_x - 1::m_x] = 0.0
        west[m_x - 1::m_x] = 0.0

    # Aportaciones Dirichlet / Neumann (vectorizado)
    if bcs["left"]["type"] == "D":
        rhs_vec[0:n:m_x] += inv_hx2 * bcs["left"]["values_int"]
    else:
        rhs_vec[0:n:m_x] += (2.0 / h_x) * bcs["left"]["flux"]
        west[::m_x] = 0.0
        if m_x > 1:
            east[0:n:m_x] = -2.0 * inv_hx2

    if bcs["right"]["type"] == "D":
        rhs_vec[m_x - 1:n:m_x] += inv_hx2 * bcs["right"]["values_int"]
    else:
        rhs_vec[m_x - 1:n:m_x] += (2.0 / h_x) * bcs["right"]["flux"]
        if m_x > 1:
            west[m_x - 2::m_x] = -2.0 * inv_hx2
        east[m_x - 1::m_x] = 0.0

    if bcs["bottom"]["type"] == "D":
        rhs_vec[:m_x] += inv_hy2 * bcs["bottom"]["values_int"]
    else:
        rhs_vec[:m_x] += (2.0 / h_y) * bcs["bottom"]["flux"]
        if m_y > 1:
            north[:m_x] = -2.0 * inv_hy2
            south[:m_x] = 0.0

    if bcs["top"]["type"] == "D":
        rhs_vec[-m_x:] += inv_hy2 * bcs["top"]["values_int"]
    else:
        rhs_vec[-m_x:] += (2.0 / h_y) * bcs["top"]["flux"]
        if m_y > 1:
            south[-m_x:] = -2.0 * inv_hy2
            north[(m_y - 1) * m_x:] = 0.0

    data = [main]
    offsets = [0]
    if east.size:
        data.append(east)
        offsets.append(1)
    if west.size:
        data.append(west)
        offsets.append(-1)
    if north.size:
        data.append(north)
        offsets.append(m_x)
    if south.size:
        data.append(south)
        offsets.append(-m_x)

    A = diags(data, offsets, shape=(n, n), format="csr")
    U_int = spsolve(A, rhs_vec).reshape((m_y, m_x))

    U = np.zeros((m_y + 2, m_x + 2), dtype=float)
    U[1:-1, 1:-1] = U_int

    if bcs["left"]["type"] == "D":
        U[:, 0] = bcs["left"]["values_full"]
    else:
        g = bcs["left"]["flux"]
        U[1:-1, 0] = (4.0 * U_int[:, 0] - U_int[:, 1] + 2.0 * h_x * g) / 3.0

    if bcs["right"]["type"] == "D":
        U[:, -1] = bcs["right"]["values_full"]
    else:
        g = bcs["right"]["flux"]
        U[1:-1, -1] = (4.0 * U_int[:, -1] - U_int[:, -2] + 2.0 * h_x * g) / 3.0

    if bcs["bottom"]["type"] == "D":
        U[0, :] = bcs["bottom"]["values_full"]
    else:
        g = bcs["bottom"]["flux"]
        U[0, 1:-1] = (4.0 * U_int[0, :] - U_int[1, :] + 2.0 * h_y * g) / 3.0

    if bcs["top"]["type"] == "D":
        U[-1, :] = bcs["top"]["values_full"]
    else:
        g = bcs["top"]["flux"]
        U[-1, 1:-1] = (4.0 * U_int[-1, :] - U_int[-2, :] + 2.0 * h_y * g) / 3.0

    return X, Y, U
