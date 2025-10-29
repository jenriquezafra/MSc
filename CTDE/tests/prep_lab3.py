import numpy as np


def solve_heat_equation_cn_2d(
    ax,
    bx,
    ay,
    by,
    nx,
    ny,
    *,
    dt,
    n_steps,
    initial_condition,
    boundary_conditions,
    diffusivity=1.0,
    store_full_solution=False,
):
    """
    Solve the 2-D heat equation u_t = diffusivity * (u_xx + u_yy) on a rectangle using Crank-Nicolson.

    Parameters
    ----------
    ax, bx : float
        Left and right limits of the spatial domain in x.
    ay, by : float
        Bottom and top limits of the spatial domain in y.
    nx, ny : int
        Number of interior grid points along x and y (boundary points are added automatically).
    dt : float
        Time step for the evolution.
    n_steps : int
        Number of time steps to advance (total time equals dt * n_steps).
    initial_condition : callable
        Function f(x, y) providing the initial condition on the interior nodes.
    boundary_conditions : dict
        Dictionary with callables for Dirichlet boundaries. Required keys: "left", "right", "bottom", "top".
        Each callable must accept (coords, t) and return the boundary values at time t, where coords is the
        coordinate array along the corresponding edge (x for bottom/top, y for left/right). Scalars are allowed.
    diffusivity : float, optional
        Thermal diffusivity coefficient (default 1.0).
    store_full_solution : bool, optional
        If True, return the full time history with shape (n_steps + 1, ny + 2, nx + 2). Otherwise only the final field.

    Returns
    -------
    times : ndarray
        Array of time instants, shape (n_steps + 1,).
    X, Y : ndarray
        Meshgrid arrays including boundaries, each with shape (ny + 2, nx + 2).
    U : ndarray
        Temperature field. Shape is (n_steps + 1, ny + 2, nx + 2) if store_full_solution is True, otherwise (ny + 2, nx + 2).
    """
    required_keys = {"left", "right", "bottom", "top"}
    if not required_keys.issubset(boundary_conditions):
        missing = sorted(required_keys.difference(boundary_conditions))
        raise ValueError(f"boundary_conditions missing keys: {missing}")

    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive integers (number of interior nodes).")
    if dt <= 0.0:
        raise ValueError("dt must be a positive float.")
    if n_steps <= 0:
        raise ValueError("n_steps must be a positive integer.")
    if diffusivity <= 0.0:
        raise ValueError("diffusivity must be positive.")

    x = np.linspace(ax, bx, nx + 2, dtype=float)
    y = np.linspace(ay, by, ny + 2, dtype=float)
    X, Y = np.meshgrid(x, y, indexing="xy")

    hx = (bx - ax) / (nx + 1)
    hy = (by - ay) / (ny + 1)
    hx2 = hx * hx
    hy2 = hy * hy

    times = np.linspace(0.0, dt * n_steps, n_steps + 1, dtype=float)

    left_fun = boundary_conditions["left"]
    right_fun = boundary_conditions["right"]
    bottom_fun = boundary_conditions["bottom"]
    top_fun = boundary_conditions["top"]

    def _boundary_array(t):
        bottom_vals = np.broadcast_to(np.asarray(bottom_fun(x, t), dtype=float), x.shape).copy()
        top_vals = np.broadcast_to(np.asarray(top_fun(x, t), dtype=float), x.shape).copy()
        left_vals = np.broadcast_to(np.asarray(left_fun(y, t), dtype=float), y.shape).copy()
        right_vals = np.broadcast_to(np.asarray(right_fun(y, t), dtype=float), y.shape).copy()

        boundary = np.zeros((ny + 2, nx + 2), dtype=float)
        boundary[:, 0] = left_vals
        boundary[:, -1] = right_vals
        boundary[0, :] = bottom_vals
        boundary[-1, :] = top_vals
        return boundary

    def _boundary_contribution(boundary):
        contrib = np.zeros((ny, nx), dtype=float)
        contrib[:, 0] += boundary[1:-1, 0] / hx2
        contrib[:, -1] += boundary[1:-1, -1] / hx2
        contrib[0, :] += boundary[0, 1:-1] / hy2
        contrib[-1, :] += boundary[-1, 1:-1] / hy2
        return contrib.ravel(order="C")

    # Assemble dense Laplacian and Crank-Nicolson matrices on interior nodes.
    num_unknowns = nx * ny
    factor = 0.5 * diffusivity * dt

    laplacian = np.zeros((num_unknowns, num_unknowns), dtype=float)

    def linear_index(i, j):
        return j * nx + i

    for j in range(ny):
        for i in range(nx):
            k = linear_index(i, j)
            laplacian[k, k] = -2.0 / hx2 - 2.0 / hy2
            if i > 0:
                laplacian[k, linear_index(i - 1, j)] = 1.0 / hx2
            if i < nx - 1:
                laplacian[k, linear_index(i + 1, j)] = 1.0 / hx2
            if j > 0:
                laplacian[k, linear_index(i, j - 1)] = 1.0 / hy2
            if j < ny - 1:
                laplacian[k, linear_index(i, j + 1)] = 1.0 / hy2

    identity = np.eye(num_unknowns, dtype=float)
    matrix_left = identity - factor * laplacian
    matrix_right = identity + factor * laplacian

    def apply_right(vec):
        return matrix_right @ vec

    def solve_linear(rhs):
        return np.linalg.solve(matrix_left, rhs)

    U_current = np.zeros((ny + 2, nx + 2), dtype=float)
    interior_ic = initial_condition(X[1:-1, 1:-1], Y[1:-1, 1:-1])
    U_current[1:-1, 1:-1] = np.asarray(interior_ic, dtype=float)

    boundary_at_t0 = _boundary_array(times[0])
    U_current[:, 0] = boundary_at_t0[:, 0]
    U_current[:, -1] = boundary_at_t0[:, -1]
    U_current[0, :] = boundary_at_t0[0, :]
    U_current[-1, :] = boundary_at_t0[-1, :]

    u_vec = U_current[1:-1, 1:-1].ravel(order="C")

    if store_full_solution:
        history = np.zeros((n_steps + 1, ny + 2, nx + 2), dtype=float)
        history[0] = U_current

    for step in range(n_steps):
        t_n = times[step]
        t_np1 = times[step + 1]

        boundary_n = _boundary_array(t_n)
        boundary_np1 = _boundary_array(t_np1)

        rhs = apply_right(u_vec)
        rhs += factor * (_boundary_contribution(boundary_n) + _boundary_contribution(boundary_np1))

        u_vec = solve_linear(rhs)
        U_current[1:-1, 1:-1] = u_vec.reshape((ny, nx), order="C")

        U_current[:, 0] = boundary_np1[:, 0]
        U_current[:, -1] = boundary_np1[:, -1]
        U_current[0, :] = boundary_np1[0, :]
        U_current[-1, :] = boundary_np1[-1, :]

        if store_full_solution:
            history[step + 1] = U_current

    if store_full_solution:
        return times, X, Y, history
    return times, X, Y, U_current


