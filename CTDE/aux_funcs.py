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