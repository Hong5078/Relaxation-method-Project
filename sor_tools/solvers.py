"""Jacobi, Gauss–Seidel, and successive over-relaxation."""

from collections import namedtuple

import numpy as np


SORResult = namedtuple("SORResult", ["x", "iterations", "converged", "history"])


def sor(A, b, x0, omega, tol, max_iter, norm=np.inf):
    '''Solve Ax = b by SOR (Burden, Faires & Burden, Algorithm 7.3).

    Returns SORResult(x, iterations, converged, history), where history[k] is
    the iterate x^(k) and history[0] is x0.
    '''
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    XO = np.array(x0, dtype=float)
    n = len(b)

    if np.any(np.diag(A) == 0):
        raise ValueError("SOR requires every diagonal entry a_ii to be nonzero")

    x = np.zeros(n)
    history = [XO.copy()]

    k = 1                                                     # Step 1
    while k <= max_iter:                                      # Step 2
        for i in range(n):                                    # Step 3
            x[i] = (1 - omega) * XO[i] + (1 / A[i, i]) * (
                omega * (-A[i, :i] @ x[:i] - A[i, i + 1:] @ XO[i + 1:] + b[i])
            )
        history.append(x.copy())

        if np.linalg.norm(x - XO, norm) < tol:                # Step 4
            return SORResult(x.copy(), k, True, history)

        k += 1                                                # Step 5
        XO = x.copy()                                         # Step 6

    print("Maximum number of iterations exceeded")            # Step 7
    return SORResult(x.copy(), max_iter, False, history)


def jacobi(A, b, x0, tol, max_iter, norm=np.inf):
    '''Solve Ax = b by the Jacobi method. Returns the same fields as sor().'''
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    XO = np.array(x0, dtype=float)
    n = len(b)

    if np.any(np.diag(A) == 0):
        raise ValueError("Jacobi requires every diagonal entry a_ii to be nonzero")

    history = [XO.copy()]
    for k in range(1, max_iter + 1):
        x = np.array([(b[i] - A[i, :i] @ XO[:i] - A[i, i + 1:] @ XO[i + 1:]) / A[i, i]
                      for i in range(n)])
        history.append(x.copy())
        if np.linalg.norm(x - XO, norm) < tol:
            return SORResult(x, k, True, history)
        XO = x

    print("Maximum number of iterations exceeded")
    return SORResult(x, max_iter, False, history)


def gauss_seidel(A, b, x0, tol, max_iter, norm=np.inf):
    """Run the SOR update with omega equal to one."""

    return sor(A, b, x0, omega=1.0, tol=tol, max_iter=max_iter, norm=norm)
