"""Iteration matrices, reference systems, and result formatting."""

from collections import namedtuple

import numpy as np


def split(A):
    '''Return D, L, U with A = D + L + U (class convention).'''
    A = np.asarray(A, dtype=float)
    return np.diag(np.diag(A)), np.tril(A, -1), np.triu(A, 1)


def jacobi_matrix(A):
    D, L, U = split(A)
    return -np.linalg.solve(D, L + U)


def sor_matrix(A, omega):
    D, L, U = split(A)
    return np.linalg.solve(D + omega * L, (1 - omega) * D - omega * U)


def sor_vector(A, b, omega):
    D, L, U = split(A)
    return omega * np.linalg.solve(D + omega * L, np.asarray(b, dtype=float))


def spectral_radius(T):
    return np.max(np.abs(np.linalg.eigvals(T)))


def optimal_omega(A):
    '''Evaluate the optimal-omega formula where its assumptions hold.'''
    return 2 / (1 + np.sqrt(1 - spectral_radius(jacobi_matrix(A)) ** 2))


def first_accurate(history, exact, threshold=0.5e-7):
    '''First k whose true error ||x^(k) - exact||_inf is below threshold, or None.'''
    for k, x in enumerate(history):
        if np.linalg.norm(x - exact, np.inf) < threshold:
            return k
    return None


System = namedtuple("System", ["name", "A", "b", "x0", "exact"])


def make_system(name, A, b, x0, exact=None):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    exact = np.linalg.solve(A, b) if exact is None else np.array(exact, dtype=float)
    return System(name, A, b, np.array(x0, dtype=float), exact)


def textbook_example():
    """Construct textbook Section 7.4, Example 1."""

    return make_system("Example 1", [[4, 3, 0], [3, 4, -1], [0, -1, 4]],
                       [24, 30, -24], [1, 1, 1], [3, 4, -5])


def is_positive_definite(A):
    return np.allclose(A, A.T) and np.all(np.linalg.eigvalsh(A) > 0)


def markdown_table(headings, rows):
    """Format experiment rows for the notebook."""

    def line(values):
        return "| " + " | ".join(map(str, values)) + " |"

    return "\n".join([line(headings), line(["---"] * len(headings)), *(line(row) for row in rows)])
