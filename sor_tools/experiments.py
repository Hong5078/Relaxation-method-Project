"""Verify Algorithm 7.3 and compare the Section 7 and Homework 3 systems."""

import contextlib
import io
from itertools import zip_longest

import numpy as np

from .helpers import (
    first_accurate, is_positive_definite, jacobi_matrix, make_system,
    markdown_table, optimal_omega, sor_matrix, sor_vector, spectral_radius,
    split, textbook_example,
)
from .plots import omega_plot
from .solvers import gauss_seidel, jacobi, sor


def _format_iterate(x):
    return "(" + ", ".join(f"{value:.6f}" for value in x) + ")"


def run_implementation_1():
    """Verify the SOR solver against textbook Example 1."""

    _, A, b, x0, exact = textbook_example()

    # Transposed so that row k holds (x1, x2, x3) at iteration k.
    TABLE_7_3 = np.array([
        [1, 5.2500000, 3.1406250, 3.0878906, 3.0549316, 3.0343323, 3.0214577, 3.0134110],
        [1, 3.8125000, 3.8828125, 3.9267578, 3.9542236, 3.9713898, 3.9821186, 3.9888241],
        [1, -5.0468750, -5.0292969, -5.0183105, -5.0114441, -5.0071526, -5.0044703, -5.0027940],
    ]).T

    TABLE_7_4 = np.array([
        [1, 6.3125000, 2.6223145, 3.1333027, 2.9570512, 3.0037211, 2.9963276, 3.0000498],
        [1, 3.5195313, 3.9585266, 4.0102646, 4.0074838, 4.0029250, 4.0009262, 4.0002586],
        [1, -6.6501465, -4.6004238, -5.0966863, -4.9734897, -5.0057135, -4.9982822, -5.0003486],
    ]).T

    # A tight tolerance keeps the run going well past the iterates being compared.
    gs = gauss_seidel(A, b, x0, tol=1e-12, max_iter=200)
    sr = sor(A, b, x0, omega=1.25, tol=1e-12, max_iter=200)

    ROUNDING = 0.5e-7 + 1e-12
    for title, result, book in [("Table 7.3: Gauss-Seidel (omega = 1)", gs, TABLE_7_3),
                                ("Table 7.4: SOR (omega = 1.25)", sr, TABLE_7_4)]:
        computed = np.array(result.history[:8])
        assert np.all(np.abs(computed - book) <= ROUNDING), f"{title} does not match the book"

    def gs_step(x):
        x1 = -0.75 * x[1] + 6
        x2 = -0.75 * x1 + 0.25 * x[2] + 7.5
        x3 = 0.25 * x2 - 6
        return np.array([x1, x2, x3])


    def sor_step(x):
        x1 = -0.25 * x[0] - 0.9375 * x[1] + 7.5
        x2 = -0.9375 * x1 - 0.25 * x[1] + 0.3125 * x[2] + 9.375
        x3 = 0.3125 * x2 - 0.25 * x[2] - 7.5
        return np.array([x1, x2, x3])


    for name, result, step in [("Gauss-Seidel", gs, gs_step), ("SOR", sr, sor_step)]:
        x = x0.astype(float)
        for k in range(1, 21):
            x = step(x)
            assert np.max(np.abs(x - result.history[k])) < 1e-12, f"{name} differs at k = {k}"

    gs_count = first_accurate(gs.history, exact)
    sr_count = first_accurate(sr.history, exact)


    assert gs_count == 34 and sr_count == 14

    tol = 1e-7
    gs_stop = gauss_seidel(A, b, x0, tol=tol, max_iter=100)
    sor_stop = sor(A, b, x0, omega=1.25, tol=tol, max_iter=100)
    for result in (gs_stop, sor_stop):
        assert result.converged and np.linalg.norm(result.x - exact, np.inf) < 10 * tol

    with contextlib.redirect_stdout(io.StringIO()):
        result = sor(A, b, x0, omega=1.25, tol=1e-7, max_iter=5)
        assert not result.converged and result.iterations == 5 and len(result.history) == 6

        try:
            sor([[0, 1], [1, 0]], [1, 1], [0, 0], omega=1.25, tol=1e-7, max_iter=5)
        except ValueError:
            pass
        else:
            raise AssertionError("a zero diagonal entry should raise ValueError")

    sweep_table = markdown_table(
        ("Sweep", "Gauss–Seidel iterate", "SOR iterate (ω = 1.25)"),
        [(k, _format_iterate(g), _format_iterate(s))
         for k, (g, s) in enumerate(zip(gs.history[:8], sr.history[:8]))],
    )
    count_table = markdown_table(
        ("Criterion", "Gauss–Seidel sweeps", "SOR sweeps (ω = 1.25)"),
        [
            ("True error below 5 × 10⁻⁸", gs_count, sr_count),
            ("Change between sweeps below 10⁻⁷", gs_stop.iterations, sor_stop.iterations),
        ],
    )
    return sweep_table, count_table


def run_implementation_2():
    """Compare relaxation parameters and validate the iteration matrices."""

    EX1 = textbook_example()
    HW3 = [
        make_system("HW3 #12(a)", [[3, -1, 1], [3, 6, 2], [3, 3, 7]], [1, 0, 7], [0, 0, 0]),
        make_system("HW3 #12(b)", [[10, -1, 0], [-1, 10, -2], [0, -2, 10]], [9, 7, 6], [0, 0, 0]),
        make_system("HW3 #14", [[2, -1, 1], [2, 2, 2], [-1, -1, 2]], [-1, 4, -5], [0, 0, 0], [1, 2, -1]),
        make_system("HW3 #15", [[1, 2, -2], [1, 1, 1], [2, 2, 1]], [7, 2, 5], [0, 0, 0], [1, 2, -1]),
    ]

    for s in [EX1] + HW3:
        assert np.allclose(s.A @ s.exact, s.b), s.name

    A_notes, b_notes = [[3, -1], [-1, 2]], [5, 4]

    jac = jacobi(A_notes, b_notes, [0, 0], tol=1e-12, max_iter=100)
    gs = sor(A_notes, b_notes, [0, 0], omega=1.0, tol=1e-12, max_iter=100)

    assert np.allclose(jac.history[1], [5 / 3, 2], atol=1e-12)
    assert np.allclose(jac.history[2], [7 / 3, 17 / 6], atol=1e-12)
    assert np.allclose(gs.history[1], [5 / 3, 17 / 6], atol=1e-12)
    assert np.allclose(jac.x, [14 / 5, 17 / 5]) and np.allclose(gs.x, [14 / 5, 17 / 5])

    rng = np.random.default_rng(455)
    for s in [EX1] + HW3:
        D, _, _ = split(s.A)
        for _ in range(5):
            x = rng.normal(size=3)
            one_step = jacobi(s.A, s.b, x, tol=np.inf, max_iter=1).x
            assert np.allclose(one_step, jacobi_matrix(s.A) @ x + np.linalg.solve(D, s.b), atol=1e-12)
            for omega in [0.3, 1.0, 1.25, 1.9]:
                one_step = sor(s.A, s.b, x, omega=omega, tol=np.inf, max_iter=1).x
                assert np.allclose(one_step, sor_matrix(s.A, omega) @ x + sor_vector(s.A, s.b, omega), atol=1e-12)

    s = HW3[1]
    # Force two sweeps for the answer-key checks.
    with contextlib.redirect_stdout(io.StringIO()):
        two_steps_11 = sor(s.A, s.b, s.x0, omega=1.1, tol=0, max_iter=2)
        two_steps_13 = sor(s.A, s.b, s.x0, omega=1.3, tol=0, max_iter=2)
    ex5 = sor(s.A, s.b, s.x0, omega=1.2, tol=1e-3, max_iter=100)
    omega_7 = optimal_omega(s.A)
    ex7 = sor(s.A, s.b, s.x0, omega=omega_7, tol=1e-3, max_iter=100)

    assert np.allclose(two_steps_11.x, [0.9876790, 0.9784935, 0.7899328], atol=0.5e-7)
    assert np.allclose(two_steps_13.x, [0.957073, 0.9903875, 0.7206569], atol=0.5e-6)
    assert ex5.iterations == 7 and np.allclose(ex5.x, [0.9958341, 0.9579041, 0.7915756], atol=0.5e-7)
    assert abs(omega_7 - 1.012823) < 0.5e-6
    assert ex7.iterations == 4 and np.allclose(ex7.x, [0.9957846, 0.9578935, 0.7915788], atol=0.5e-7)

    OMEGAS = np.round(np.arange(0.02, 1.99, 0.01), 2)


    def rho_curve(A):
        return np.array([spectral_radius(sor_matrix(A, w)) for w in OMEGAS])


    RHO = {s.name: rho_curve(s.A) for s in [EX1] + HW3}

    # A finer grid, used to locate the best omega and the edge of convergence more precisely
    FINE = np.round(np.arange(0.001, 2.0, 0.001), 3)
    FINE_RHO = {s.name: np.array([spectral_radius(sor_matrix(s.A, w)) for w in FINE]) for s in [EX1] + HW3}


    def best_omega(name):
        '''The omega with the smallest spectral radius on the fine grid, and that radius.'''
        i = np.argmin(FINE_RHO[name])
        return FINE[i], FINE_RHO[name][i]

    # Theorem 7.24 (Kahan), for every system
    for s in [EX1] + HW3:
        assert np.all(RHO[s.name] >= np.abs(OMEGAS - 1) - 1e-9), s.name

    # Theorems 7.25 and 7.26, for the positive definite tridiagonal systems
    for s in [EX1, HW3[1]]:
        assert is_positive_definite(s.A)
        assert np.all(RHO[s.name] < 1)
        rho_j = spectral_radius(jacobi_matrix(s.A))
        rho_g = spectral_radius(sor_matrix(s.A, 1.0))
        w_opt = optimal_omega(s.A)
        assert abs(rho_g - rho_j ** 2) < 1e-12
        assert abs(spectral_radius(sor_matrix(s.A, w_opt)) - (w_opt - 1)) < 1e-6
        assert abs(best_omega(s.name)[0] - w_opt) <= 0.001  # fine-grid spacing


    # Example 2: rho(T_J) = sqrt(0.625), optimal omega about 1.24
    assert abs(spectral_radius(jacobi_matrix(EX1.A)) - np.sqrt(0.625)) < 1e-12
    assert round(optimal_omega(EX1.A), 2) == 1.24

    # Homework 3 #14 and #15, parts (a) and (c)
    A14, A15 = HW3[2].A, HW3[3].A
    assert abs(spectral_radius(jacobi_matrix(A14)) - np.sqrt(5) / 2) < 1e-12
    assert abs(spectral_radius(sor_matrix(A14, 1.0)) - 1 / 2) < 1e-12
    # rho(T_J) = 0 for #15: T_J is nilpotent (T_J^3 = 0). Checked directly, because eigvals() of a
    # nilpotent matrix is only accurate to about 1e-5.
    assert np.allclose(np.linalg.matrix_power(jacobi_matrix(A15), 3), 0, atol=1e-12)
    assert abs(spectral_radius(sor_matrix(A15, 1.0)) - 2) < 1e-12

    MAX_ITER = 3000


    def solve_for_study(s, omega=None):
        """Run Jacobi or the relaxed solver under the same iteration settings."""
        # Divergent runs can overflow or reach the sweep limit.
        with np.errstate(over="ignore", invalid="ignore"), contextlib.redirect_stdout(io.StringIO()):
            if omega is None:
                return jacobi(s.A, s.b, s.x0, tol=1e-12, max_iter=MAX_ITER)
            return sor(s.A, s.b, s.x0, omega=omega, tol=1e-12, max_iter=MAX_ITER)

    def iterations_needed(s, omega=None):
        result = solve_for_study(s, omega)
        return first_accurate(result.history, s.exact)

    example_counts = [iterations_needed(EX1, omega) for omega in OMEGAS]
    gs_count = iterations_needed(EX1, 1.0)
    sor_count = iterations_needed(EX1, 1.25)
    assert gs_count == 34 and sor_count == 14

    rows = []
    for system in [EX1] + HW3:
        omega, _ = best_omega(system.name)
        counts = [iterations_needed(system, w) for w in (None, 1.0, omega)]
        rows.append((system.name, f"{omega:.3f}", *(n if n is not None else "Diverges" for n in counts)))

    table = markdown_table(("System", "Selected ω", "Jacobi", "Gauss–Seidel", "Relaxed solver"), rows)

    homework = HW3[1]
    homework_omega, _ = best_omega(homework.name)
    homework_methods = (
        ("Jacobi", None),
        ("Gauss–Seidel", 1.0),
        (f"SOR (ω = {homework_omega:.3f})", homework_omega),
    )
    homework_rows = []
    error_columns = []
    for method, omega in homework_methods:
        history = solve_for_study(homework, omega).history
        count = first_accurate(history, homework.exact)
        assert count is not None, f"{method} did not reach the error target on {homework.name}"
        homework_rows.append((method, *map(_format_iterate, history[1:3]), count))
        errors = np.max(np.abs(np.asarray(history[:count + 1]) - homework.exact), axis=1)
        error_columns.append([f"{error:.3e}" for error in errors])

    homework_table = markdown_table(
        ("Method", "Sweep 1", "Sweep 2", "Sweeps to the error target"), homework_rows,
    )
    error_table = markdown_table(
        ("Sweep", *(method for method, _ in homework_methods)),
        [(k, *errors) for k, errors in enumerate(zip_longest(*error_columns, fillvalue="—"))],
    )

    example_radii = RHO[EX1.name]
    sample_omegas = {0.80, 1.00, 1.20, 1.22, 1.24, 1.25, 1.50}
    omega_table = markdown_table(
        ("ω", "Spectral radius", "Sweeps to the error target"),
        [(f"{omega:.2f}", f"{rho:.4f}", count)
         for omega, rho, count in zip(OMEGAS, example_radii, example_counts)
         if omega in sample_omegas],
    )
    figure = omega_plot(OMEGAS, example_counts, example_radii, optimal_omega(EX1.A),
                        gs_count, sor_count, spectral_radius(sor_matrix(EX1.A, 1.0)))
    tables = [
        ("Homework 3 Problem 12(b)", homework_table),
        ("Homework 3 Problem 12(b) error by sweep", error_table),
        ("Results across the five systems", table),
        ("Section 7 Example 1 at selected ω values", omega_table),
    ]
    return tables, figure
