# SOR experiments for Section 7 and Homework 3

This project implements Algorithm 7.3 from Section 7.4 of Burden, Faires, and Burden, *Numerical Analysis*, 10th ed. It reproduces the textbook example and compares Jacobi, Gauss–Seidel, and SOR on Math 455 Homework 3.

- [SOR Findings Report](SOR%20Findings%20Report.docx) explains the method, validation checks, and results.
- [Review notebook](SOR_experiment_review.ipynb) shows the saved tables and plot. Start with the method check, then read the three-way comparison for Homework 3 Problems 12(b) and 13(b).
- [sor_tools](sor_tools/) contains the solver, experiment, and plotting code used by the notebook.

To rerun the notebook, install the dependencies in your Python environment.

    python -m pip install jupyter numpy matplotlib
