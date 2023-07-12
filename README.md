# randomized-trace-logdet-diag-diaginv
Some Python implementations of randomized, matrix-free algorithms for estimating $\text{tr}(A)$, $\log \det(A)$, $\text{diag}(A)$, and $\text{diag}(A^{-1})$. Here $A$ is SPD or SPSD. We also provide implementations of non-randomized methods requiring access to matrices for convenience.

This package is a work-in-progress. The goal is to implement some of the algorithms detailed in the references below. 


## References

<a id="1">[1]</a>  M.F. Hutchinson (1989). A Stochastic Estimator of the Trace of the Influence Matrix for Laplacian Smoothing Splines. Communications in Statistics - Simulation and Computation, 18(3), 1059-1076.

<a id="2">[2]</a> Avron, H., & Toledo, S. (2011). Randomized Algorithms for Estimating the Trace of an Implicit Symmetric Positive Semi-Definite Matrix. J. ACM, 58(2).

<a id="3">[3]</a> Roosta-Khorasani, F., & Ascher, U. (2015). Improved Bounds on Sample Size for Implicit Matrix Trace Estimators. Foundations of Computational Mathematics, 15(5), 1187–1212.

<a id="4">[4]</a> Saibaba, A., Alexanderian, A., & Ipsen, I. (2017). Randomized matrix-free trace and log-determinant estimators. Numerische Mathematik, 137(2), 353–395.

<a id="5">[5]</a> C. Bekas, E. Kokiopoulou, & Y. Saad (2007). An estimator for the diagonal of a matrix. Applied Numerical Mathematics, 57(11), 1214-1229.

<a id="6">[6]</a> Christos Boutsidis, Petros Drineas, Prabhanjan Kambadur, Eugenia-Maria Kontopoulou, & Anastasios Zouzias (2017). A randomized algorithm for approximating the log determinant of a symmetric positive definite matrix. Linear Algebra and its Applications, 533, 95-117.

<a id="7">[7]</a> Han, I., Malioutov, D.M., & Shin, J. (2015). Large-scale log-determinant computation through stochastic Chebyshev expansions. International Conference on Machine Learning.

<a id="8">[8]</a> Chen, J. (2016). How Accurately Should I Compute Implicit Matrix-Vector Products When Applying the Hutchinson Trace Estimator? SIAM J. Sci. Comput., 38.

<a id="9">[9]</a> Meyer, R.A., Musco, C., Musco, C., & Woodruff, D.P. (2020). Hutch++: Optimal Stochastic Trace Estimation. Proceedings of the SIAM Symposium on Simplicity in Algorithms, 2021, 142-155 .

<a id="10">[10]</a> Persson, D., Cortinovis, A., & Kressner, D. (2021). Improved variants of the Hutch++ algorithm for trace estimation. SIAM J. Matrix Anal. Appl., 43, 1162-1185.


