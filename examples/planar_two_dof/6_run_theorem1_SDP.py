"""
Solve the SDP of Theorem 1 (Eq. 8) to identify a data-consistent
matrix zonotope M(C*, s*, G) and covariance bound Sigma*_kappa.

"""

import pickle
import numpy as np
import cvxpy as cp
from scipy.stats import norm
from examples.planar_two_dof import P_EXAMPLE_2_DOF
 
# ----------------------------------------------------------------------- #
#  1. Load dataset                                                        #
# ----------------------------------------------------------------------- #
p_d = P_EXAMPLE_2_DOF / "data" / "dof_2_ef_0.1"
with open(p_d / "id_dataset.pckl", "rb") as f:
    data = pickle.load(f)
 
U0, Y0, Y1 = data["U0"], data["Y0"], data["Y1"]
V_hat      = data["V_hat"]
A_lin      = data["A_lin"]
B_lin      = data["B_lin"]
n, m, T    = data["n"], data["m"], U0.shape[1]
print(f"Loaded: n={n}, m={m}, T={T}")
 
# ----------------------------------------------------------------------- #
#  2. Fixed generators  G^(i) = [G_A^(i), G_B^(i)]                         #
#     Entry-wise generators: q = n*n + n*m                                 #
# ----------------------------------------------------------------------- #
generators = []
for i in range(n):
    for j in range(n):
        G_A = np.zeros((n, n)); G_A[i, j] = 1.0
        generators.append((G_A, np.zeros((n, m))))
for i in range(n):
    for j in range(m):
        G_B = np.zeros((n, m)); G_B[i, j] = 1.0
        generators.append((np.zeros((n, n)), G_B))
q = len(generators)
print(f"Number of generators q = {q}")
 
# Fixed disturbance-zonotope generator G_omega in R^{n x s_omega}
s_omega = n
G_omega = np.eye(n) * 1.0
 
# ----------------------------------------------------------------------- #
#  3. Confidence level:  eta = Phi^{-1}(1 - delta/(2 n T))                 #
# ----------------------------------------------------------------------- #
delta = 0.05
eta   = norm.ppf(1.0 - delta / (2.0 * n * T))
print(f"delta = {delta}, eta = {eta:.4f}")
 
# ----------------------------------------------------------------------- #
#  4. Decision variables  (C, s, kappa, Sigma_kappa, c_omega, lambda, t)  #
#     Z_v is NOT an independent variable: per the header, z_v = t.         #
# ----------------------------------------------------------------------- #
C_A         = cp.Variable((n, n))
C_B         = cp.Variable((n, m))
s_zon       = cp.Variable(q,  nonneg=True)        # (8b) s >= 0
kappa       = cp.Variable(nonneg=True)            # (8b) kappa >= 0
Sigma_kappa = cp.Variable((n, n), PSD=True)
c_omega     = cp.Variable(n)
lambdas     = cp.Variable((s_omega, T))           # {bar_lambda_k}
t           = cp.Variable(n, nonneg=True)         # (8b) t >= 0
 
# Fixed weights lambda_kappa, lambda_sigma  (lambda on sum s_i is 1 per (8a))
lambda_kappa = 0.0
lambda_sigma = 1.0
 
# ----------------------------------------------------------------------- #
#  5. Precompute V[k,i,j] = (G_A^(j) y_k + G_B^(j) u_k)_i                  #
# ----------------------------------------------------------------------- #
V = np.zeros((T, n, q))
for j, (G_A_j, G_B_j) in enumerate(generators):
    V[:, :, j] = (G_A_j @ Y0 + G_B_j @ U0).T
absV = np.abs(V)
 
# ----------------------------------------------------------------------- #
#  6. Constraints                                                         #
# ----------------------------------------------------------------------- #
constraints = []
 
# (8d)  kappa >= ||C_A||_2 + sum_i s_i ||G_A^(i)||_2
norms_GA = np.array([np.linalg.norm(g[0], 2) for g in generators])
constraints.append(kappa >= cp.norm(C_A, 2) + s_zon @ norms_GA)
 
# (8c)+(8e)  Z_v = diag(z_v) with z_v = t  ->  Schur LMI gives
#            Sigma_kappa >= diag(t^2). This is the operative covariance bound.
Z_v = cp.diag(t)
constraints.append(cp.bmat([[Sigma_kappa, Z_v],
                            [Z_v.T,        np.eye(n)]]) >> 0)
 

constraints += [lambdas <= 1, lambdas >= -1]
 
# (8f)  r_k = y_{k+1} - C_A y_k - C_B u_k
R = Y1 - C_A @ Y0 - C_B @ U0                                  # (n, T), affine
 
# (8g)  |e_i^T (r_k - c_omega - G_omega bar_lambda_k)| <= eta t_i - ||e_i^T Z_k(s)||_1
ones_T     = np.ones((1, T))
c_omega_bc = cp.reshape(c_omega, (n, 1), order="F") @ ones_T  # signed center, (n, T)
Gl         = G_omega @ lambdas                               # signed generators, (n, T)
# term2[i,k] = ||e_i^T Z_k(s)||_1 = sum_j s_j |V[k,i,j]|   (s_j >= 0)
term2      = cp.vstack([absV[:, i, :] @ s_zon for i in range(n)])   # (n, T)
t_bc       = cp.reshape(t, (n, 1), order="F") @ ones_T       # (n, T)
constraints.append(cp.abs(R - c_omega_bc - Gl) + term2 <= eta * t_bc)
 

constraints.append(cp.square(t) <= cp.diag(Sigma_kappa))
 
# ----------------------------------------------------------------------- #
#  7. Objective (8a)                                                      #
# ----------------------------------------------------------------------- #
obj = (cp.sum(s_zon)                            # sum_i s_i      (weight 1)
       + lambda_kappa * kappa                   # lambda_kappa * kappa
       + lambda_sigma * cp.trace(Sigma_kappa)   # lambda_sigma * tr(Sigma_kappa)
       + cp.sum(cp.abs(lambdas)))               # sum_{k=0}^{T-1} ||bar_lambda_k||_1
 
prob = cp.Problem(cp.Minimize(obj), constraints)
 
print(f"\nProblem size: {len(constraints)} constraints, "
      f"{sum(v.size for v in prob.variables())} scalar variables")
print("Solving SDP ...\n")
prob.solve(solver=cp.CLARABEL, verbose=True)
 
# ----------------------------------------------------------------------- #
#  8. Diagnostics                                                         #
# ----------------------------------------------------------------------- #
print("\n" + "=" * 60)
print(f"SDP status: {prob.status}")
print(f"Optimal objective: {prob.value:.6f}")
print("=" * 60)
 
if prob.status not in ("optimal", "optimal_inaccurate"):
    print("SDP did not solve cleanly.")
    raise SystemExit
 
A_id            = C_A.value
B_id            = C_B.value
s_star          = np.maximum(s_zon.value, 0)
Sigma_kappa_val = Sigma_kappa.value
c_omega_val     = c_omega.value
t_val           = t.value
z_v_val         = t_val            # z_v = t  (see header)
kappa_val       = float(kappa.value)
 
print(f"\n[Identified center C* = (A_id, B_id)]")
print(f"A_id =\n{A_id}")
print(f"\nDifference  A_id - A_lin (should be small):\n{A_id - A_lin}")
print(f"\nB_id =\n{B_id}")
print(f"\nDifference  B_id - B_lin (should be small):\n{B_id - B_lin}")
print(f"\n||A_id - A_lin||_F = {np.linalg.norm(A_id - A_lin, 'fro'):.6e}")
print(f"||A_id - A_lin||_2 = {np.linalg.norm(A_id - A_lin, 2):.6e}")
 
print(f"\n[Zonotope scaling s*]")
active = np.where(s_star > 1e-6)[0]
print(f"Total generators: {q}, active (s_i > 1e-6): {len(active)}")
print(f"Max s_i: {s_star.max():.6f},  sum s_i: {s_star.sum():.6f}")
 
print(f"\n[Spectral bound kappa = {kappa_val:.4f}]   ||A_id||_2 = {np.linalg.norm(A_id, 2):.4f}")
 
print(f"\n[Disturbance center c_omega]\n{c_omega_val}")
 
print(f"\n[Covariance bound Sigma*_kappa]\n{Sigma_kappa_val}")
print(f"\ndiag(Sigma_kappa) = {np.diag(Sigma_kappa_val)}")
print(f"t^2 (should equal diag Sigma at optimum) = {t_val**2}")
print(f"Schur lower bound check  min eig[[Sigma, diag t],[diag t, I]] = "
      f"{np.linalg.eigvalsh(np.block([[Sigma_kappa_val, np.diag(t_val)],[np.diag(t_val), np.eye(n)]])).min():.3e}")
 
print(f"\n[Confidence half-widths]")
print(f"  t_i                       = {t_val}")
print(f"  eta * t_i  (= eta sqrt Sigma_ii) = {eta * t_val}")
print(f"  sqrt(diag V_hat)          = {np.sqrt(np.diag(V_hat))}")
 
# ----------------------------------------------------------------------- #
#  9. Save                                                                #
# ----------------------------------------------------------------------- #
identified = {
    "A_id":        A_id,
    "B_id":        B_id,
    "s_star":      s_star,
    "Sigma_kappa": Sigma_kappa_val,
    "c_omega":     c_omega_val,
    "G_omega":     G_omega,
    "kappa":       kappa_val,
    "z_v":         z_v_val,          # = t
    "t":           t_val,
    "generators":  generators,
    "eta":         eta,
    "delta":       delta,
}
out = p_d / "identified_model.pckl"
with open(out, "wb") as f:
    pickle.dump(identified, f)
print(f"\nIdentified model saved to: {out}")
 
# Sanity: residuals under the identified model
residuals_id = Y1 - A_id @ Y0 - B_id @ U0
print(f"\n[Residual analysis under identified model]")
print(f"  Mean per dim: {residuals_id.mean(axis=1)}")
print(f"  Std  per dim: {residuals_id.std(axis=1)}")
print(f"  Max |.|:      {np.abs(residuals_id).max(axis=1)}")
print(f"  Std should be close to sqrt(diag(V_hat)) = {np.sqrt(np.diag(V_hat))}")
 
# Persistence-of-excitation diagnostic
Z = np.vstack([Y0, U0])
sv_Z = np.linalg.svd(Z, compute_uv=False)
print(f"\n[Persistence of excitation]")
print(f"  Regressor shape: {Z.shape}")
print(f"  Rank: {np.linalg.matrix_rank(Z, tol=1e-6)} / {Z.shape[0]}")
print(f"  Singular values: {sv_Z}")
print(f"  Condition number: {sv_Z[0] / sv_Z[-1]:.2f}")