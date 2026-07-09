"""
Theorem 1 (Eq. 8) identification SDP, 6-DOF, VECTORIZED.
Identical math to the loop version, but constraints are built as array
operations so CVXPY compiles in seconds instead of hours.
"""
import pickle
import numpy as np
import cvxpy as cp
from scipy.stats import norm
from examples.six_dof import P_EXAMPLE_6_DOF

REDUCED_GENERATORS = False

# ----------------------------------------------------------------------- #
#  1. Load dataset                                                        #
# ----------------------------------------------------------------------- #
p_d = P_EXAMPLE_6_DOF / "data" / "dof_6_ef_0.02"
with open(p_d / "id_dataset.pckl", "rb") as f:
    data = pickle.load(f)

U0, Y0, Y1 = data["U0"], data["Y0"], data["Y1"]
V_hat      = data["V_hat"]
A_lin      = data["A_lin"]
B_lin      = data["B_lin"]
n, m, T    = data["n"], data["m"], U0.shape[1]
print(f"Loaded: n={n}, m={m}, T={T}")

# ----------------------------------------------------------------------- #
#  2. Generators                                                          #
# ----------------------------------------------------------------------- #
generators = []
if not REDUCED_GENERATORS:
    for i in range(n):
        for j in range(n):
            G_A = np.zeros((n, n)); G_A[i, j] = 1.0
            generators.append((G_A, np.zeros((n, m))))
    for i in range(n):
        for j in range(m):
            G_B = np.zeros((n, m)); G_B[i, j] = 1.0
            generators.append((np.zeros((n, n)), G_B))
else:
    nq = n // 2
    for i in range(nq, n):
        for j in range(n):
            G_A = np.zeros((n, n)); G_A[i, j] = 1.0
            generators.append((G_A, np.zeros((n, m))))
        for j in range(m):
            G_B = np.zeros((n, m)); G_B[i, j] = 1.0
            generators.append((np.zeros((n, n)), G_B))
q = len(generators)
print(f"Number of generators q = {q}")

s_omega = n
G_omega = np.eye(n) * 1.0

# ----------------------------------------------------------------------- #
#  3. Confidence level                                                    #
# ----------------------------------------------------------------------- #
delta = 0.05
eta   = norm.ppf(1.0 - delta / (2.0 * n * T))
print(f"delta = {delta}, eta = {eta:.4f}")

# ----------------------------------------------------------------------- #
#  4. Decision variables                                                  #
# ----------------------------------------------------------------------- #
C_A         = cp.Variable((n, n))
C_B         = cp.Variable((n, m))
s_zon       = cp.Variable(q,  nonneg=True)
kappa       = cp.Variable(nonneg=True)
Sigma_kappa = cp.Variable((n, n), PSD=True)
c_omega     = cp.Variable(n)
lambdas     = cp.Variable((s_omega, T))
t = cp.Variable(n, nonneg=True)   # (8b): t >= 0, per-coordinate confidence half-width

w_s, w_kappa, w_sigma = 1.0, 0.0, 1.0

# ----------------------------------------------------------------------- #
#  5. Precompute V[k, i, j]  (vectorized over time)                       #
# ----------------------------------------------------------------------- #
print("Precomputing generator response tensor V (vectorized) ...")
V = np.zeros((T, n, q))
for j, (G_A_j, G_B_j) in enumerate(generators):
    V[:, :, j] = (G_A_j @ Y0 + G_B_j @ U0).T   # (T, n) per generator
Vabs = np.abs(V)                                # (T, n, q)

# ----------------------------------------------------------------------- #
#  6. Constraints (VECTORIZED)                                            #
# ----------------------------------------------------------------------- #
constraints = []

# (8d)
norms_GA = np.array([np.linalg.norm(g[0], 2) for g in generators])
constraints.append(kappa >= cp.norm(C_A, 2) + s_zon @ norms_GA)

# (8e) LMI
Z_v = cp.diag(t)
constraints.append(cp.bmat([[Sigma_kappa, Z_v], [Z_v.T, np.eye(n)]]) >> 0)

# lambda box
constraints += [lambdas <= 1, lambdas >= -1]

# (8f) residual  r_k = y_{k+1} - C_A y_k - C_B u_k
R = Y1 - C_A @ Y0 - C_B @ U0                                  # (n, T), affine

# (8g) |e_i^T (r_k - c_omega - G_omega lambda_k)| <= eta t_i - ||e_i^T Z_k(s)||_1
ones_T     = np.ones((1, T))
c_omega_bc = cp.reshape(c_omega, (n, 1), order="F") @ ones_T  # (n, T), signed center
Gl         = G_omega @ lambdas                               # (n, T), signed generators

# term2[i,k] = ||e_i^T Z_k(s)||_1 = sum_j s_j |V[k,i,j]|
term2 = cp.vstack([Vabs[:, i, :] @ s_zon for i in range(n)]) # (n, T)

t_bc = cp.reshape(t, (n, 1), order="F") @ ones_T             # (n, T)
constraints.append(cp.abs(R - c_omega_bc - Gl) + term2 <= eta * t_bc)

# (8h) variance epigraph: e_i^T Sigma_kappa e_i <= t_i^2
constraints.append(cp.square(t) <= cp.diag(Sigma_kappa))


# ----------------------------------------------------------------------- #
#  7. Objective                                                           #
# ----------------------------------------------------------------------- #
obj = (cp.sum(s_zon)                       # sum_i s_i           (weight fixed to 1 by the theorem)
       + w_kappa * kappa                   # lambda_kappa * kappa
       + w_sigma * cp.trace(Sigma_kappa)   # lambda_sigma * tr(Sigma_kappa)
       + cp.sum(cp.abs(lambdas)))          # sum_{k=0}^{T-1} ||lambda_k||_1  (ALL columns)

prob = cp.Problem(cp.Minimize(obj), constraints)

print(f"\nProblem size: {len(constraints)} constraint objects, "
      f"{sum(v.size for v in prob.variables())} scalar variables")
print("Solving SDP (vectorized; compilation should be fast) ...\n")
prob.solve(solver=cp.CLARABEL, verbose=True)

# ----------------------------------------------------------------------- #
#  8. Diagnostics                                                         #
# ----------------------------------------------------------------------- #
print("\n" + "="*60)
print(f"SDP status: {prob.status}")
print(f"Optimal objective: {prob.value:.6f}")
print("="*60)

if prob.status not in ("optimal", "optimal_inaccurate"):
    print("SDP did not solve cleanly.")
    raise SystemExit

A_id, B_id      = C_A.value, C_B.value
s_star          = np.maximum(s_zon.value, 0)
Sigma_kappa_val = Sigma_kappa.value
c_omega_val     = c_omega.value
kappa_val       = float(kappa.value)

print(f"\n[Identified center]")
print(f"||A_id - A_lin||_F = {np.linalg.norm(A_id - A_lin):.6e}")
print(f"||B_id - B_lin||_F = {np.linalg.norm(B_id - B_lin):.6e}")
rel_B = np.linalg.norm(B_id - B_lin) / np.linalg.norm(B_lin)
print(f"Relative error in B: {rel_B:.2%}")

print(f"\n[Zonotope scaling s*]")
active = np.where(s_star > 1e-6)[0]
print(f"Total generators: {q}, active: {len(active)}, "
      f"max s_i: {s_star.max():.6f}, sum: {s_star.sum():.6f}")

print(f"\n[kappa = {kappa_val:.4f}],  ||A_id||_2 = {np.linalg.norm(A_id, 2):.4f}")

# ----------------------------------------------------------------------- #
#  9. Save                                                                #
# ----------------------------------------------------------------------- #
identified = {
    "A_id": A_id, "B_id": B_id,
    "s_star": s_star, "Sigma_kappa": Sigma_kappa_val,
    "c_omega": c_omega_val, "G_omega": G_omega,
    "kappa": kappa_val, 
    "generators": generators, "eta": eta, "delta": delta,
}
out = p_d / "identified_model.pckl"
with open(out, "wb") as f:
    pickle.dump(identified, f)
print(f"\nIdentified model saved to: {out}")

residuals_id = Y1 - A_id @ Y0 - B_id @ U0
print(f"\n[Residuals under identified model]")
print(f"  Std per dim: {residuals_id.std(axis=1)}")
print(f"  Max |.|:     {np.abs(residuals_id).max(axis=1)}")