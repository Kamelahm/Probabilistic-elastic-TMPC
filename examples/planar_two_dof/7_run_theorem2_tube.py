"""
Solve the contractive tube optimization (Theorem 2, Eq. 22) to obtain:
    P_k, K_k, rho_k, lambda_k, h_{e,k+1}

"""

import pickle
import numpy as np
import cvxpy as cp
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov, sqrtm
from examples.planar_two_dof import P_EXAMPLE_2_DOF

POS_TIGHTEN = 1.0
VEL_LOOSEN  = 1.0
W_INFLATE   = 1.0
H_E_0       = 0.035
NOISE_MODEL = "gaussian"        

# ----------------------------------------------------------------------- #
#  1. Load identified model                                               #
# ----------------------------------------------------------------------- #
p_d = P_EXAMPLE_2_DOF / "data" / "dof_2_ef_0.1"
with open(p_d / "identified_model.pckl", "rb") as f:
    M = pickle.load(f)

A_bar, B_bar = M["A_id"], M["B_id"]
s_star       = M["s_star"]
Sigma_kappa  = M["Sigma_kappa"]
c_omega      = M["c_omega"]
G_omega_star = M["G_omega"]          # the IDENTIFIED bounded generator (= I)
generators   = M["generators"]
eta, kappa   = M["eta"], M["kappa"]

n, m = A_bar.shape[0], B_bar.shape[1]
q    = len(generators)
nq   = n // 2
print(f"Loaded identified model:  n={n}, m={m}, q={q}")
print(f"  ||A_bar||_2 = {np.linalg.norm(A_bar, 2):.4f}   kappa = {kappa:.4f}   "
      f"sum(s*) = {s_star.sum():.6f}")

# ----------------------------------------------------------------------- #
#  2. Stabilizing LQR + Lyapunov polytope                                 #
# ----------------------------------------------------------------------- #
Q_lqr = np.diag([100.0, 100.0, 10.0, 10.0])
R_lqr = np.eye(m) * 0.01
P_dare = solve_discrete_are(A_bar, B_bar, Q_lqr, R_lqr)
K_lqr  = -np.linalg.solve(R_lqr + B_bar.T @ P_dare @ B_bar,
                          B_bar.T @ P_dare @ A_bar)
M_cl   = A_bar + B_bar @ K_lqr
print(f"\nLQR closed-loop eig magnitudes: {np.abs(np.linalg.eigvals(M_cl))}")
print(f"||K_lqr||_2 = {np.linalg.norm(K_lqr, 2):.2f}")

P_lyap = solve_discrete_lyapunov(M_cl.T, np.eye(n))
P_sqrt = sqrtm(P_lyap).real
print(f"Lyapunov P eigenvalues: {np.linalg.eigvalsh(P_lyap)}")

P_sqrt_scaled = P_sqrt.copy()
P_sqrt_scaled[:, :nq] *= POS_TIGHTEN
P_sqrt_scaled[:, nq:] *= VEL_LOOSEN
H_e = np.vstack([P_sqrt_scaled, -P_sqrt_scaled])
s_f = H_e.shape[0]
print(f"\nTube polytope: {s_f} facets")

# ----------------------------------------------------------------------- #
#  3. Tube + disturbance budget  (THE FIX IS HERE)                        #
# ----------------------------------------------------------------------- #
h_e_k = H_E_0 * np.ones(s_f)
M_e_k = H_E_0 * np.linalg.norm(np.linalg.pinv(P_sqrt_scaled), 2)

# Residual, kept for REPORTING only (not used as a second noise bound).
with open(p_d / "id_dataset.pckl", "rb") as f:
    dataset = pickle.load(f)
Y0_, Y1_, U0_ = dataset["Y0"], dataset["Y1"], dataset["U0"]
w_max = np.max(np.abs(Y1_ - A_bar @ Y0_ - B_bar @ U0_), axis=1)
print(f"\nResidual max |w_tilde| per dim (report only): {w_max}")

# Term 7: Gaussian part of the mixed zonotope (per facet).
D_kappa = eta * np.sqrt(np.maximum(np.diag(H_e @ Sigma_kappa @ H_e.T), 0))

# Term 3: bounded-zonotope generator G_omega_star.
if NOISE_MODEL == "gaussian":
    G_omega_term = np.zeros(s_f)
elif NOISE_MODEL == "deterministic":
    # Deterministic worst case from the residual; then D_kappa must be dropped.
    G_dev = np.diag(W_INFLATE * w_max)
    G_omega_term = np.sum(np.abs(H_e @ G_dev), axis=1)
    D_kappa = np.zeros(s_f)            # avoid the double-count
else:
    raise ValueError(NOISE_MODEL)
print(f"NOISE_MODEL = {NOISE_MODEL}:  "
      f"max G_omega_term = {G_omega_term.max():.2e},  max D_kappa = {D_kappa.max():.2e}")

He_c_omega = H_e @ c_omega
y_bar_k, u_bar_k = np.zeros(n), np.zeros(m)
sum_GA = np.zeros(s_f); sum_GB = np.zeros(s_f); sum_nom = np.zeros(s_f)
for i in range(q):
    sum_GA  += s_star[i] * np.linalg.norm(H_e @ generators[i][0], axis=1)
    sum_GB  += s_star[i] * np.linalg.norm(H_e @ generators[i][1], axis=1)
    sum_nom += s_star[i] * np.abs(
        H_e @ (generators[i][0] @ y_bar_k + generators[i][1] @ u_bar_k))

# Budget headroom check (independent of the SDP)
budget = (He_c_omega + G_omega_term + M_e_k * sum_GA + sum_nom + D_kappa)
print(f"Per-facet noise budget max = {budget.max():.4f}   vs tube h_e = {H_E_0:.4f}")
if budget.max() >= H_E_0:
    print("  WARNING: noise budget already exceeds the tube radius before "
          "contraction; raise H_E_0 or lower noise.")

# ----------------------------------------------------------------------- #
#  4. Theorem 2 SDP                                                       #
# ----------------------------------------------------------------------- #
P_k    = cp.Variable((s_f, s_f), nonneg=True)
K_k    = cp.Variable((m, n))
rho_k  = cp.Variable(nonneg=True)
lam_k  = cp.Variable(nonneg=True)
h_next = cp.Variable(s_f, nonneg=True)
sigma_w = 1.0

constraints = [
    P_k @ h_e_k <= (lam_k * h_e_k - He_c_omega - G_omega_term
                    - M_e_k * sum_GA - rho_k * M_e_k * sum_GB
                    - sum_nom - D_kappa),
    P_k @ H_e == H_e @ (A_bar + B_bar @ K_k),
    h_next == lam_k * h_e_k,
    cp.norm(K_k, 2) <= rho_k,
    lam_k <= 0.999,
    lam_k >= 0.001,
]
prob = cp.Problem(cp.Minimize(rho_k + sigma_w * lam_k), constraints)
print("\nSolving contractive tube SDP ...")
prob.solve(solver=cp.CLARABEL, verbose=False)
print(f"Status: {prob.status}")
if prob.status not in ("optimal", "optimal_inaccurate"):
    print("\nStill infeasible. With the double-count removed this means the "
          "noise genuinely doesn't fit the tube at this V_hat: raise H_E_0 or "
          "reduce the noise level.")
    raise SystemExit

# ----------------------------------------------------------------------- #
#  5. Diagnostics                                                         #
# ----------------------------------------------------------------------- #
K_val, lam_val, rho_val = K_k.value, float(lam_k.value), float(rho_k.value)
P_val, h_next_v = P_k.value, h_next.value
print("\n" + "=" * 60)
print(f"  Contraction factor lambda_k  = {lam_val:.4f}")
print(f"  Gain norm bound    rho_k     = {rho_val:.4f}")
print(f"  ||K_k||_2          (actual)  = {np.linalg.norm(K_val, 2):.4f}")
print(f"  Closed-loop |A+BK_k| (max)   = "
      f"{np.abs(np.linalg.eigvals(A_bar + B_bar @ K_val)).max():.4f}")
print("=" * 60)

err_22c = np.linalg.norm(P_val @ H_e - H_e @ (A_bar + B_bar @ K_val))
slack   = (lam_val * h_e_k - He_c_omega - G_omega_term - M_e_k * sum_GA
           - rho_val * M_e_k * sum_GB - sum_nom - D_kappa) - P_val @ h_e_k
print(f"\n[Sanity] Residual of (22c) = {err_22c:.2e}")
print(f"[Sanity] Min slack in (22b) = {slack.min():.4e}  (should be >= 0)")

# ----------------------------------------------------------------------- #
#  6. Save                                                                #
# ----------------------------------------------------------------------- #
tube_data = {
    "H_e": H_e, "P_sqrt": P_sqrt_scaled, "P_lyap": P_lyap,
    "h_e_0": h_e_k, "h_e_next": h_next_v,
    "K_k": K_val, "K_lqr": K_lqr, "P_k": P_val,
    "lambda_k": lam_val, "rho_k": rho_val, "M_e_k": M_e_k,
    "D_kappa": D_kappa, "G_omega": G_omega_star, "G_omega_term": G_omega_term,
    "sum_GA": sum_GA, "sum_GB": sum_GB, "sum_nom": sum_nom,
    "He_c_omega": He_c_omega,
}
with open(p_d / "tube_initial.pckl", "wb") as f:
    pickle.dump(tube_data, f)
print(f"\nTube parameters saved to: {p_d / 'tube_initial.pckl'}")