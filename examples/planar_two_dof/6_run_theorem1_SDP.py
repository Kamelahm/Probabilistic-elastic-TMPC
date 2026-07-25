"""
Theorem 1 (region route): effective-uncertainty identification.

What this solves
----------------
    min   sum(s) + lambda_sigma tr(Sigma_kappa) + sum|lambda_k|
    s.t.  data consistency (10f)   |e_i'(r_k - c_om - G_om lam_k)| <= eta_a t_i - ||e_i' Z_k(s)||_1
          Schur / epigraph (10c-e,g)
          COVERAGE      (10h)      |g_i| + eta_a sqrt(Sigma_ii) >= eps_bar_i + |e_i' c_om|

and exports, for Lemma 3 / Theorem 2:

    (A_bar, B_bar) = mid(F)      nominal model   (NOT C*)
    Gamma          = 0.5(ub-lb)  mismatch box
    h_wtilde       = |g| + eta_a sqrt(diag(Sigma*))   confidence-region half-widths
    c_omega*, Sigma_kappa*
"""

import pickle
import numpy as np
import cvxpy as cp
from scipy.stats import norm
from scipy.optimize import linprog

from examples.planar_two_dof import P_EXAMPLE_2_DOF
from identification_common import entrywise_generators, interval_hull_of_F

np.set_printoptions(precision=4, suppress=False, linewidth=140)

# ---- must match 0_check_priors.py ---------------------------------------- #
K_BAR = 500
DELTA_A = 0.05
SAFETY = 2.0
G_SCALE = 2.0
OMEGA_FLOOR = 1e-6
KAPPA = 1.10                      # a priori spectral bound (Assumption 3), FIXED
LAMBDA_SIGMA = 1.0
LAMBDA_LAM = 1.0

# ----------------------------------------------------------------------- #
#  1. Dataset                                                             #
# ----------------------------------------------------------------------- #
p_d = P_EXAMPLE_2_DOF / "data" / "dof_2_ef_0.1"
with open(p_d / "id_dataset.pckl", "rb") as f:
    D = pickle.load(f)
Y0, Y1, U0, V_hat = D["Y0"], D["Y1"], D["U0"], D["V_hat"]
n, m, T = D["n"], D["m"], U0.shape[1]
A_lin, B_lin = D["A_lin"], D["B_lin"]      # ideal ZOH double integrator
sigma_bar = float(np.sqrt(np.diag(V_hat).max()))
print(f"Loaded: n={n}, m={m}, T={T}, K_bar={K_BAR},  sigma = {sigma_bar:.2e}")

# ----------------------------------------------------------------------- #
#  2. Priors (Assumption 3).  Read from 0_check_priors.py if available, so  #
#     the two scripts cannot disagree; otherwise recompute with the SAME    #
#     formula (Chebyshev eps_min, one LP per row).                          #
# ----------------------------------------------------------------------- #
eta_a = norm.ppf(1.0 - DELTA_A / (2.0 * n * (T + K_BAR)))
noise = eta_a * sigma_bar * np.sqrt(1.0 + KAPPA ** 2)

try:
    pri = np.load(p_d / "priors.npz")
    omega_bar = pri["omega_bar"]
    assert int(pri["T"]) == T, "priors.npz was computed for a different T"
    assert abs(float(pri["kappa"]) - KAPPA) < 1e-12, "priors.npz uses a different kappa"
    print("[Priors]    loaded from priors.npz")
except FileNotFoundError:
    Z0 = np.vstack([Y0, U0])
    d = Z0.shape[0]
    A_ub = np.block([[Z0.T, -np.ones((T, 1))], [-Z0.T, -np.ones((T, 1))]])
    c_lp = np.zeros(d + 1)
    c_lp[-1] = 1.0
    eps_min = np.array([
        linprog(c_lp, A_ub=A_ub, b_ub=np.concatenate([Y1[a, :], -Y1[a, :]]),
                bounds=[(None, None)] * d + [(0.0, None)], method="highs").x[-1]
        for a in range(n)])
    omega_bar = np.maximum(SAFETY * np.maximum(eps_min - noise, 0.0), OMEGA_FLOOR)
    print(f"[Priors]    priors.npz not found -- recomputed "
          f"(eps_min = {eps_min})")

eps_bar = omega_bar + noise
g = G_SCALE * omega_bar
G_omega = np.diag(g)
s_omega = n

print(f"[Priors]    omega_bar={omega_bar}  sigma_bar={sigma_bar:.1e}  kappa={KAPPA}")
print(f"[Envelope]  eta_a={eta_a:.4f}   noise={noise:.4e}   eps_bar={eps_bar}")

# ----------------------------------------------------------------------- #
#  3. Generators                                                          #
# ----------------------------------------------------------------------- #
generators, idx = entrywise_generators(n, m)
q = len(generators)

# ----------------------------------------------------------------------- #
#  4. Data-feasible set F: hull -> midpoint model + Gamma (Lemma 3)        #
# ----------------------------------------------------------------------- #
print("\nComputing interval hull of F (2q LPs) ...")

Theta_lin = np.hstack([A_lin, B_lin])
lb, ub = interval_hull_of_F(Y0, U0, Y1, eps_bar,
                            Theta_lin=Theta_lin, nq=n // 2, Ts=D["dt"],
                            struct_safety=2.0)
Gamma = 0.5 * (ub - lb)
Theta_mid = 0.5 * (lb + ub)
A_bar, B_bar = Theta_mid[:, :n], Theta_mid[:, n:]
Gamma_A, Gamma_B = Gamma[:, :n], Gamma[:, n:]
print(f"  Gamma: max A-radius = {Gamma_A.max():.3e}  "
      f"max B-radius = {Gamma_B.max():.3e}  sum = {Gamma.sum():.4e}")

# [6] DIAGNOSTIC (was a hard gate): is the a-priori kappa consistent with F?
kap_diag = np.linalg.norm(A_bar, 2) + Gamma_A.sum()
print(f"  [diag] sup_F ||A||_2 <= {kap_diag:.6f}  vs kappa = {KAPPA}  ->  "
      f"{'consistent' if kap_diag <= KAPPA else 'PRIOR MAY BE TOO TIGHT'}")

# what the identification buys (Section V-A)
W_lin = Y1 - A_lin @ Y0 - B_lin @ U0
W_mid = Y1 - A_bar @ Y0 - B_bar @ U0
print(f"  [model] ||mid(F)_A - A_lin||_2 = {np.linalg.norm(A_bar - A_lin, 2):.4e}"
      f"   ||mid(F)_B - B_lin||_2 = {np.linalg.norm(B_bar - B_lin, 2):.4e}")
print(f"          residual std vs (A_lin,B_lin) = {W_lin.std(axis=1)}")
print(f"          residual std vs mid(F)        = {W_mid.std(axis=1)}")

# ----------------------------------------------------------------------- #
#  5. Variables    ([1] kappa is gone)                                     #
# ----------------------------------------------------------------------- #
SB = sigma_bar                     # [7] nondimensionalisation scale
eps_h = eps_bar / SB               # ~ 6.9   (was 6.9e-5)
g_h = g / SB                       # ~ 0.2
G_omega_h = G_omega / SB

C_A = cp.Variable((n, n))
C_B = cp.Variable((n, m))
s_h = cp.Variable(q, nonneg=True)          # s = SB * s_h
Sig_h = cp.Variable((n, n), PSD=True)      # Sigma = SB^2 * Sig_h
c_h = cp.Variable(n)                       # c_omega = SB * c_h
lambdas = cp.Variable((s_omega, T))
t_h = cp.Variable(n, nonneg=True)          # t = SB * t_h

# ----------------------------------------------------------------------- #
#  6. Precompute V[k,i,j] = (G_A^(j) y_k + G_B^(j) u_k)_i                  #
# ----------------------------------------------------------------------- #
V = np.zeros((T, n, q))
for j, (G_A_j, G_B_j) in enumerate(generators):
    V[:, :, j] = (G_A_j @ Y0 + G_B_j @ U0).T
absV = np.abs(V)

# ----------------------------------------------------------------------- #
#  7. Constraints                                                          #
# ----------------------------------------------------------------------- #
cons = []

# Schur / epigraph are form-invariant under the sigma_bar scaling
Z_v = cp.diag(t_h)                                                  # (10c)
cons.append(cp.bmat([[Sig_h, Z_v], [Z_v.T, np.eye(n)]]) >> 0)       # (10d)
cons.append(cp.square(t_h) <= cp.diag(Sig_h))                       # (10g)
cons += [lambdas <= 1, lambdas >= -1]                               # (10b)

# data consistency (10f), divided through by sigma_bar
R_h = (Y1 - C_A @ Y0 - C_B @ U0) / SB                               # (10e)
ones_T = np.ones((1, T))
c_bc = cp.reshape(c_h, (n, 1), order="F") @ ones_T
Gl = G_omega_h @ lambdas
term2 = cp.vstack([absV[:, i, :] @ s_h for i in range(n)])
t_bc = cp.reshape(t_h, (n, 1), order="F") @ ones_T
cons.append(cp.abs(R_h - c_bc - Gl) + term2 <= eta_a * t_bc)

# [4] COVERAGE (10h), divided through by sigma_bar:
#     g_h + eta_a sqrt(Sig_h_ii) >= eps_h + |c_h|   (concave >= convex, DCP)
cons.append(eta_a * cp.sqrt(cp.diag(Sig_h)) >= eps_h + cp.abs(c_h) - np.abs(g_h))

# ----------------------------------------------------------------------- #
#  8. Solve   ([1] no lambda_kappa*kappa term)                             #
# ----------------------------------------------------------------------- #
# All three terms are now O(1).  The lambda term is averaged over T, otherwise
# its 8000 elements swamp the 4 entries of tr(Sig_h).
obj = (cp.sum(s_h)
       + LAMBDA_SIGMA * cp.trace(Sig_h)
       + LAMBDA_LAM * cp.sum(cp.abs(lambdas)) / T)

prob = cp.Problem(cp.Minimize(obj), cons)
print(f"\nSolving SDP: {len(cons)} constraint blocks, "
      f"{sum(v.size for v in prob.variables())} scalar vars")
prob.solve(solver=cp.CLARABEL, verbose=True)

print("\n" + "=" * 64)
print(f"status = {prob.status},  objective = {prob.value:.6e}")
print("=" * 64)
if prob.status not in ("optimal", "optimal_inaccurate"):
    raise SystemExit(
        "SDP did not solve.  If INFEASIBLE, the most likely cause is that the\n"
        "coverage constraint (10h) forces Sigma up to a level at which the data\n"
        "consistency constraint (10f) can no longer be met: check that\n"
        "eta_a*sqrt(Sigma_ii) ~ eps_bar - g exceeds eps_min for every row.")

C_star = np.hstack([C_A.value, C_B.value])
s_star = np.maximum(s_h.value, 0.0) * SB              # un-scale
Sig = Sig_h.value * SB ** 2
c_om = c_h.value * SB
t_val = t_h.value * SB

# region-route confidence-region half-widths
h_wtilde = np.abs(g) + eta_a * np.sqrt(np.maximum(np.diag(Sig), 0.0))

# ----------------------------------------------------------------------- #
#  9. Diagnostics                                                          #
# ----------------------------------------------------------------------- #
# [7] guard against a recurrence of the scaling failure: the optimum must sit
# at the analytic floor  max(coverage, data-consistency).
S_cov = ((eps_bar - np.abs(g)) / eta_a) ** 2
S_floor = S_cov.max()
overshoot = np.diag(Sig).max() / S_floor
print(f"\n[scaling]  Sigma_ii = {np.diag(Sig)}")
print(f"           analytic coverage floor = {S_floor:.4e}   "
      f"overshoot = {overshoot:.2f}x")
if overshoot > 2.0:
    print("           *** WARNING: Sigma is well above the floor -- the solve is")
    print("           *** likely inaccurate.  Check the solver status and scaling.")

cov_slack = (np.abs(g) + eta_a * np.sqrt(np.maximum(np.diag(Sig), 0.0))
             - (eps_bar + np.abs(c_om)))
print(f"\n[coverage] slack per coordinate = {cov_slack}   "
      f"({'OK' if cov_slack.min() >= -1e-9 else 'VIOLATED'})")
print(f"[c_omega]  {c_om}")
print(f"[Sigma]    diag = {np.diag(Sig)}   (true noise var = {sigma_bar**2:.3e})")
print(f"[h_wtilde] {h_wtilde}")
print(f"           eps_bar = {eps_bar}   -> h_wtilde >= eps_bar: "
      f"{'yes' if np.all(h_wtilde >= eps_bar - 1e-12) else 'NO'}")
print(f"[s*]       active(>1e-9) = {int((s_star > 1e-9).sum())}/{q}   "
      f"sum = {s_star.sum():.3e}")
if s_star.sum() < 1e-9:
    print("           NOTE: s* collapsed to 0 -- expected under the region route,")
    print("           since nothing constrains M(C*,s*,G) from below any more.")
    print("           M is a singleton; it carries no guarantee.  Do not describe")
    print("           it as a nontrivial identified model set in the paper.")
print(f"[nominal]  the model exported for Lemma 3 / Theorem 2 is mid(F), NOT C*:")
print(f"           ||mid(F) - C*||_2 = {np.linalg.norm(Theta_mid - C_star, 2):.4e}")

# ----------------------------------------------------------------------- #
# 10. Save                                                                 #
# ----------------------------------------------------------------------- #
identified = {
    # --- region-route quantities consumed downstream --------------------- #
    "A_bar": A_bar, "B_bar": B_bar,          # nominal = mid(F)
    "Gamma": Gamma, "Gamma_A": Gamma_A, "Gamma_B": Gamma_B,
    "h_wtilde": h_wtilde,
    "c_omega": c_om, "Sigma_kappa": Sig, "G_omega": G_omega, "g": g,
    # --- envelope / priors ------------------------------------------------ #
    "eta_a": eta_a, "eps_bar": eps_bar, "delta_a": DELTA_A, "K_bar": K_BAR,
    "omega_bar": omega_bar, "sigma_bar": sigma_bar, "kappa": KAPPA,
    "hull_lb": lb, "hull_ub": ub,
    # --- identification-stage only (no role in the guarantee) ------------- #
    "C_star": C_star, "s_star": s_star, "generators": generators, "t": t_val,
}
with open(p_d / "identified_model.pckl", "wb") as f:
    pickle.dump(identified, f)
print(f"\nSaved -> {p_d / 'identified_model.pckl'}")
print("NEXT: python examples/planar_two_dof/7_run_theorem2_tube.py")