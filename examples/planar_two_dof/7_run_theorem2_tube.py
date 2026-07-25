"""
Theorem 2 (region route): contractive tube condition, augmented facet template.

"""

import pickle
import numpy as np
import cvxpy as cp
from scipy.linalg import solve_discrete_lyapunov, sqrtm
from scipy.signal import place_poles
from scipy.optimize import linprog
from examples.planar_two_dof import P_EXAMPLE_2_DOF

np.set_printoptions(precision=4, suppress=False, linewidth=140)

# ---- tube template design ------------------------------------------------ #
POLE_R = 0.90          # [T3] closed-loop pole radius (lambda >= POLE_R)
POLE_SPREAD = np.array([1.0, 0.98, 0.96, 0.94])   # relative pole placement
J_AUG = 16             # [T1] augmentation order -> 2n(J+1) = 136 facets
GM = 0.9270            # [T2] geometric ratio (from 7a_design_search.py)
H_E_0 = 0.17

# ---- nominal pair at which the INITIAL tube is designed ------------------ #
USE_WORST_CASE_NOMINAL = False
YBAR_WC = np.array([np.pi, np.pi, 2.0, 2.0])
UBAR_WC = np.array([20.0, 20.0])

U_MAX = 20.0
CLEARANCE = 0.12       # corridor clearance [rad]
LAM_MAX = 0.99
OPTIMIZE_K = False     
ADD_NESTING = True     # (26g)-(26i)
C_BAR_MARGIN = 1.10
SIGMA_W = 1.0

# ----------------------------------------------------------------------- #
#  1. Load identified quantities (region route)                            #
# ----------------------------------------------------------------------- #
p_d = P_EXAMPLE_2_DOF / "data" / "dof_2_ef_0.1"
with open(p_d / "identified_model.pckl", "rb") as f:
    M = pickle.load(f)

A_bar, B_bar = M["A_bar"], M["B_bar"]
Gamma_A, Gamma_B = M["Gamma_A"], M["Gamma_B"]
h_wtilde, c_omega, eta_a = M["h_wtilde"], M["c_omega"], M["eta_a"]
n, m = A_bar.shape[0], B_bar.shape[1]
nq = n // 2
print(f"Loaded (region route): n={n}, m={m},  eta_a={eta_a:.4f}")
print(f"  h_wtilde = {h_wtilde}")

# ----------------------------------------------------------------------- #
#  2. [T3] pole placement -> design gain                                   #
# ----------------------------------------------------------------------- #
poles = POLE_R * POLE_SPREAD[:n]
K_des = -place_poles(A_bar, B_bar, poles).gain_matrix
Acl_des = A_bar + B_bar @ K_des
rho_des = np.abs(np.linalg.eigvals(Acl_des)).max()
print(f"\n[design gain] poles at radius {POLE_R}: rho(Acl) = {rho_des:.4f}, "
      f"||K_des||_2 = {np.linalg.norm(K_des, 2):.1f}")

# ----------------------------------------------------------------------- #
#  3. [T1][T2] augmented template with geometric h                         #
# ----------------------------------------------------------------------- #
P_lyap = solve_discrete_lyapunov(Acl_des.T, np.eye(n))
P_sqrt = sqrtm(P_lyap).real
H0 = np.vstack([P_sqrt, -P_sqrt])

blocks, Mp = [H0], np.eye(n)
for j in range(1, J_AUG + 1):
    Mp = Mp @ Acl_des
    blocks.append(H0 @ Mp)
H_e = np.vstack(blocks)
s_f = H_e.shape[0]

gm = GM if GM is not None else 1.03 * POLE_R
h_shape = np.concatenate([gm ** j * np.ones(H0.shape[0])
                          for j in range(J_AUG + 1)])
h_e_k = H_E_0 * h_shape
print(f"[template]  J={J_AUG} -> {s_f} facets,  geometric ratio gm={gm:.3f}, "
      f"H_E_0={H_E_0}")

# M_ek: E_k is contained in the j=0 block {|P_sqrt e| <= H_E_0}, so the
# P_sqrt-based bound is valid (and conservative) for the augmented set.
M_scale = np.linalg.norm(np.linalg.pinv(P_sqrt), 2)
M_e_k = H_E_0 * M_scale


def lam_pure(H, h, Acl):
    """max_i [ max_{e: He<=h} H_i Acl e ] / h_i -- exact, one LP per facet."""
    out = []
    for i in range(H.shape[0]):
        r = linprog(-(H[i] @ Acl), A_ub=H, b_ub=h,
                    bounds=[(None, None)] * H.shape[1], method="highs")
        if not r.success:
            return np.inf
        out.append((-r.fun) / h[i])
    return max(out)


lam_des = lam_pure(H_e, h_e_k, Acl_des)
print(f"[template]  pure contraction at K_des: lambda = {lam_des:.4f}  "
      f"(floor rho = {rho_des:.4f})")

# ----------------------------------------------------------------------- #
#  4. Facet constants (19)-(20) recomputed for the NEW template            #
# ----------------------------------------------------------------------- #
absHe = np.abs(H_e)
rowsA = absHe @ Gamma_A
rowsB = absHe @ Gamma_B
phi_A = np.linalg.norm(rowsA, axis=1)
phi_B = np.linalg.norm(rowsB, axis=1)


def psi(ybar, ubar):
    return rowsA @ np.abs(ybar) + rowsB @ np.abs(ubar)


additive = H_e @ c_omega + absHe @ h_wtilde
y_bar_k, u_bar_k = ((YBAR_WC, UBAR_WC) if USE_WORST_CASE_NOMINAL
                    else (np.zeros(n), np.zeros(m)))
psi_k = psi(y_bar_k, u_bar_k)

# The Gamma terms enter (26b) against h_e,i = H_E_0*shape_i, so they must be
rot = (M_scale * (phi_A + np.linalg.norm(K_des, 2) * phi_B) / h_shape).max()
frac = ((additive + psi_k) / h_e_k).max()
gate = lam_des + rot
print(f"\n[budget]  lam_pure                            = {lam_des:.4f}")
print(f"          rot (Gamma terms, H_E_0-independent) = {rot:.4f}")
print(f"          gate = lam_pure + rot               = {gate:.4f}   "
      f"(hard: no H_E_0 helps)   {'OK' if gate < LAM_MAX else 'FAIL'}")
print(f"          (additive+psi)/h_e, worst facet     = {frac:.4f}")
print(f"          predicted lambda needed             = "
      f"{gate + frac:.4f}   (must be <= {LAM_MAX})")
if gate < LAM_MAX:
    print(f"          implied H_E_0 minimum               = "
          f"{frac * H_E_0 / (LAM_MAX - gate):.4f}   (current {H_E_0})")

# ----------------------------------------------------------------------- #
#  5. Theorem 2 SDP (26)                                                   #
# ----------------------------------------------------------------------- #
P_k = cp.Variable((s_f, s_f), nonneg=True)
K_k = cp.Variable((m, n)) if OPTIMIZE_K else K_des
rho_k = cp.Variable(nonneg=True)
lam_k = cp.Variable(nonneg=True)

cons = [
    P_k @ h_e_k <= (cp.multiply(lam_k, h_e_k) - additive
                    - M_e_k * phi_A - rho_k * M_e_k * phi_B - psi_k),   # (26b)
    P_k @ H_e == H_e @ (A_bar + B_bar @ K_k),                           # (26c)
    lam_k <= LAM_MAX, lam_k >= 0.01,
]
if OPTIMIZE_K:
    cons.append(cp.norm(K_k, 2) <= rho_k)                               # (26f)
else:
    cons.append(rho_k >= np.linalg.norm(K_des, 2))

if ADD_NESTING:                                                         # (26g)-(26i)
    H_u = np.vstack([np.eye(m), -np.eye(m)])
    Hp0 = np.linalg.pinv(H_e)
    cbar_init = C_BAR_MARGIN * (np.abs(H_u @ K_des @ Hp0) @ h_e_k)
    M_u = cp.Variable((H_u.shape[0], s_f), nonneg=True)
    cons += [M_u @ H_e == H_u @ K_k, M_u @ h_e_k <= cbar_init]
    print(f"[nesting] cbar^(-1) = {cbar_init}   (u_max = {U_MAX})")

prob = cp.Problem(cp.Minimize(rho_k + SIGMA_W * lam_k), cons)
print(f"\nSolving contractive tube SDP  ({s_f} facets, "
      f"{sum(v.size for v in prob.variables())} scalar vars) ...")
prob.solve(solver=cp.CLARABEL, verbose=False)
print(f"Status: {prob.status}")
if prob.status not in ("optimal", "optimal_inaccurate"):
    raise SystemExit(
        "\nInfeasible.  Check, in order:\n"
        f"  1. 'predicted lambda needed' above vs LAM_MAX = {LAM_MAX}\n"
        "  2. raise H_E_0 (shrinks the (additive+psi)/h_e term)\n"
        "  3. raise J_AUG, or lower POLE_R (but watch rot grow with ||K||)\n"
        "  4. set USE_WORST_CASE_NOMINAL=False -- psi at the origin is ~9x\n"
        "     smaller, and the controller re-evaluates psi every step anyway\n"
        "  5. set ADD_NESTING=False to test whether the nesting block binds")

K_val = K_k.value if OPTIMIZE_K else K_des
lam_val, rho_val = float(lam_k.value), float(rho_k.value)
P_val = P_k.value
h_next_v = lam_val * h_e_k

# ----------------------------------------------------------------------- #
#  6. Diagnostics                                                          #
# ----------------------------------------------------------------------- #
print("\n" + "=" * 62)
print(f"  lambda_k = {lam_val:.4f}   rho_k = {rho_val:.4f}   "
      f"||K_k||_2 = {np.linalg.norm(K_val, 2):.1f}")
print(f"  rho(A+BK_k) = "
      f"{np.abs(np.linalg.eigvals(A_bar + B_bar @ K_val)).max():.4f}")
print("=" * 62)

err = np.linalg.norm(P_val @ H_e - H_e @ (A_bar + B_bar @ K_val))
slack = (lam_val * h_e_k - additive - M_e_k * phi_A
         - rho_val * M_e_k * phi_B - psi_k) - P_val @ h_e_k
print(f"[Sanity] residual of (26c)  = {err:.2e}")
print(f"[Sanity] min slack in (26b) = {slack.min():.4e}  (>= 0)")

# independent verification of lambda-contractivity at the OPTIMAL K
lam_check = lam_pure(H_e, h_e_k, A_bar + B_bar @ K_val)
print(f"[Verify] pure contraction at K_k = {lam_check:.4f}  "
      f"(SDP lambda_k = {lam_val:.4f})")
if lam_check > lam_val + 1e-6:
    print("         *** the template does not support this K: the SDP moved K")
    print("         *** away from K_des.  Set OPTIMIZE_K = False to pin it.")

Hp = np.linalg.pinv(H_e)
q_hw = (np.abs(Hp[:nq, :]) @ h_e_k).max()
H_u_box = np.vstack([np.eye(m), -np.eye(m)])
cbar = np.abs(H_u_box @ K_val @ Hp) @ h_e_k
print(f"[Corridor] tube half-width in q = {q_hw:.4f} rad  "
      f"(clearance {CLEARANCE})  "
      f"{'OK' if q_hw < CLEARANCE else '*** TOO WIDE ***'}")
print(f"[Input]    cbar = {cbar}")
print(f"           budget h_u - cbar = {U_MAX - cbar}   "
      f"{'OK' if cbar.max() < U_MAX else '*** NO INPUT LEFT ***'}")

# ----------------------------------------------------------------------- #
#  7. Save                                                                 #
# ----------------------------------------------------------------------- #
tube_data = {
    "H_e": H_e, "h_e_0": h_e_k, "h_e_next": h_next_v, "h_shape": h_shape,
    "P_sqrt": P_sqrt, "P_lyap": P_lyap, "P_k": P_val,
    "K_k": K_val, "K_des": K_des, "lambda_k": lam_val, "rho_k": rho_val,
    "M_e_k": M_e_k, "M_scale": M_scale,
    "A_bar": A_bar, "B_bar": B_bar,
    "Gamma_A": Gamma_A, "Gamma_B": Gamma_B,
    "phi_A": phi_A, "phi_B": phi_B, "rowsA": rowsA, "rowsB": rowsB,
    "additive": additive, "h_wtilde": h_wtilde, "c_omega": c_omega,
    "eta_a": eta_a, "cbar_0": cbar,
    "template": {"POLE_R": POLE_R, "J_AUG": J_AUG, "gm": gm, "H_E_0": H_E_0},
}
with open(p_d / "tube_initial.pckl", "wb") as f:
    pickle.dump(tube_data, f)
print(f"\nSaved -> {p_d / 'tube_initial.pckl'}")