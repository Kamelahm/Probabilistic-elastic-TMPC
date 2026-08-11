"""
Table 1 / Figure 1 -- conservatism of the identified uncertainty set.
Region route of Theorem 1, planar 2-DOF manipulator.

"""
import os
import json
import itertools
from pathlib import Path

import numpy as np
import cvxpy as cp
from scipy.stats import norm
from scipy.linalg import sqrtm
from scipy.optimize import linprog
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)
np.set_printoptions(precision=4, suppress=True)

try:
    _here = Path(__file__).resolve().parent
except NameError:
    _here = Path.cwd()
OUT_DIR = Path(os.environ.get("THM1_OUT_DIR", _here / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================== switches ====
ISOTROPIC     = True      # Assumption 2 verbatim
KAPPA         = 1.10      # a priori spectral bound kappa_hat, FIXED
DELTA_A       = 0.05
K_BAR         = 500       # closed-loop horizon entering eta_a
SAFETY        = 2.0       # margin on the calibrated omega_hat
OMEGA_FLOOR   = 1e-8
GEN_SCALE     = 1.0       # g = GEN_SCALE * omega_hat when OPTIMIZE_G is False
OPTIMIZE_G    = False
LAMBDA_REG    = True      # L1 term on lambda in the objective
LAMBDA_SIGMA  = 1.0       # lambda_sigma of (11a): stochastic vs deterministic

ABLATION      = os.environ.get("THM1_ABLATION", "sdp")   # "sdp" | "prior_only"
assert ABLATION in ("sdp", "prior_only")

# --- M1: enforce Lemma 1 (per-sample covariance bound) inside (11).
ENFORCE_LEMMA1 = True

T_ID          = 1500
T_VAL         = 800
T_CAL         = 400

# Parametric uncertainty polytope Theta -- used for BOTH beta and the prior
THETA_MASS    = 0.20
THETA_LC2     = 0.12

# Calibrate the prior over the same Theta that beta covers.
MATCHED_THETA = True
THETA_MASS_CAL = THETA_MASS if MATCHED_THETA else 0.10
THETA_LC2_CAL  = THETA_LC2  if MATCHED_THETA else 0.06

CAL_SEED      = 2718      # fixed ACROSS corners: eps_min becomes a clean
                          # function of Theta rather than of the excitation
                          # realisation drawn at each corner
VERIFY_A3     = True
A3_T          = 400
A3_INTERIOR   = 8
A3_AMP_OOD    = 0.9

# --- diagnostics
RUN_SUPPORT_GAP = True    # interval hull of F vs exact support (T3)
SUPPORT_SAMPLES = 8

# ============================================================================
# 0. M1 diagnostics
# ============================================================================
def audit_theorem1(g, hw, c_om, eps_bar, R_id, eta_a, s_star=None,
                   sigma_bar_hw=None, rtol=2e-2, verbose=True):
    """Does (11) contribute anything to the SIZE of the description?

    Compares the two routes to the achieved half-widths h_wtilde:
        coverage (11f) : eps_bar + |c_omegabar|
        residual (11d) : max_k |r_k - c_omegabar|
    If (11f) binds everywhere the identified region IS the Lemma-2 envelope.
    """
    g, hw, c_om, eps_bar = map(np.asarray, (g, hw, c_om, eps_bar))
    n = len(hw)
    coverage_route = eps_bar + np.abs(c_om)                     # (11f)
    residual_route = np.abs(R_id - c_om[:, None]).max(axis=1)   # (11d)
    binds = ["coverage(11f)" if coverage_route[i] >= residual_route[i]
             else "residual(11d)" for i in range(n)]
    headroom = coverage_route / np.maximum(residual_route, 1e-300)
    equals_prior = bool(np.allclose(hw, coverage_route, rtol=rtol))
    inert = all(b == "coverage(11f)" for b in binds)

    out = dict(h_wtilde=hw.tolist(),
               coverage_route=coverage_route.tolist(),
               residual_route=residual_route.tolist(),
               binding=binds,
               headroom=headroom.tolist(),
               headroom_min=float(headroom.min()),
               headroom_max=float(headroom.max()),
               identified_equals_prior=equals_prior,
               centre_at_origin=bool(np.allclose(
                   c_om, 0.0, atol=rtol * np.abs(eps_bar).max())),
               sdp_inert=inert,
               s_star_sum=None if s_star is None else float(np.sum(s_star)),
               s_star_active=None if s_star is None
               else int((np.abs(s_star) > 1e-9).sum()))

    if verbose:
        print("\n" + "=" * 74)
        print("M1 AUDIT -- what does Theorem 1 contribute?")
        print("=" * 74)
        w = 14
        print(f"  {'coord':>5} {'h_wtilde':>{w}} {'coverage(11f)':>{w}} "
              f"{'residual(11d)':>{w}} {'binds':>15}")
        for i in range(n):
            print(f"  {i:>5} {hw[i]:{w}.4e} {coverage_route[i]:{w}.4e} "
                  f"{residual_route[i]:{w}.4e} {binds[i]:>15}")
        print(f"\n  centre c*_omegabar          = "
              f"{np.array2string(c_om, precision=3)}")
        if s_star is not None:
            print(f"  sum(s*) / active generators = {np.sum(s_star):.3e} / "
                  f"{int((np.abs(s_star) > 1e-9).sum())}")
        if sigma_bar_hw is not None:
            print(f"  Lemma-1 floor eta_a*sqrt(Sigma_bar_ii) = "
                  f"{np.array2string(np.atleast_1d(sigma_bar_hw), precision=5)}")
        print(f"  coverage/residual headroom  = "
              f"{headroom.min():.2f}x to {headroom.max():.2f}x")
        print(f"  identified region == prior? = "
              f"{'YES' if equals_prior else 'no'}")
        print("\n  VERDICT: ", end="")
        if inert:
            print("(11f) binds in EVERY coordinate.")
            print("           The identified region is the Lemma-2 envelope;")
            print("           (11) does not tighten it.  Contribution 1 must")
            print("           be stated as certified packaging, not as a")
            print("           reduction in conservatism.")
        else:
            tight = [i for i in range(n) if binds[i] == "residual(11d)"]
            print(f"(11d) binds in coordinates {tight}.")
            print("           The data does constrain the description there.")
        print("=" * 74)
    return out


def _row_ub(Z, y_row, eps_i):
    """Constraints of F restricted to one output row (see (13))."""
    return (np.vstack([Z.T, -Z.T]),
            np.concatenate([y_row + eps_i, -y_row + eps_i]))


def feasible_hull(Y1r, Z, eps):
    """(14): componentwise extrema of F.  Returns Theta_bar, Gamma."""
    n_, d = Y1r.shape[0], Z.shape[0]
    Tb, Gm = np.empty((n_, d)), np.empty((n_, d))
    for i in range(n_):
        A_ub, b_ub = _row_ub(Z, Y1r[i], eps[i])
        lo, hi = np.empty(d), np.empty(d)
        for b in range(d):
            cvec = np.zeros(d)
            cvec[b] = 1.0
            r1 = linprog(cvec, A_ub=A_ub, b_ub=b_ub,
                         bounds=[(None, None)] * d, method="highs")
            r2 = linprog(-cvec, A_ub=A_ub, b_ub=b_ub,
                         bounds=[(None, None)] * d, method="highs")
            if not (r1.success and r2.success):
                raise RuntimeError("hull LP failed; check Assumption 4 "
                                   "(full row rank Z_0)")
            lo[b], hi[b] = r1.fun, -r2.fun
        Tb[i], Gm[i] = 0.5 * (lo + hi), 0.5 * (hi - lo)
    return Tb, Gm


def support_gap(Y1r, Z, eps, Theta_bar, Gamma, n_samples=8, seed=0,
                verbose=True):
    """Box-hull bound (Lemma 3) vs exact support over F.

    Lemma 3 bounds |e_a^T [dA dB] psi| by sum_b Gamma_ab |psi_b|, the support
    of the BOX HULL of F.  F is a thin correlated slab, so the exact support
    is smaller.  The ratio is conservatism injected by the relaxation in (14),
    not by the uncertainty itself.
    """
    r_ = np.random.default_rng(seed)
    n_, T_ = Y1r.shape
    d = Z.shape[0]
    ratios = []
    for _ in range(n_samples):
        k = int(r_.integers(0, T_))
        psi = Z[:, k]
        for i in range(n_):
            box = Gamma[i] @ np.abs(psi)
            A_ub, b_ub = _row_ub(Z, Y1r[i], eps[i])
            ex = 0.0
            for sgn in (1.0, -1.0):
                r = linprog(-sgn * psi, A_ub=A_ub, b_ub=b_ub,
                            bounds=[(None, None)] * d, method="highs")
                if r.success:
                    ex = max(ex, -r.fun - sgn * (psi @ Theta_bar[i]))
            ratios.append(box / max(ex, 1e-300))
    ratios = np.array(ratios)
    out = dict(median=float(np.median(ratios)), min=float(ratios.min()),
               max=float(ratios.max()), n=int(len(ratios)))
    if verbose:
        print("\n[interval-hull conservatism]  Lemma 3 vs exact support over F")
        print(f"        box/exact over {out['n']} samples: "
              f"median {out['median']:.2f}x  "
              f"range {out['min']:.2f}-{out['max']:.2f}x")
        print("        -> this much tube inflation comes from the box")
        print("           relaxation in (14), removable without any new")
        print("           assumption or probability budget.")
    return out


def chebyshev_eps(Xn, Un):
    """min_Theta max_k |y_{k+1,a} - Theta z_k| per coordinate (one LP/row).

    Doubles as the Chebyshev floor: no data-consistent envelope can be
    smaller than this, so eps_bar / floor is the honest statement of how
    much slack the prior carries.
    """
    Ya0, Ya1, Ua0 = Xn[:-1].T, Xn[1:].T, Un.T
    Tn = Ya0.shape[1]
    Zc = np.vstack([Ya0, Ua0])
    dd = Zc.shape[0]
    A_ub = np.block([[Zc.T, -np.ones((Tn, 1))],
                     [-Zc.T, -np.ones((Tn, 1))]])
    c_lp = np.zeros(dd + 1)
    c_lp[-1] = 1.0
    out = []
    for a in range(Ya1.shape[0]):
        res_ = linprog(c_lp, A_ub=A_ub,
                       b_ub=np.concatenate([Ya1[a, :], -Ya1[a, :]]),
                       bounds=[(None, None)] * dd + [(0.0, None)],
                       method="highs")
        if not res_.success:
            raise RuntimeError(f"Chebyshev LP failed on row {a}: {res_.message}")
        out.append(res_.x[-1])
    return np.array(out)


# ============================================================================
# 1. Manipulator model
# ============================================================================
G = 9.81


def make_params(m1, m2, l1, l2, lc1f=0.5, lc2f=0.5):
    p = dict(m1=m1, m2=m2, l1=l1, l2=l2, lc1=lc1f * l1, lc2=lc2f * l2,
             I1=m1 * l1 ** 2 / 12.0, I2=m2 * l2 ** 2 / 12.0)
    p["a1"] = (p["I1"] + p["I2"] + m1 * p["lc1"] ** 2
               + m2 * (l1 ** 2 + p["lc2"] ** 2))
    p["a2"] = m2 * l1 * p["lc2"]
    p["a3"] = p["I2"] + m2 * p["lc2"] ** 2
    return p


P_TRUE = make_params(m1=2.0, m2=1.5, l1=0.5, l2=0.4)
P_NOM = make_params(m1=2.0 * 1.15, m2=1.5 * 0.85, l1=0.5, l2=0.4, lc2f=0.55)


def _check_true_in_theta():
    """The paper claims Theta contains the true mismatch.  Verify it."""
    r_m1 = P_TRUE["m1"] / P_NOM["m1"]
    r_m2 = P_TRUE["m2"] / P_NOM["m2"]
    r_lc2 = (P_TRUE["lc2"] / P_TRUE["l2"]) / (P_NOM["lc2"] / P_NOM["l2"])
    ok = (abs(r_m1 - 1) <= THETA_MASS and abs(r_m2 - 1) <= THETA_MASS
          and abs(r_lc2 - 1) <= THETA_LC2)
    print(f"[Theta] true/nominal ratios: m1={r_m1:.4f} m2={r_m2:.4f} "
          f"lc2f={r_lc2:.4f}")
    print(f"        Theta = +/-{THETA_MASS:.0%} mass, +/-{THETA_LC2:.0%} lc2"
          f"  -> P_TRUE in Theta: {'YES' if ok else 'NO'}")
    if not ok:
        raise AssertionError("P_TRUE lies outside Theta; beta is not a valid "
                             "bound for this plant.")
    return dict(r_m1=r_m1, r_m2=r_m2, r_lc2=r_lc2)


def Mmat(q2, p):
    c2 = np.cos(q2)
    return (p["a1"] + 2 * p["a2"] * c2,
            p["a3"] + p["a2"] * c2,
            p["a3"] * np.ones_like(q2))


def Cqd(q2, qd1, qd2, p):
    h = p["a2"] * np.sin(q2)
    return -h * (2 * qd1 * qd2 + qd2 ** 2), h * qd1 ** 2


def gvec(q1, q2, p):
    g1 = ((p["m1"] * p["lc1"] + p["m2"] * p["l1"]) * G * np.cos(q1)
          + p["m2"] * p["lc2"] * G * np.cos(q1 + q2))
    g2 = p["m2"] * p["lc2"] * G * np.cos(q1 + q2)
    return g1, g2


def delta_theta(q, qd, u, pt, pn):
    q1, q2 = q[:, 0], q[:, 1]
    qd1, qd2 = qd[:, 0], qd[:, 1]
    M11n, M12n, M22n = Mmat(q2, pn)
    C1n, C2n = Cqd(q2, qd1, qd2, pn)
    g1n, g2n = gvec(q1, q2, pn)
    tau1 = M11n * u[:, 0] + M12n * u[:, 1] + C1n + g1n
    tau2 = M12n * u[:, 0] + M22n * u[:, 1] + C2n + g2n
    M11, M12, M22 = Mmat(q2, pt)
    C1, C2 = Cqd(q2, qd1, qd2, pt)
    g1, g2 = gvec(q1, q2, pt)
    r1, r2 = tau1 - C1 - g1, tau2 - C2 - g2
    det = M11 * M22 - M12 ** 2
    return np.stack([(M22 * r1 - M12 * r2) / det - u[:, 0],
                     (-M12 * r1 + M11 * r2) / det - u[:, 1]], axis=1)


def Mfull(q2, p):
    M11, M12, M22 = Mmat(np.array([q2]), p)
    return np.array([[M11[0], M12[0]], [M12[0], M22[0]]])


def Cfull(q2, qd1, qd2, p):
    h = p["a2"] * np.sin(q2)
    return np.array([[-h * qd2, -h * (qd1 + qd2)], [h * qd1, 0.0]])


def gfull(q1, q2, p):
    g1, g2 = gvec(np.array([q1]), np.array([q2]), p)
    return np.array([g1[0], g2[0]])


def f_true(x, u, pt=None):
    pt = P_TRUE if pt is None else pt
    dl = delta_theta(x[:2][None], x[2:][None], u[None], pt, P_NOM)[0]
    return np.concatenate([x[2:], u + dl])


def rk4_step(x, u, dt, nsub=10, pt=None):
    h = dt / nsub
    for _ in range(nsub):
        k1 = f_true(x, u, pt)
        k2 = f_true(x + .5 * h * k1, u, pt)
        k3 = f_true(x + .5 * h * k2, u, pt)
        k4 = f_true(x + h * k3, u, pt)
        x = x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


# ============================================================================
# 2. Data collection
# ============================================================================
TS = 0.01

# -- simulated noise vs a priori bound, kept explicitly distinct.
SIGMA_TRUE = 2.0e-4       # what collect() actually injects
SIGMA_HAT  = 2.0e-4       # Assumption 3 prior used by identification/tube


EPS_P, EPS_V = SIGMA_TRUE, SIGMA_TRUE
U_MAX, Q_MAX, QD_MAX = 5.0, 1.2, 2.0
A0 = np.block([[np.eye(2), TS * np.eye(2)], [np.zeros((2, 2)), np.eye(2)]])
B0 = np.vstack([TS ** 2 / 2 * np.eye(2), TS * np.eye(2)])


def collect(T, seed, amp=0.7, pt=None):
    r = np.random.default_rng(seed)
    w = 2 * np.pi * np.array([0.23, 0.61, 1.13, 1.87])
    A = amp * np.array([0.45, 0.30, 0.17, 0.08])
    ph = r.uniform(0, 2 * np.pi, (2, 4))
    t = np.arange(T + 1) * TS
    qr = np.stack([(A * np.sin(np.outer(t, w) + ph[i])).sum(1)
                   for i in range(2)], 1)
    qdr = np.stack([(A * w * np.cos(np.outer(t, w) + ph[i])).sum(1)
                    for i in range(2)], 1)
    qddr = np.stack([(-A * w ** 2 * np.sin(np.outer(t, w) + ph[i])).sum(1)
                     for i in range(2)], 1)
    x = np.r_[qr[0], qdr[0]]
    U = np.zeros((T, 2))
    Xtrue = np.zeros((T + 1, 4))
    Xtrue[0] = x
    Vn = np.c_[r.normal(0.0, EPS_P, (T + 1, 2)),
               r.normal(0.0, EPS_V, (T + 1, 2))]
    for k in range(T):
        u = (qddr[k] + 80 * (qr[k] - x[:2]) + 18 * (qdr[k] - x[2:])
             + r.uniform(-.4, .4, 2))
        u = np.clip(u, -U_MAX, U_MAX)
        U[k] = u
        x = rk4_step(x, u, TS, pt=pt)
        Xtrue[k + 1] = x
    return Xtrue + Vn, U, Xtrue


print("=" * 74)
theta_info = _check_true_in_theta()
print("=" * 74)
if not np.isclose(SIGMA_TRUE, 1.0e-5):
    print(f"[note] SIGMA_TRUE = {SIGMA_TRUE:.1e} but Sec. V-C states a")
    print(f"       simulated covariance of (1e-5)^2 I.  With SIGMA_TRUE == "
          f"SIGMA_HAT = {SIGMA_HAT:.1e} the claimed 20x prior conservatism")
    print("       is NOT reproduced.  Reconcile script and manuscript.")
print(f"[config] ablation = {ABLATION}   enforce_lemma1 = {ENFORCE_LEMMA1}")
print("Collecting PE data (no injected disturbance)...")
X_id, U_id, Xtrue_id = collect(T=T_ID, seed=11, amp=0.7)
X_va, U_va, Xtrue_va = collect(T=T_VAL, seed=42, amp=0.6)

Y0, Y1, U0 = X_id[:-1].T, X_id[1:].T, U_id.T
n, m = Y0.shape[0], U0.shape[0]
Z_reg = np.vstack([Y0, U0])                      # regressors z_k = [y_k; u_k]

# Assumption 4 check
rank_Z = np.linalg.matrix_rank(Z_reg)
print(f"[Assumption 4] rank(Z_0) = {rank_Z} / {n + m} -> "
      f"{'SATISFIED' if rank_Z == n + m else 'VIOLATED'}")

# ============================================================================
# 3. Noise envelope and calibration of omega_hat
# ============================================================================
eta_a = norm.ppf(1.0 - DELTA_A / (2.0 * n * (T_ID + K_BAR)))

sigma_vec = np.full(n, SIGMA_TRUE)
sigma_hat = SIGMA_HAT
if ISOTROPIC:
    # Lemma 2, eq. (10):  Cov(varsigma_k) <= sigma_hat^2 (1 + kappa_hat^2) I
    sd = np.full(n, sigma_hat * np.sqrt(1.0 + KAPPA ** 2))
else:
    sd = np.sqrt(sigma_vec ** 2 + KAPPA ** 2 * sigma_hat ** 2)
noise_hw = eta_a * sd


Sigma_bar_diag = sigma_hat ** 2 * (1.0 + KAPPA ** 2) * np.ones(n)
tau_floor = eta_a * np.sqrt(Sigma_bar_diag)          # == noise_hw here
assert np.allclose(tau_floor, noise_hw), \
    "Lemma-1 floor and Lemma-2 envelope must coincide under Assumption 2"

# The old stacked bound, retained ONLY to report that it is strictly looser.
lemma1_old = sigma_hat ** 2 * (1.0 + KAPPA) ** 2
lemma1_ratio = float(np.sqrt(lemma1_old / Sigma_bar_diag[0]))

if not ISOTROPIC:
    print("[warn] anisotropic noise: Assumption 2 as stated is violated; "
          "Lemma 2 still holds via sigma_hat = max_i sigma_i.")

status = ("[MATCHED to beta]" if MATCHED_THETA
          else "[MISMATCHED -- 'sensitivity only']")
print(f"Corner-ensemble calibration over +/-{THETA_MASS_CAL:.0%} mass, "
      f"+/-{THETA_LC2_CAL:.0%} lc2 {status}...")


def theta_corners(th_m, th_l):
    return [make_params(P_NOM["m1"] * f1, P_NOM["m2"] * f2, 0.5, 0.4,
                        lc2f=0.55 * f3)
            for f1 in (1 - th_m, 1 + th_m)
            for f2 in (1 - th_m, 1 + th_m)
            for f3 in (1 - th_l, 1 + th_l)]


cal_corners = theta_corners(THETA_MASS_CAL, THETA_LC2_CAL)
eps_min_cal = np.zeros(n)
eps_per_corner = []
for pt_c in cal_corners:
    X_c, U_c, _ = collect(T=T_CAL, seed=CAL_SEED, amp=0.7, pt=pt_c)
    e_c = chebyshev_eps(X_c, U_c)
    eps_per_corner.append(e_c)
    eps_min_cal = np.maximum(eps_min_cal, e_c)
eps_per_corner = np.array(eps_per_corner)

eps_floor_id = chebyshev_eps(X_id, U_id)

omega_bar = np.maximum(SAFETY * eps_min_cal, OMEGA_FLOOR)
eps_bar = omega_bar + noise_hw
g = GEN_SCALE * omega_bar
G_omega = np.diag(g)
headroom_floor = eps_bar / np.maximum(eps_floor_id, 1e-300)

print(f"[envelope] eta_a={eta_a:.4f}  isotropic={ISOTROPIC}  "
      f"sigma_hat={sigma_hat:.3e}  SAFETY={SAFETY}")
print(f"           noise hw (Lemma 2)      = {noise_hw}")
print(f"           Lemma-1 floor on tau    = {tau_floor}")
print(f"           old stacked (1+k)^2 bnd = {lemma1_ratio:.3f}x looser "
      f"-> not used")
print(f"           eps_min (calib, T={T_CAL:4d}) = {eps_min_cal}")
print(f"           eps_min corner spread   = "
      f"{eps_per_corner.max(0) / np.maximum(eps_per_corner.min(0), 1e-300)}")
print(f"           Chebyshev floor (ident) = {eps_floor_id}")
print(f"           headroom eps_bar/floor  = {headroom_floor}")
print(f"           omega_hat (calibrated)  = {omega_bar}")
print(f"           eps_bar                 = {eps_bar}")

# ============================================================================
# 4. Theorem-1 identification program (region route)
#    tau_i := eta_a sqrt(e_i^T Sigma_kappa e_i);  SOCP/QP, not an SDP.
# ============================================================================
generators = []
for i in range(n):
    for j in range(n):
        GA = np.zeros((n, n))
        GA[i, j] = 1.0
        generators.append((GA, np.zeros((n, m))))
for i in range(n):
    for j in range(m):
        GB = np.zeros((n, m))
        GB[i, j] = 1.0
        generators.append((np.zeros((n, n)), GB))
q = len(generators)

V = np.zeros((T_ID, n, q))
for k in range(T_ID):
    for j, (GA, GB) in enumerate(generators):
        V[k, :, j] = GA @ Y0[:, k] + GB @ U0[:, k]

if ABLATION == "prior_only":
    print("\n[ABLATION] prior_only: skipping (11) entirely.")
    print("           h_wtilde := eps_bar (Lemma-2 envelope), c := 0,")
    print("           predictor := nominal (A0, B0).")
    A_hat, B_hat = A0.copy(), B0.copy()
    s_star = np.zeros(q)
    c_om = np.zeros(n)
    g_out = g.copy()
    D_kappa = noise_hw.copy()
    solve_status = "skipped"
else:
    print("\nConfiguring identification program (Theorem 1, region route)...")
    SB = sigma_hat                              # nondimensionalisation scale
    C_A = cp.Variable((n, n))
    C_B = cp.Variable((n, m))
    s_h = cp.Variable(q, nonneg=True)           # s = SB * s_h
    tau_h = cp.Variable(n, nonneg=True)         # tau = SB * tau_h
    c_h = cp.Variable(n)                        # c_omega = SB * c_h

    cons = []
    # --- Lemma 1 (per-sample covariance bound), imposed:
    if ENFORCE_LEMMA1:
        cons.append(tau_h >= tau_floor / SB)

    if OPTIMIZE_G:
        g_h = cp.Variable(n, nonneg=True)
        zeta = cp.Variable((n, T_ID))
        cons += [cp.abs(zeta[i, :]) <= g_h[i] for i in range(n)]
        bounded_term = zeta
    else:
        lambdas = cp.Variable((n, T_ID))
        cons += [lambdas <= 1, lambdas >= -1]
        bounded_term = cp.diag(g / SB) @ lambdas

    # (11c)+(11d): residual consistency
    for k in range(T_ID):
        r_k = (Y1[:, k] - C_A @ Y0[:, k] - C_B @ U0[:, k]) / SB
        for i in range(n):
            cons.append(cp.abs(r_k[i] - c_h[i] - bounded_term[i, k])
                        <= tau_h[i] - (s_h @ np.abs(V[k, i, :])))

    # (11f): coverage of the Lemma-2 envelope
    if OPTIMIZE_G:
        cons.append(g_h + tau_h >= eps_bar / SB + cp.abs(c_h))
        obj = (cp.sum(g_h) + LAMBDA_SIGMA * cp.sum_squares(tau_h) / eta_a ** 2
               + cp.sum(s_h))
        if LAMBDA_REG:
            obj = obj + cp.sum(cp.abs(zeta)) / T_ID
    else:
        cons.append(tau_h >= eps_bar / SB + cp.abs(c_h) - np.abs(g) / SB)
        obj = (cp.sum(s_h)
               + LAMBDA_SIGMA * cp.sum_squares(tau_h) / eta_a ** 2)
        if LAMBDA_REG:
            obj = obj + cp.sum(cp.abs(lambdas)) / T_ID

    print("Solving (SOCP/QP -- no LMI; see revision note 2)...")
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver=cp.CLARABEL, verbose=False)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Theorem-1 program failed: status={prob.status}")
    if prob.status == "optimal_inaccurate":
        print("[warn] returned optimal_inaccurate")
    solve_status = prob.status

    A_hat, B_hat = C_A.value, C_B.value
    s_star = np.maximum(s_h.value, 0.0) * SB
    c_om = c_h.value * SB
    g_out = (np.maximum(g_h.value, 0.0) * SB) if OPTIMIZE_G else g
    D_kappa = tau_h.value * SB
    print(f"-> active s_i={int((s_star > 1e-9).sum())}/{q}, "
          f"sum(s*)={s_star.sum():.3e}")
    # Sigma*_kappa recovered from tau*
    Sigma_star_diag = (D_kappa / eta_a) ** 2
    at_floor = np.allclose(Sigma_star_diag, Sigma_bar_diag, rtol=2e-2)
    print("-> Sigma*_kappa diag = "
          f"{np.array2string(Sigma_star_diag, precision=3, suppress_small=False)}")
    print("   Lemma-1 floor     = "
          f"{np.array2string(Sigma_bar_diag, precision=3, suppress_small=False)}")
    print(f"   at the floor?      {'YES' if at_floor else 'no'}"
          + ("   (=> Sigma_kappa is data, not a decision variable;"
             " state it as such)" if at_floor else ""))

hw_dd = np.abs(g_out) + D_kappa
c_dd = c_om.copy()


def residuals(X, U, A, B):
    return X[1:] - X[:-1] @ A.T - U @ B.T


# --------------------------------------------------- which constraint binds --
R_id = Y1 - A_hat @ Y0 - B_hat @ U0                    # n x T
cov_hw = eps_bar + np.abs(c_om)                        # (11f) route
data_hw = np.abs(R_id - c_om[:, None]).max(axis=1)     # (11d) route

routes = np.vstack([cov_hw, data_hw])
which = [["coverage(11f)", "residual(11d)"][j]
         for j in np.argmax(routes, axis=0)]
data_margin = data_hw / np.maximum(cov_hw, 1e-300) - 1.0

print("\n[binding constraint, per coordinate]")
print(f"        achieved h_wtilde        = {hw_dd}")
print(f"        coverage (11f) route     = {cov_hw}")
print(f"        residual (11d) route     = {data_hw}")
print(f"        active                   = {which}")
print(f"        data contribution        = {data_margin}"
      f"   (<= 0 everywhere => identification is inert)")

audit = audit_theorem1(g=g_out, hw=hw_dd, c_om=c_om, eps_bar=eps_bar,
                       R_id=R_id, eta_a=eta_a, s_star=s_star,
                       sigma_bar_hw=tau_floor)
audit["ablation"] = ABLATION
audit["solve_status"] = solve_status
audit["headroom_floor"] = headroom_floor.tolist()
audit["lemma1_old_over_new"] = lemma1_ratio

# --- regression guard: if the binding route ever changes, the prose in
#     Section V-B and Table 1 must be revisited.
EXPECTED_INERT = True
if audit["sdp_inert"] != EXPECTED_INERT:
    print("\n[REGRESSION] binding route changed vs EXPECTED_INERT="
          f"{EXPECTED_INERT}; Section V-B text and Table 1 need revisiting.")

# ============================================================================
# 5. Contraction metric P
# ============================================================================
print("\nSolving SDP for the contraction metric P...")
thetas = theta_corners(THETA_MASS, THETA_LC2)
Ns = 4000
qs = np.c_[rng.uniform(-Q_MAX, Q_MAX, Ns), rng.uniform(-Q_MAX, Q_MAX, Ns)]
qds = np.c_[rng.uniform(-QD_MAX, QD_MAX, Ns), rng.uniform(-QD_MAX, QD_MAX, Ns)]
us = rng.uniform(-U_MAX, U_MAX, (Ns, 2))
dmax = np.zeros(2)
for th in thetas:
    dmax = np.maximum(dmax, np.abs(delta_theta(qs, qds, us, th, P_NOM)).max(0))
wc_x = np.abs(B0) @ dmax
verts_w = np.array(list(itertools.product([-1, 1], repeat=4))) * wc_x
A_xc = np.vstack([np.c_[np.eye(2), np.zeros((2, 2))],
                  -np.c_[np.eye(2), np.zeros((2, 2))],
                  np.c_[np.zeros((2, 2)), np.eye(2)],
                  -np.c_[np.zeros((2, 2)), np.eye(2)]])
A_uc = np.vstack([np.eye(2), -np.eye(2)])


def solve_P(rho):
    E = cp.Variable((n, n), PSD=True)
    Y = cp.Variable((m, n))
    cx = cp.Variable(A_xc.shape[0])
    cu = cp.Variable(A_uc.shape[0])
    wb = cp.Variable()
    o = (1 / (2 * (1 - rho))
         * ((A_xc.shape[0] + A_uc.shape[0]) * wb + cx.sum() / 0.1
            + cu.sum() / U_MAX))
    cn = [E >> np.eye(n),
          cp.bmat([[rho ** 2 * E, (A0 @ E + B0 @ Y).T],
                   [(A0 @ E + B0 @ Y), E]]) >> 0,
          cx >= 0, cu >= 0, wb >= 0]
    for i in range(A_xc.shape[0]):
        cn += [cp.bmat([[cx[i, None, None], A_xc[i, None] @ E],
                        [(A_xc[i, None] @ E).T, E]]) >> 0]
    for i in range(A_uc.shape[0]):
        cn += [cp.bmat([[cu[i, None, None], A_uc[i, None] @ Y],
                        [(A_uc[i, None] @ Y).T, E]]) >> 0]
    for i in range(verts_w.shape[0]):
        cn += [cp.bmat([[wb[None, None], verts_w[i][None]],
                        [verts_w[i][:, None], E]]) >> 0]
    cp.Problem(cp.Minimize(o), cn).solve(solver=cp.CLARABEL)
    if E.value is None:
        return None, None, None
    return np.linalg.inv(E.value), Y.value @ np.linalg.inv(E.value), rho


P = K = rho_used = None
for rho in np.linspace(0.8, 0.97, 12):
    P, K, rho_used = solve_P(rho)
    if P is not None:
        break
if P is None:
    raise RuntimeError("Contraction-metric SDP infeasible for every rho.")
print(f"   contraction rate rho = {rho_used:.4f}")

Psqrt = np.real(sqrtm(P))
PB = Psqrt @ B0


def pnorm(W):
    return np.sqrt(np.einsum('ki,ij,kj->k', W, P, W))


def prad(center, halfw):
    corners = np.array(list(itertools.product([-1, 1], repeat=4))) * halfw \
        + center
    return pnorm(corners).max()


# ============================================================================
# 6. Analytical bound beta
# ============================================================================
print("Recomputing analytical constants a,b,c...")
qs2 = np.c_[rng.uniform(-Q_MAX, Q_MAX, 1200), rng.uniform(-Q_MAX, Q_MAX, 1200)]
qds2 = np.c_[rng.uniform(-QD_MAX, QD_MAX, 1200),
             rng.uniform(-QD_MAX, QD_MAX, 1200)]
a = b = c = 0.0
for th in thetas:
    for (q1, q2), (qd1, qd2) in zip(qs2, qds2):
        M = Mfull(q2, th)
        Minv = np.linalg.inv(M)
        a = max(a, np.linalg.norm(PB @ (Minv @ (M - Mfull(q2, P_NOM))), 2))
        b = max(b, np.linalg.norm(
            PB @ (Minv @ (Cfull(q2, qd1, qd2, th)
                          - Cfull(q2, qd1, qd2, P_NOM))), 2))
        gtil = Minv @ (gfull(q1, q2, th) - gfull(q1, q2, P_NOM))
        acc = rng.uniform(-U_MAX, U_MAX, 2)
        x = np.r_[q1, q2, qd1, qd2]
        xode = solve_ivp(
            lambda t, xt, u: np.r_[
                xt[2:], u + delta_theta(xt[:2][None], xt[2:][None], u[None],
                                        th, P_NOM)[0]],
            [0, TS], x, args=(acc,)).y[:, -1]
        dth = delta_theta(x[:2][None], x[2:][None], acc[None], th, P_NOM)[0]
        c = max(c, np.linalg.norm(
            PB @ gtil + Psqrt @ (xode - A0 @ x - B0 @ (acc + dth)), 2))
print(f"   a={a:.4e} b={b:.4e} c={c:.4e}")

# ============================================================================
# 7. Comparisons
# ============================================================================
W_id = residuals(X_id, U_id, A_hat, B_hat)
qd_true = Xtrue_id[:-1, 2:]        # beta is stated at EXACT states
beta = (a * np.linalg.norm(U_id, axis=1)
        + b * np.linalg.norm(qd_true, axis=1) + c)
wP = pnorm(W_id)

D_true = Xtrue_id[1:] - Xtrue_id[:-1] @ A0.T - U_id @ B0.T
dP_true = pnorm(D_true)
disc_hw = np.abs(D_true).max(axis=0)
rad_disc = prad(np.zeros(n), disc_hw)

rad_omega = prad(np.zeros(n), omega_bar)
rad_dd_P = prad(c_dd, hw_dd)
rad_noise = prad(np.zeros(n), D_kappa)
rad_data = prad(c_om, data_hw)          # (11d) counterfactual

ratio_I_max = beta.max() / rad_omega
ratio_I_med = np.median(beta) / rad_omega
ratio_II_max = beta.max() / rad_dd_P
ratio_II_med = np.median(beta) / rad_dd_P

beta_aug_max = beta.max() + rad_noise
beta_aug_med = np.median(beta) + rad_noise
ratio_III_max = beta_aug_max / rad_dd_P
ratio_III_med = beta_aug_med / rad_dd_P
beta_aug = beta + rad_noise

W_va = residuals(X_va, U_va, A_hat, B_hat)
cov_dd = 100 * np.all(np.abs(W_va - c_dd) <= hw_dd + 1e-12, axis=1).mean()
cov_data = 100 * np.all(np.abs(W_va - c_om) <= data_hw + 1e-12, axis=1).mean()

noise_content = np.minimum(noise_hw / np.maximum(omega_bar, 1e-300), 1.0)
disc_share = np.minimum(disc_hw / np.maximum(omega_bar, 1e-300), 1.0)
cov_beta_true = 100 * np.mean(dP_true <= beta)

# --- data-feasible set F and its interval hull (14), plus the support gap ---
print("\nComputing the data-feasible set F and its interval hull (14)...")
Theta_bar_F, Gamma_F = feasible_hull(Y1, Z_reg, eps_bar)
Theta_star_true = np.hstack([A0, B0])
in_hull = bool(np.all(np.abs(Theta_star_true - Theta_bar_F)
                      <= Gamma_F + 1e-12))
print(f"   Gamma max entry            = {Gamma_F.max():.4e}")
print(f"   |Theta* - Theta_bar| max   = "
      f"{np.abs(Theta_star_true - Theta_bar_F).max():.4e}")
print(f"   true parameters inside hull: {'YES' if in_hull else 'NO'}")

sgap = None
if RUN_SUPPORT_GAP:
    sgap = support_gap(Y1, Z_reg, eps_bar, Theta_bar_F, Gamma_F,
                       n_samples=SUPPORT_SAMPLES)

lines = [
    "=== Table 1 (region route): comparisons in the P-weighted norm ===",
    f"ablation = {ABLATION}   enforce_lemma1 = {ENFORCE_LEMMA1}",
    f"eta_a = {eta_a:.4f}   isotropic = {ISOTROPIC}   kappa_hat = {KAPPA}"
    f"   optimise_g = {OPTIMIZE_G}   SAFETY = {SAFETY}",
    f"sigma_true = {SIGMA_TRUE:.2e}   sigma_hat = {SIGMA_HAT:.2e}",
    f"Theta (beta)  = +/-{THETA_MASS:.0%} mass, +/-{THETA_LC2:.0%} lc2",
    f"Theta (prior) = +/-{THETA_MASS_CAL:.0%} mass, +/-{THETA_LC2_CAL:.0%} lc2"
    f"   {'[MATCHED]' if MATCHED_THETA else '[MISMATCHED]'}",
    f"analytical constants  a={a:.4e}  b={b:.4e}  c={c:.4e}",
    "",
    "(0) LEMMA 1 (per-sample covariance bound), imposed on (11)",
    f"    Sigma_bar diag              = "
    f"{np.array2string(Sigma_bar_diag, precision=3, suppress_small=False)}",
    f"    tau floor = eta_a sqrt(.)   = {tau_floor}",
    f"    old stacked (1+k)^2 route   = {lemma1_ratio:.3f}x looser, unused",
    "",
    "(I) DISCREPANCY ONLY",
    f"    omega_hat (calibrated)      = {omega_bar}",
    f"    realised discrepancy box    = {disc_hw}",
    f"    noise content of omega_hat  = {noise_content}",
    f"    realised-disc share         = {disc_share}",
    f"    P-radius omega_hat          = {rad_omega:.4e}",
    f"    P-radius realised disc.     = {rad_disc:.4e}",
    f"    P-radius beta_max           = {beta.max():.4e}",
    f"    ratio vs omega_hat          : max {ratio_I_max:.2f}x , "
    f"median {ratio_I_med:.2f}x",
    f"    ratio vs realised disc.     : max {beta.max() / rad_disc:.2f}x , "
    f"median {np.median(beta) / rad_disc:.2f}x",
    "",
    "(II) FULL EFFECTIVE UNCERTAINTY",
    f"    h_wtilde = |g| + tau        = {hw_dd}",
    f"      of which measurement noise = {D_kappa}",
    f"      eps_bar (Lemma 2 envelope) = {eps_bar}",
    f"    P-radius Z_wtilde           = {rad_dd_P:.4e}",
    f"      of which noise alone      = {rad_noise:.4e}  "
    f"({100 * rad_noise / rad_dd_P:.0f}% of the set)",
    f"    ratio                       : max {ratio_II_max:.2f}x , "
    f"median {ratio_II_med:.2f}x",
    "",
    "(III) LIKE-FOR-LIKE -- both descriptions cover discrepancy + noise",
    f"      noise envelope P-radius   = {rad_noise:.4e}",
    f"      beta (+) N : max {beta_aug_max:.4e}   med {beta_aug_med:.4e}",
    f"      Z_wtilde                  = {rad_dd_P:.4e}",
    f"      conservatism ratio        : max {ratio_III_max:.2f}x , "
    f"median {ratio_III_med:.2f}x",
    "",
    "(IV) COUNTERFACTUAL -- what the residuals alone would support",
    f"      (11d) half-widths           = {data_hw}",
    f"      P-radius of (11d) set       = {rad_data:.4e}",
    f"      tightening vs Z_wtilde      = {rad_dd_P / rad_data:.2f}x",
    f"      validation coverage of (11d) set = {cov_data:.1f}%  "
    f"(target {100 * (1 - DELTA_A):.0f}%)",
    f"      binding route per coord     = {which}",
    f"      Chebyshev floor             = {eps_floor_id}",
    f"      headroom eps_bar / floor    = {headroom_floor}"
    f"   ({headroom_floor.min():.1f}x to {headroom_floor.max():.1f}x)",
    "",
    "(V) INTERVAL-HULL CONSERVATISM IN (14)  [T3]",
    f"      Gamma max entry             = {Gamma_F.max():.4e}",
    f"      true params inside hull     = {in_hull}",
]
if sgap is not None:
    lines += [f"      box/exact support ratio     : median "
              f"{sgap['median']:.2f}x  range {sgap['min']:.2f}-"
              f"{sgap['max']:.2f}x"]
lines += [
    "",
    f"beta(x,u) along traj : med={np.median(beta):.4e}  max={beta.max():.4e}",
    f"||w_tilde||_P        : med={np.median(wP):.4e}  max={wP.max():.4e}",
    f"||d_k||_P (true)     : med={np.median(dP_true):.4e}  "
    f"max={dP_true.max():.4e}",
    f"coverage ||d_k||_P <= beta      : {cov_beta_true:.1f}%",
    f"Z_wtilde validation coverage    : {cov_dd:.1f}%  "
    f"(target {100 * (1 - DELTA_A):.0f}%)",
    "",
    "M1 VERDICT: " + ("identification INERT -- (11f) binds everywhere; "
                      "restate Contribution 1"
                      if audit["sdp_inert"] else
                      "(11d) binds somewhere -- data does constrain"),
]
summary = "\n".join(lines)
print("\n" + summary)

# ============================================================================
# 7b. Assumption 3 on the identification record
# ============================================================================
print("\n[Assumption 3] identification record")
print(f"        max |d_k| per coordinate = {disc_hw}")
print(f"        omega_hat                = {omega_bar}")
cov_box_id = 100 * np.all(np.abs(D_true) <= omega_bar + 1e-15, axis=1).mean()
print(f"        coverage |d_k| <= omega_hat : {cov_box_id:.1f}%")
print("        -> " + ("SATISFIED" if cov_box_id == 100.0 else "VIOLATED"))

# ============================================================================
# 7c. Assumption 3 over Theta and over wider trajectories
# ============================================================================
a3 = {}
if VERIFY_A3:
    print(f"\n[Assumption 3a] Theta vertices ({len(thetas)} corners "
          f"x {A3_T} steps)")
    worst_vert = np.zeros(n)
    for ci, pt_c in enumerate(thetas):
        _, U_r, Xt_r = collect(T=A3_T, seed=9000 + ci, amp=0.7, pt=pt_c)
        D_r = Xt_r[1:] - Xt_r[:-1] @ A0.T - U_r @ B0.T
        worst_vert = np.maximum(worst_vert, np.abs(D_r).max(axis=0))
    ok_vert = bool(np.all(worst_vert <= omega_bar))
    print(f"        worst |d_k| at vertices = {worst_vert}")
    print(f"        required inflation      = "
          f"{worst_vert / np.maximum(omega_bar, 1e-300)}")
    print("        -> " + ("SATISFIED" if ok_vert else "VIOLATED"))

    print(f"\n[Assumption 3b] Theta interior ({A3_INTERIOR} draws x {A3_T})")
    worst_int = np.zeros(n)
    for r_i in range(A3_INTERIOR):
        rr = np.random.default_rng(9500 + r_i)
        pt_r = make_params(
            m1=P_NOM["m1"] * (1 + rr.uniform(-THETA_MASS, THETA_MASS)),
            m2=P_NOM["m2"] * (1 + rr.uniform(-THETA_MASS, THETA_MASS)),
            l1=0.5, l2=0.4,
            lc2f=0.55 * (1 + rr.uniform(-THETA_LC2, THETA_LC2)))
        _, U_r, Xt_r = collect(T=A3_T, seed=9500 + r_i, amp=0.7, pt=pt_r)
        D_r = Xt_r[1:] - Xt_r[:-1] @ A0.T - U_r @ B0.T
        worst_int = np.maximum(worst_int, np.abs(D_r).max(axis=0))
    ok_int = bool(np.all(worst_int <= omega_bar))
    print(f"        worst |d_k| interior    = {worst_int}")
    print(f"        required inflation      = "
          f"{worst_int / np.maximum(omega_bar, 1e-300)}")
    print("        -> " + ("SATISFIED" if ok_int else "VIOLATED"))

    print(f"\n[Assumption 3c] out-of-distribution trajectories "
          f"(amp={A3_AMP_OOD}, true plant)")
    worst_traj = np.zeros(n)
    for r_i in range(8):
        _, U_r, Xt_r = collect(T=A3_T, seed=7000 + r_i, amp=A3_AMP_OOD,
                               pt=P_TRUE)
        D_r = Xt_r[1:] - Xt_r[:-1] @ A0.T - U_r @ B0.T
        worst_traj = np.maximum(worst_traj, np.abs(D_r).max(axis=0))
    ok_traj = bool(np.all(worst_traj <= omega_bar))
    print(f"        worst |d_k| wider traj  = {worst_traj}")
    print(f"        required inflation      = "
          f"{worst_traj / np.maximum(omega_bar, 1e-300)}")
    print("        -> " + ("SATISFIED" if ok_traj else "VIOLATED"))

    ok_all = ok_vert and ok_int and ok_traj
    if not ok_all:
        print("\n        E_env does NOT hold for all reported conditions;")
        print("        Theorem 1, Corollary 1, Theorems 2-4 do not apply.")
    a3 = dict(vertices=worst_vert.tolist(), interior=worst_int.tolist(),
              trajectory=worst_traj.tolist(), ok=ok_all)
    summary += ("\n\nAssumption 3: " + ("SATISFIED" if ok_all else "VIOLATED")
                + f"\n  worst |d_k| vertices = {worst_vert}"
                + f"\n  worst |d_k| interior = {worst_int}"
                + f"\n  worst |d_k| traj     = {worst_traj}"
                + f"\n  omega_hat            = {omega_bar}")

with open(OUT_DIR / f"summary_{ABLATION}.txt", "w") as f:
    f.write(summary + "\n")

# ============================================================================
# 8. Machine-readable output + LaTeX macros
# ============================================================================
res = dict(
    config=dict(eta_a=float(eta_a), kappa=KAPPA, delta_a=DELTA_A,
                safety=SAFETY, isotropic=ISOTROPIC, optimize_g=OPTIMIZE_G,
                lambda_reg=LAMBDA_REG, lambda_sigma=LAMBDA_SIGMA,
                T_id=T_ID, T_cal=T_CAL, matched_theta=MATCHED_THETA,
                theta_mass=THETA_MASS, theta_lc2=THETA_LC2,
                theta_mass_cal=THETA_MASS_CAL, theta_lc2_cal=THETA_LC2_CAL,
                rho=float(rho_used), sigma_hat=float(sigma_hat),
                sigma_true=float(SIGMA_TRUE), ablation=ABLATION,
                enforce_lemma1=ENFORCE_LEMMA1),
    lemma1=dict(Sigma_bar_diag=Sigma_bar_diag.tolist(),
                tau_floor=tau_floor.tolist(),
                old_stacked_ratio=lemma1_ratio),
    audit=audit,
    support_gap=sgap,
    theta_check=theta_info,
    abc=dict(a=float(a), b=float(b), c=float(c)),
    omega_hat=omega_bar.tolist(),
    eps_min_cal=eps_min_cal.tolist(), eps_floor_id=eps_floor_id.tolist(),
    headroom_floor=headroom_floor.tolist(),
    h_wtilde=hw_dd.tolist(), data_hw=data_hw.tolist(), cov_hw=cov_hw.tolist(),
    binding_route=which, data_margin=data_margin.tolist(),
    realised_disc_box=disc_hw.tolist(),
    noise_content=noise_content.tolist(),
    gamma_max=float(Gamma_F.max()), true_in_hull=in_hull,
    radii=dict(omega=float(rad_omega), Z=float(rad_dd_P),
               noise=float(rad_noise), data=float(rad_data),
               disc=float(rad_disc), beta_max=float(beta.max()),
               beta_med=float(np.median(beta)),
               beta_aug_max=float(beta_aug_max),
               beta_aug_med=float(beta_aug_med)),
    ratios=dict(I_max=float(ratio_I_max), I_med=float(ratio_I_med),
                II_max=float(ratio_II_max), II_med=float(ratio_II_med),
                III_max=float(ratio_III_max), III_med=float(ratio_III_med),
                data_tightening=float(rad_dd_P / rad_data)),
    coverage=dict(Z_validation=float(cov_dd), data_validation=float(cov_data),
                  beta_true=float(cov_beta_true), box_id=float(cov_box_id)),
    realised=dict(w_med=float(np.median(wP)), w_max=float(wP.max()),
                  d_med=float(np.median(dP_true)), d_max=float(dP_true.max())),
    assumption3=a3,
)
with open(OUT_DIR / f"table1_numbers_{ABLATION}.json", "w") as f:
    json.dump(res, f, indent=2)


def _sci(x, d=2):
    """LaTeX scientific notation, e.g. 2.04{\\times}10^{-2}."""
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    mant = x / 10 ** e
    return f"{mant:.{d}f}{{\\times}}10^{{{e}}}"


macros = {
    "TabRadOmega": _sci(rad_omega), "TabRadZ": _sci(rad_dd_P),
    "TabRadNoise": _sci(rad_noise), "TabRadData": _sci(rad_data),
    "TabRadDisc": _sci(rad_disc),
    "TabBetaMax": _sci(beta.max()), "TabBetaMed": _sci(np.median(beta)),
    "TabBetaAugMax": _sci(beta_aug_max), "TabBetaAugMed": _sci(beta_aug_med),
    "TabRatioIMax": f"{ratio_I_max:.2f}", "TabRatioIMed": f"{ratio_I_med:.2f}",
    "TabRatioIIIMax": f"{ratio_III_max:.2f}",
    "TabRatioIIIMed": f"{ratio_III_med:.2f}",
    "TabDataTighten": f"{rad_dd_P / rad_data:.1f}",
    "TabNoiseShare": f"{100 * rad_noise / rad_dd_P:.0f}",
    "TabWMax": _sci(wP.max()), "TabWMed": _sci(np.median(wP)),
    "TabDMax": _sci(dP_true.max()), "TabDMed": _sci(np.median(dP_true)),
    "TabCovZ": f"{cov_dd:.0f}", "TabCovData": f"{cov_data:.0f}",
    "TabEtaA": f"{eta_a:.3f}",
    "TabConstA": _sci(a), "TabConstB": _sci(b), "TabConstC": _sci(c),

    "TabMarginMin": f"{audit['headroom_min']:.1f}",
    "TabMarginMax": f"{audit['headroom_max']:.1f}",

    "TabFloorHeadroomMin": f"{headroom_floor.min():.1f}",
    "TabFloorHeadroomMax": f"{headroom_floor.max():.1f}",
    "TabLemmaOneRatio": f"{lemma1_ratio:.2f}",
    "TabTauFloor": _sci(float(tau_floor[0])),
    "TabIdentInert": "yes" if audit["sdp_inert"] else "no",
}
if sgap is not None:
    macros["TabSupportGapMed"] = f"{sgap['median']:.1f}"
    macros["TabSupportGapMin"] = f"{sgap['min']:.1f}"
    macros["TabSupportGapMax"] = f"{sgap['max']:.1f}"

with open(OUT_DIR / f"table1_macros_{ABLATION}.tex", "w") as f:
    f.write("% auto-generated -- do not edit; regenerate with this script\n")
    for k, v in macros.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

# ============================================================================
# 9. Figure
# ============================================================================
TWO_COLUMN = False
FIG_W = 7.16 if TWO_COLUMN else 3.5
FIG_H = 3.20 if TWO_COLUMN else 2.65
FS = 9 if TWO_COLUMN else 8

plt.rcParams.update({
    "font.size": FS, "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
    "axes.linewidth": 0.6,
    "xtick.labelsize": FS - 1, "ytick.labelsize": FS - 1,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "figure.dpi": 400, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
C_DD, C_ANL, C_AUG, C_OM, C_DAT = ("#1f77b4", "#d62728", "#ff7f0e",
                                   "#2ca02c", "#9467bd")

fig, axT = plt.subplots(figsize=(FIG_W, FIG_H))
kk = np.arange(len(wP))

axT.plot(kk, wP, color="0.60", lw=0.5, alpha=0.85, zorder=1,
         label=r"$\Vert\tilde w_k\Vert_P$")
axT.axhline(rad_omega, color=C_OM, lw=2.0, ls=":", zorder=5,
            label=r"$\hat\omega$ radius")
axT.plot(kk, beta, color=C_ANL, lw=1.3, ls="--", zorder=4,
         label=r"$\beta(x_k,u_k)$")
axT.axhline(rad_dd_P, color=C_DD, lw=1.8, zorder=4,
            label=r"$\mathcal{Z}_{\tilde w}$ radius")
axT.plot(kk, beta_aug, color=C_AUG, lw=1.5, ls="-.", zorder=5,
         label=r"$\beta\oplus\mathcal{N}$")
axT.axhline(rad_data, color=C_DAT, lw=1.2, ls=(0, (1, 1)), zorder=3,
            label=r"residual route")


def _gap(x_frac, r1, r2, txt):
    """Annotate the gap between two levels; orientation-safe."""
    lo, hi = min(r1, r2), max(r1, r2)
    if hi / max(lo, 1e-300) < 1.03:      # too close to annotate legibly
        return
    x = int(x_frac * len(kk))
    axT.annotate("", xy=(x, hi), xytext=(x, lo),
                 arrowprops=dict(arrowstyle="<->", color="0.25", lw=0.8,
                                 shrinkA=0, shrinkB=0), zorder=6)
    axT.text(x - 0.015 * len(kk), np.sqrt(lo * hi), txt, fontsize=FS - 1.5,
             color="0.25", ha="right", va="center", zorder=6,
             bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none",
                       alpha=0.8))


_gap(0.60, rad_omega, np.median(beta), rf"${ratio_I_med:.2f}\times$")
_gap(0.90, rad_dd_P, beta_aug_med, rf"${ratio_III_med:.2f}\times$")
_gap(0.30, rad_data, rad_dd_P, rf"${rad_dd_P / rad_data:.1f}\times$")

axT.set_yscale("log")
axT.set_ylabel(r"$P$-weighted magnitude", labelpad=2)
axT.set_xlabel(r"time step $k$", labelpad=2)
axT.set_ylim(min(wP.min() * 0.22, rad_data * 0.5), beta_aug_max * 3.0)
axT.margins(x=0.01)
axT.legend(loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=3,
           frameon=False, fontsize=FS - 1.5, handlelength=1.4,
           handletextpad=0.35, columnspacing=0.9, borderaxespad=0.3,
           labelspacing=0.25)

for ext in ("pdf", "png"):
    fig.savefig(OUT_DIR / f"fig1_residual_sets_{ABLATION}.{ext}")

print(f"\nWritten to {OUT_DIR} (ablation={ABLATION}):")
print(f"  fig1_residual_sets_{ABLATION}.pdf/.png")
print(f"  summary_{ABLATION}.txt")
print(f"  table1_numbers_{ABLATION}.json   (machine-readable)")
print(f"  table1_macros_{ABLATION}.tex     (\\input this so the paper "
      f"cannot drift)")
print("\nTo run the ablation:  THM1_ABLATION=prior_only python "
      "9_compare_wtilde_vs_delta.py")