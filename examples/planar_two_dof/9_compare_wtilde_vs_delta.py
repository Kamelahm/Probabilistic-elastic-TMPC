"""
Numerical comparison for Table 1 / Fig. 1, REGION ROUTE.

Two comparisons are reported, because they answer different questions:

  (I)  DISCREPANCY ONLY -- apples to apples.
       beta(x,u) of Prop. 1 in Wullt et al. bounds the model discrepancy
       B*Delta_theta + Delta_disc using EXACT states.  The corresponding
       quantity here is the calibrated process-disturbance bound omega_bar,
       which by Section V-B is that same discrepancy.  Both are then
       compared in the same P-weighted norm.

  (II) FULL EFFECTIVE UNCERTAINTY -- what the controller actually carries.
       Z_wtilde must additionally cover the measurement-noise propagation
       upsilon_{k+1} - A* upsilon_k at the STATED confidence eta_a (~4.5 sigma),
       so it is necessarily larger than beta, which omits noise entirely.

"""
import os, itertools
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

# ---------------------------------------------------------------- switches --
ISOTROPIC   = True      # [6] True = paper's Assumption 2 verbatim
KAPPA       = 1.01       # [1] a priori spectral bound, FIXED
DELTA_A     = 0.05
K_BAR       = 500        # [3] closed-loop horizon entering eta_a
SAFETY      = 2.0        # [5] margin on the calibrated omega_bar
OMEGA_FLOOR = 1e-8

# ======================================================================
# 1. Manipulator model (unchanged)
# ======================================================================
G = 9.81
def make_params(m1, m2, l1, l2, lc1f=0.5, lc2f=0.5):
    p = dict(m1=m1, m2=m2, l1=l1, l2=l2, lc1=lc1f*l1, lc2=lc2f*l2,
             I1=m1*l1**2/12.0, I2=m2*l2**2/12.0)
    p["a1"] = p["I1"]+p["I2"]+m1*p["lc1"]**2+m2*(l1**2+p["lc2"]**2)
    p["a2"] = m2*l1*p["lc2"]; p["a3"] = p["I2"]+m2*p["lc2"]**2
    return p

P_TRUE = make_params(m1=2.0, m2=1.5, l1=0.5, l2=0.4)
P_NOM  = make_params(m1=2.0*1.15, m2=1.5*0.85, l1=0.5, l2=0.4, lc2f=0.55)

def Mmat(q2, p):
    c2 = np.cos(q2)
    return p["a1"]+2*p["a2"]*c2, p["a3"]+p["a2"]*c2, p["a3"]*np.ones_like(q2)
def Cqd(q2, qd1, qd2, p):
    h = p["a2"]*np.sin(q2); return -h*(2*qd1*qd2+qd2**2), h*qd1**2
def gvec(q1, q2, p):
    g1 = (p["m1"]*p["lc1"]+p["m2"]*p["l1"])*G*np.cos(q1)+p["m2"]*p["lc2"]*G*np.cos(q1+q2)
    g2 = p["m2"]*p["lc2"]*G*np.cos(q1+q2); return g1, g2
def delta_theta(q, qd, u, pt, pn):
    q1,q2=q[:,0],q[:,1]; qd1,qd2=qd[:,0],qd[:,1]
    M11n,M12n,M22n=Mmat(q2,pn); C1n,C2n=Cqd(q2,qd1,qd2,pn); g1n,g2n=gvec(q1,q2,pn)
    tau1=M11n*u[:,0]+M12n*u[:,1]+C1n+g1n; tau2=M12n*u[:,0]+M22n*u[:,1]+C2n+g2n
    M11,M12,M22=Mmat(q2,pt); C1,C2=Cqd(q2,qd1,qd2,pt); g1,g2=gvec(q1,q2,pt)
    r1,r2=tau1-C1-g1,tau2-C2-g2; det=M11*M22-M12**2
    return np.stack([(M22*r1-M12*r2)/det-u[:,0], (-M12*r1+M11*r2)/det-u[:,1]], axis=1)
def Mfull(q2,p):
    M11,M12,M22=Mmat(np.array([q2]),p); return np.array([[M11[0],M12[0]],[M12[0],M22[0]]])
def Cfull(q2,qd1,qd2,p):
    h=p["a2"]*np.sin(q2); return np.array([[-h*qd2,-h*(qd1+qd2)],[h*qd1,0.0]])
def gfull(q1,q2,p):
    g1,g2=gvec(np.array([q1]),np.array([q2]),p); return np.array([g1[0],g2[0]])

def f_true(x, u):
    dl = delta_theta(x[:2][None], x[2:][None], u[None], P_TRUE, P_NOM)[0]
    return np.concatenate([x[2:], u+dl])
def rk4_step(x, u, dt, nsub=10):
    h=dt/nsub
    for _ in range(nsub):
        k1=f_true(x,u);k2=f_true(x+.5*h*k1,u);k3=f_true(x+.5*h*k2,u);k4=f_true(x+h*k3,u)
        x=x+h/6.0*(k1+2*k2+2*k3+k4)
    return x

# ======================================================================
# 2. PE data collection (unchanged)
# ======================================================================
TS = 0.01
EPS_P, EPS_V = 2e-4, 2e-3
U_MAX, Q_MAX, QD_MAX = 5.0, 1.2, 2.0
A0 = np.block([[np.eye(2), TS*np.eye(2)], [np.zeros((2,2)), np.eye(2)]])
B0 = np.vstack([TS**2/2*np.eye(2), TS*np.eye(2)])

def collect(T, seed, amp=0.7):
    r = np.random.default_rng(seed)
    w = 2*np.pi*np.array([0.23,0.61,1.13,1.87]); A = amp*np.array([0.45,0.30,0.17,0.08])
    ph = r.uniform(0,2*np.pi,(2,4)); t = np.arange(T+1)*TS
    qr  = np.stack([(A*np.sin(np.outer(t,w)+ph[i])).sum(1) for i in range(2)],1)
    qdr = np.stack([(A*w*np.cos(np.outer(t,w)+ph[i])).sum(1) for i in range(2)],1)
    qddr= np.stack([(-A*w**2*np.sin(np.outer(t,w)+ph[i])).sum(1) for i in range(2)],1)
    x = np.r_[qr[0],qdr[0]]; X=np.zeros((T+1,4)); U=np.zeros((T,2))
    Xtrue=np.zeros((T+1,4)); Xtrue[0]=x
    Vn = np.c_[r.normal(0.0, EPS_P, (T+1,2)), r.normal(0.0, EPS_V, (T+1,2))]
    for k in range(T):
        u = qddr[k]+80*(qr[k]-x[:2])+18*(qdr[k]-x[2:])+r.uniform(-.4,.4,2)
        u = np.clip(u,-U_MAX,U_MAX); U[k]=u
        x = rk4_step(x,u,TS)
        Xtrue[k+1]=x
    return Xtrue+Vn, U, Xtrue

print("Collecting PE data (no injected disturbance)...")
T_id = 1500
X_id, U_id, Xtrue_id = collect(T=T_id, seed=11, amp=0.7)
X_va, U_va, Xtrue_va = collect(T=800,  seed=42, amp=0.6)
Y0, Y1, U0 = X_id[:-1].T, X_id[1:].T, U_id.T
n, m = Y0.shape[0], U0.shape[0]

# ======================================================================
# 3. [3][6] Lemma-2 envelope, and [5] calibration of omega_bar
# ======================================================================
eta_a = norm.ppf(1.0 - DELTA_A/(2.0*n*(T_id + K_BAR)))

sigma_vec = np.array([EPS_P, EPS_P, EPS_V, EPS_V])
if ISOTROPIC:                       # Assumption 2 verbatim
    sd = np.full(n, sigma_vec.max()*np.sqrt(1.0 + KAPPA**2))
else:                               # per-coordinate: Var = e_i'(V + A V A')e_i
    Vhat = np.diag(sigma_vec**2)
    sd = np.sqrt(np.diag(Vhat + A0 @ Vhat @ A0.T))
noise_hw = eta_a * sd               # measurement-noise half-widths

# Chebyshev eps_min per row: min_Theta max_k |y_{k+1,a} - Theta z_k|  (one LP)
Z0 = np.vstack([Y0, U0]); d = n + m
A_ub = np.block([[Z0.T, -np.ones((T_id,1))],[-Z0.T, -np.ones((T_id,1))]])
c_lp = np.zeros(d+1); c_lp[-1] = 1.0
eps_min = np.array([
    linprog(c_lp, A_ub=A_ub, b_ub=np.concatenate([Y1[a,:], -Y1[a,:]]),
            bounds=[(None,None)]*d+[(0.0,None)], method="highs").x[-1]
    for a in range(n)])
# Assumption-3 prior bound.  The calibration deliberately does NOT subtract the
# measurement envelope: eps_min bounds discrepancy AND noise jointly, and the two
# are not separable from noisy data, so subtracting noise_hw returns zero wherever
# noise dominates and yields an omega_bar that fails Assumption 3.  Taking
# SAFETY*eps_min is conservative by construction (it double-counts noise) but is a
# valid prior bound on the process disturbance.
omega_bar = np.maximum(SAFETY*eps_min, OMEGA_FLOOR)
eps_bar = omega_bar + noise_hw
g = 2.0*omega_bar                    # fixed diagonal generator, |g_i| >= omega_bar_i
G_omega = np.diag(g)

print(f"[envelope] eta_a={eta_a:.4f}  isotropic={ISOTROPIC}")
print(f"           noise half-widths = {noise_hw}")
print(f"           eps_min           = {eps_min}")
print(f"           omega_bar (calib) = {omega_bar}")
print(f"           eps_bar           = {eps_bar}")

# ======================================================================
# 4. [1][2] Theorem-1 SDP, region route
# ======================================================================
print("Configuring SDP (Theorem 1, region route)...")
generators = []
for i in range(n):
    for j in range(n):
        GA=np.zeros((n,n));GA[i,j]=1.0; generators.append((GA, np.zeros((n,m))))
for i in range(n):
    for j in range(m):
        GB=np.zeros((n,m));GB[i,j]=1.0; generators.append((np.zeros((n,n)), GB))
q = len(generators)

V = np.zeros((T_id, n, q))
for k in range(T_id):
    for j,(GA,GB) in enumerate(generators):
        V[k,:,j] = GA@Y0[:,k] + GB@U0[:,k]

SB = sigma_vec.max()                            # nondimensionalisation scale
C_A=cp.Variable((n,n)); C_B=cp.Variable((n,m))
s_h=cp.Variable(q,nonneg=True)                  # s = SB * s_h
Sig_h=cp.Variable((n,n),PSD=True)               # Sigma = SB^2 * Sig_h
c_h=cp.Variable(n)                              # c_omega = SB * c_h
t_h=cp.Variable(n,nonneg=True)
lambdas=cp.Variable((n, T_id))

cons  = [cp.bmat([[Sig_h, cp.diag(t_h)], [cp.diag(t_h), np.eye(n)]]) >> 0]
cons += [cp.square(t_h) <= cp.diag(Sig_h)]
cons += [Sig_h >> np.diag((sd / SB) ** 2)]
cons += [lambdas <= 1, lambdas >= -1]

for k in range(T_id):
    r_k = (Y1[:,k] - C_A@Y0[:,k] - C_B@U0[:,k]) / SB
    for i in range(n):
        cons.append(
            cp.abs(r_k[i] - c_h[i] - (G_omega[i,:] @ lambdas[:,k])/SB)
            <= eta_a*t_h[i] - (s_h @ np.abs(V[k,i,:]))
        )
# [2] coverage (10h), in scaled units
cons.append(eta_a*cp.sqrt(cp.diag(Sig_h))
            >= eps_bar/SB + cp.abs(c_h) - np.abs(g)/SB)

obj = (cp.sum(s_h) + cp.trace(Sig_h)
       + cp.sum(cp.abs(lambdas))/T_id)
print("Solving SDP...")
cp.Problem(cp.Minimize(obj), cons).solve(solver=cp.CLARABEL, verbose=False)

A_hat, B_hat = C_A.value, C_B.value
s_star = np.maximum(s_h.value, 0.0)*SB
Sig    = Sig_h.value*SB**2
c_om   = c_h.value*SB
print(f"-> active s_i={int((s_star>1e-9).sum())}/{q}, sum(s*)={s_star.sum():.3e}"
      f"   (s* -> 0 is expected: nothing bounds M from below)")

# [4] Z_wtilde confidence-region half-widths
D_kappa = eta_a*np.sqrt(np.maximum(np.diag(Sig),0.0))
hw_dd   = np.abs(g) + D_kappa
c_dd    = c_om.copy()
print(f"[Z_wtilde] h_wtilde = {hw_dd}")
print(f"           eps_bar  = {eps_bar}   -> coverage binds: "
      f"{np.allclose(hw_dd, eps_bar, rtol=1e-3)}")

def residuals(X,U,A,B): return X[1:] - X[:-1]@A.T - U@B.T

floor_gap = np.linalg.eigvalsh(Sig - np.diag(sd**2)).min()
print(f"[Sigma floor] min eig(Sigma - diag(sd^2)) = {floor_gap:.3e}"
      f"  ({'OK' if floor_gap >= -1e-12 else 'VIOLATED'})")

# ======================================================================
# 5. Contraction metric P (unchanged)
# ======================================================================
print("Solving App.-B SDP for P...")
THETA_mass, THETA_lc2 = 0.20, 0.12
thetas = [make_params(P_NOM["m1"]*f1, P_NOM["m2"]*f2, 0.5, 0.4, lc2f=0.55*f3)
          for f1 in (1-THETA_mass,1+THETA_mass)
          for f2 in (1-THETA_mass,1+THETA_mass)
          for f3 in (1-THETA_lc2,1+THETA_lc2)]
Ns=4000
qs=np.c_[rng.uniform(-Q_MAX,Q_MAX,Ns),rng.uniform(-Q_MAX,Q_MAX,Ns)]
qds=np.c_[rng.uniform(-QD_MAX,QD_MAX,Ns),rng.uniform(-QD_MAX,QD_MAX,Ns)]
us=rng.uniform(-U_MAX,U_MAX,(Ns,2))
dmax=np.zeros(2)
for th in thetas:
    dmax=np.maximum(dmax, np.abs(delta_theta(qs,qds,us,th,P_NOM)).max(0))
wc_x = np.abs(B0)@dmax
verts_w = np.array(list(itertools.product([-1,1],repeat=4)))*wc_x
A_xc = np.vstack([np.c_[np.eye(2),np.zeros((2,2))], -np.c_[np.eye(2),np.zeros((2,2))],
                  np.c_[np.zeros((2,2)),np.eye(2)], -np.c_[np.zeros((2,2)),np.eye(2)]])
A_uc = np.vstack([np.eye(2),-np.eye(2)])
def solve_P(rho):
    E=cp.Variable((n,n),PSD=True); Y=cp.Variable((m,n))
    cx=cp.Variable(A_xc.shape[0]); cu=cp.Variable(A_uc.shape[0]); wb=cp.Variable()
    o=1/(2*(1-rho))*((A_xc.shape[0]+A_uc.shape[0])*wb+cx.sum()/0.1+cu.sum()/U_MAX)
    cn=[E>>np.eye(n),cp.bmat([[rho**2*E,(A0@E+B0@Y).T],[(A0@E+B0@Y),E]])>>0,cx>=0,cu>=0,wb>=0]
    for i in range(A_xc.shape[0]):
        cn+=[cp.bmat([[cx[i,None,None],A_xc[i,None]@E],[(A_xc[i,None]@E).T,E]])>>0]
    for i in range(A_uc.shape[0]):
        cn+=[cp.bmat([[cu[i,None,None],A_uc[i,None]@Y],[(A_uc[i,None]@Y).T,E]])>>0]
    for i in range(verts_w.shape[0]):
        cn+=[cp.bmat([[wb[None,None],verts_w[i][None]],[verts_w[i][:,None],E]])>>0]
    cp.Problem(cp.Minimize(o),cn).solve(solver=cp.CLARABEL)
    return (np.linalg.inv(E.value), Y.value@np.linalg.inv(E.value)) if E.value is not None else (None,None)
P=K=None
for rho in np.linspace(0.8,0.97,12):
    P,K = solve_P(rho)
    if P is not None: break
Psqrt = np.real(sqrtm(P)); PB = Psqrt@B0
def pnorm(W): return np.sqrt(np.einsum('ki,ij,kj->k', W, P, W))
def prad(center, halfw):
    corners = np.array(list(itertools.product([-1,1],repeat=4)))*halfw + center
    return pnorm(corners).max()

# ======================================================================
# 6. Analytical bound beta (unchanged)
# ======================================================================
print("Recomputing analytical constants a,b,c...")
qs2=np.c_[rng.uniform(-Q_MAX,Q_MAX,1200),rng.uniform(-Q_MAX,Q_MAX,1200)]
qds2=np.c_[rng.uniform(-QD_MAX,QD_MAX,1200),rng.uniform(-QD_MAX,QD_MAX,1200)]
a=b=c=0.0
for th in thetas:
    for (q1,q2),(qd1,qd2) in zip(qs2,qds2):
        M=Mfull(q2,th); Minv=np.linalg.inv(M)
        a=max(a, np.linalg.norm(PB@(Minv@(M-Mfull(q2,P_NOM))),2))
        b=max(b, np.linalg.norm(PB@(Minv@(Cfull(q2,qd1,qd2,th)-Cfull(q2,qd1,qd2,P_NOM))),2))
        gtil=Minv@(gfull(q1,q2,th)-gfull(q1,q2,P_NOM))
        acc=rng.uniform(-U_MAX,U_MAX,2); x=np.r_[q1,q2,qd1,qd2]
        xode=solve_ivp(lambda t,xt,u:np.r_[xt[2:],u+delta_theta(xt[:2][None],xt[2:][None],u[None],th,P_NOM)[0]],
                       [0,TS],x,args=(acc,)).y[:,-1]
        dth=delta_theta(x[:2][None],x[2:][None],acc[None],th,P_NOM)[0]
        c=max(c, np.linalg.norm(PB@gtil + Psqrt@(xode-A0@x-B0@(acc+dth)),2))
print(f"   a={a:.4e} b={b:.4e} c={c:.4e}")

# ======================================================================
# 7. THE TWO COMPARISONS
# ======================================================================
W_id = residuals(X_id, U_id, A_hat, B_hat)
qd_true = Xtrue_id[:-1, 2:]
beta = a*np.linalg.norm(U_id,axis=1) + b*np.linalg.norm(qd_true,axis=1) + c
wP   = pnorm(W_id)

# (I) discrepancy only:  omega_bar   vs   beta      [like for like]
rad_omega = prad(np.zeros(n), omega_bar)
ratio_I_max = beta.max()/rad_omega
ratio_I_med = np.median(beta)/rad_omega

# (II) full effective uncertainty:  Z_wtilde   vs   beta
rad_dd_P = prad(c_dd, hw_dd)
rad_noise = prad(np.zeros(n), D_kappa)
ratio_II_max = beta.max()/rad_dd_P
ratio_II_med = np.median(beta)/rad_dd_P

W_va = residuals(X_va, U_va, A_hat, B_hat)
cov_dd = 100*np.all(np.abs(W_va - c_dd) <= hw_dd + 1e-12, axis=1).mean()
cov_beta = 100*np.mean(pnorm(W_id) <= beta)

lines = [
 "=== Table 1 (region route): two comparisons in the same P-weighted norm ===",
 f"eta_a = {eta_a:.4f}   isotropic noise assumption = {ISOTROPIC}   kappa = {KAPPA}",
 f"Theta = +/-{int(THETA_mass*100)}% mass, +/-{int(THETA_lc2*100)}% lc2",
 f"analytical constants  a={a:.4e}  b={b:.4e}  c={c:.4e}",
 "",
 "(I) DISCREPANCY ONLY -- like for like (both bound B*Delta_theta + Delta_disc)",
 f"    omega_bar (calibrated)      = {omega_bar}",
 f"    P-radius omega_bar          = {rad_omega:.4e}",
 f"    P-radius beta_max           = {beta.max():.4e}",
 f"    conservatism ratio          : max {ratio_I_max:.2f}x , median {ratio_I_med:.2f}x",
 "",
 "(II) FULL EFFECTIVE UNCERTAINTY -- Z_wtilde also covers measurement noise",
 f"    h_wtilde = |g| + eta_a sqrt(Sigma_ii) = {hw_dd}",
 f"      of which measurement noise          = {D_kappa}",
 f"      eps_bar (Lemma 2 envelope)          = {eps_bar}",
 f"    P-radius Z_wtilde           = {rad_dd_P:.4e}",
 f"      of which noise alone      = {rad_noise:.4e}  "
 f"({100*rad_noise/rad_dd_P:.0f}% of the set)",
 f"    conservatism ratio          : max {ratio_II_max:.2f}x , median {ratio_II_med:.2f}x",
 "",
 f"beta(x,u) along traj : med={np.median(beta):.4e}  max={beta.max():.4e}",
 f"||w_tilde||_P        : med={np.median(wP):.4e}  max={wP.max():.4e}",
 f"coverage ||w||_P <= beta        : {cov_beta:.1f}%",
 f"Z_wtilde validation coverage    : {cov_dd:.1f}%  (target {100*(1-DELTA_A):.0f}%)",
 "",
 "NOTE  beta omits measurement noise and uses exact states, so (II) is",
 "      conservative in favour of the analytical bound; (I) is the comparison",
 "      that isolates what the identification actually buys.",
]
summary = "\n".join(lines); print(summary)
with open(OUT_DIR/"summary.txt","w") as f: f.write(summary+"\n")

# ======================================================================
# 7b. VERIFICATION: does the prior bound omega_bar satisfy Assumption 3?
#     In simulation the noise-free states are available, so the exact
#     realization of B*Delta_theta + Delta_disc can be formed and checked
#     against omega_bar.  This must report 100% for the finite-sample
#     guarantee of Theorem 1 to apply to this dataset.
# ======================================================================
D_true  = Xtrue_id[1:] - Xtrue_id[:-1] @ A0.T - U_id @ B0.T
dP_true = pnorm(D_true)
print("\n[diagnostic] TRUE discrepancy d_k = x_{k+1} - A0 x_k - B0 u_k")
print(f"             max |d_k| per coordinate = {np.abs(D_true).max(axis=0)}")
print(f"             omega_bar                = {omega_bar}")
print(f"             coverage |d_k| <= omega_bar (componentwise) : "
      f"{100*np.all(np.abs(D_true) <= omega_bar + 1e-15, axis=1).mean():.1f}%")
_cov_box = 100*np.all(np.abs(D_true) <= omega_bar + 1e-15, axis=1).mean()
print(f"             coverage ||d_k||_P <= beta(x_k,u_k)         : "
      f"{100*np.mean(dP_true <= beta):.1f}%")
print("             -> Assumption 3 "
      + ("SATISFIED on this dataset" if _cov_box == 100.0
         else "VIOLATED: raise SAFETY or widen the calibration"))
print(f"             ||d_k||_P : med={np.median(dP_true):.4e}  max={dP_true.max():.4e}")

# ======================================================================
# 7b. Comparison (III): like-for-like at equal coverage.
#     beta covers the model discrepancy only; Z_wtilde must also cover the
#     measurement-noise propagation.  Augmenting beta with the same noise
#     envelope puts both on equal footing.  {x : ||x||_P <= beta} is a
#     P-ball, so radii add exactly under the Minkowski sum.
# ======================================================================
beta_aug     = beta + rad_noise                 # per-step augmented radius
rad_aug_max  = beta.max()      + rad_noise
rad_aug_med  = np.median(beta) + rad_noise
ratio_III_max = rad_aug_max / rad_dd_P
ratio_III_med = rad_aug_med / rad_dd_P

print("\n(III) LIKE-FOR-LIKE -- both descriptions cover discrepancy + noise")
print(f"      noise envelope P-radius     = {rad_noise:.4e}")
print(f"      beta (+) N : max {rad_aug_max:.4e}   med {rad_aug_med:.4e}")
print(f"      Z_wtilde                    = {rad_dd_P:.4e}")
print(f"      conservatism ratio          : max {ratio_III_max:.2f}x , "
      f"median {ratio_III_med:.2f}x")

# ======================================================================
# 8. Figure: publication version (IEEE column width), both comparisons
# ======================================================================
TWO_COLUMN = False
FIG_W = 7.16 if TWO_COLUMN else 3.5
FIG_H = 3.20 if TWO_COLUMN else 2.65
FS    = 9    if TWO_COLUMN else 8

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
C_DD, C_ANL, C_AUG, C_OM, C_PTS = "#1f77b4", "#d62728", "#ff7f0e", "#2ca02c", "0.60"

fig, axT = plt.subplots(figsize=(FIG_W, FIG_H))
kk = np.arange(len(wP))

axT.plot(kk, wP, color="0.60", lw=0.5, alpha=0.85, zorder=1,
         label=r"$\Vert\tilde w_k\Vert_P$")
# comparison (I): discrepancy only
axT.axhline(rad_omega, color=C_OM, lw=2.0, ls=":", zorder=5,
            label=r"$\bar\omega$ radius")
axT.plot(kk, beta, color=C_ANL, lw=1.3, ls="--", zorder=4,
         label=r"$\beta(x_k,u_k)$")
# comparison (III): discrepancy + noise
axT.axhline(rad_dd_P, color=C_DD, lw=1.8, zorder=4,
            label=r"$\mathcal{Z}_{\tilde w}$ radius")
axT.plot(kk, beta_aug, color=C_AUG, lw=1.5, ls="-.", zorder=5,
         label=r"$\beta\oplus\mathcal{N}$")

def _gap(x_frac, lo, hi, txt):
    x = int(x_frac * len(kk))
    axT.annotate("", xy=(x, hi), xytext=(x, lo),
                 arrowprops=dict(arrowstyle="<->", color="0.25", lw=0.8,
                                 shrinkA=0, shrinkB=0), zorder=6)
    axT.text(x - 0.015*len(kk), np.sqrt(lo*hi), txt, fontsize=FS-1.5,
             color="0.25", ha="right", va="center", zorder=6,
             bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8))

_gap(0.60, rad_omega, np.median(beta), rf"${ratio_I_med:.1f}\times$")
_gap(0.90, rad_dd_P, rad_aug_med,      rf"${ratio_III_med:.2f}\times$")

axT.set_yscale("log")
axT.set_ylabel(r"$P$-weighted magnitude", labelpad=2)
axT.set_xlabel(r"time step $k$", labelpad=2)
axT.set_ylim(wP.min()*0.22, rad_aug_max*3.0)
axT.margins(x=0.01)
axT.legend(loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=3,
           frameon=False, fontsize=FS-1.5, handlelength=1.4,
           handletextpad=0.35, columnspacing=0.9, borderaxespad=0.3,
           labelspacing=0.25)

for ext in ("pdf", "png"):
    fig.savefig(OUT_DIR / f"fig1_residual_sets.{ext}")
print(f"\nFigure and summary written to: {OUT_DIR}")