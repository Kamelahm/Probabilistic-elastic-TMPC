"""
Numerical illustration of the advantage of Theorem 1 (data-driven set-membership
identification of the post-feedback-linearization dynamics) over the analytical
flexible bound beta(x,a) of Proposition 1 in Wullt et al. (2026).

Effective-noise model (no separate process disturbance):
    omega_k := B*Delta_theta(x_k,u_k) + Delta_disc(x_k,u_k)        (the discrepancy IS omega_k)
    w_tilde_k = upsilon_{k+1} - A* upsilon_k + omega_k             (measurement noise + omega_k)

Both descriptions are compared in the SAME P-weighted norm, along the SAME
persistently-exciting trajectory, where P is the contraction metric obtained
from the App.-B SDP for this plant.

Outputs: fig1_residual_sets.png + summary.txt
"""
import os, itertools
from pathlib import Path
import numpy as np
import cvxpy as cp
from scipy.stats import norm
from scipy.linalg import sqrtm
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

# ======================================================================
# 1. Manipulator model (2-DOF planar, revolute, gravity in the plane)
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
# matrix forms (for the analytical constants)
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
# 2. PE data collection.  NO separate process disturbance:
#    omega_k IS the model discrepancy B*Delta_theta + Delta_disc, already
#    present in the nonlinear rollout.  Only measurement noise is added.
# ======================================================================
TS = 0.01
EPS_P, EPS_V = 2e-4, 2e-3              # measurement-noise bounds (our setup)
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
    x = np.r_[qr[0],qdr[0]]; X=np.zeros((T+1,4)); U=np.zeros((T,2)); Xtrue=np.zeros((T+1,4)); Xtrue[0]=x
    Vn = np.c_[r.normal(0.0, EPS_P, (T+1,2)), r.normal(0.0, EPS_V, (T+1,2))]
    for k in range(T):
        u = qddr[k]+80*(qr[k]-x[:2])+18*(qdr[k]-x[2:])+r.uniform(-.4,.4,2)
        u = np.clip(u,-U_MAX,U_MAX); U[k]=u
        x = rk4_step(x,u,TS)                       # discrepancy enters here = omega_k
        Xtrue[k+1]=x
    return Xtrue+Vn, U, Xtrue                       # measurement noise upsilon_k

print("Collecting PE data (no injected disturbance)...")
T_id = 1500
X_id, U_id, Xtrue_id = collect(T=T_id, seed=11, amp=0.7)
X_va, U_va, Xtrue_va = collect(T=800,  seed=42, amp=0.6)
Y0, Y1, U0 = X_id[:-1].T, X_id[1:].T, U_id.T
n, m = Y0.shape[0], U0.shape[0]

# ======================================================================
# 3. Theorem-1 SDP.  Learned mixed-zonotope set: center c_omega + bounded
#    zonotope G_omega*lambda_k (lambda_k learned, ||.||_inf<=1) + confidence
#    eta*sqrt(Sigma_kappa).  G_omega is a fixed generator (no injected disturbance).
# ======================================================================
print("Configuring SDP (Theorem 1)...")
generators = []
for i in range(n):
    for j in range(n):
        GA=np.zeros((n,n));GA[i,j]=1.0; generators.append((GA, np.zeros((n,m))))
for i in range(n):
    for j in range(m):
        GB=np.zeros((n,m));GB[i,j]=1.0; generators.append((np.zeros((n,n)), GB))
q = len(generators)
delta_conf = 0.05
eta = norm.ppf(1.0 - delta_conf/(2.0*n*T_id))

# fixed bounded-zonotope generator G_omega (s_omega columns); lambda_k learned per step.
s_omega = n

G_omega = np.diag([2*EPS_P+TS*EPS_V, 2*EPS_P+TS*EPS_V, 2*EPS_V, 2*EPS_V])

C_A=cp.Variable((n,n)); C_B=cp.Variable((n,m)); s_zon=cp.Variable(q,nonneg=True)
kappa=cp.Variable(nonneg=True); z_v=cp.Variable(n,nonneg=True)
Sigma_kappa=cp.Variable((n,n),PSD=True); c_omega=cp.Variable(n)
lambdas=cp.Variable((s_omega, T_id))             # learned bounded activations, ||.||_inf<=1

V = np.zeros((T_id, n, q))
for k in range(T_id):
    for j,(GA,GB) in enumerate(generators):
        V[k,:,j] = GA@Y0[:,k] + GB@U0[:,k]

cons = [kappa >= cp.norm(C_A,2) + s_zon @ np.array([np.linalg.norm(g[0],2) for g in generators])]
cons.append(cp.bmat([[Sigma_kappa, cp.diag(z_v)], [cp.diag(z_v).T, np.eye(n)]]) >> 0)
cons += [lambdas <= 1, lambdas >= -1]
for k in range(T_id):
    r_k = Y1[:,k] - C_A@Y0[:,k] - C_B@U0[:,k]
    for i in range(n):
        # cons.append(cp.abs(r_k[i]) + s_zon @ np.abs(V[k,i,:])
        #             + cp.abs(c_omega[i]) + cp.abs(G_omega[i,:] @ lambdas[:,k]) <= eta*z_v[i])
        cons.append(
            cp.abs(r_k[i] - c_omega[i] - G_omega[i,:] @ lambdas[:,k]) 
            <= eta*z_v[i] - s_zon @ np.abs(V[k,i,:])
        )

# 1. Keep the variance weight that gave you good physical noise bounds
LAMBDA_SIGMA = 1e3

# 2. Normalize the lambda penalty so it doesn't artificially scale by 1500
LAMBDA_LAMBDA = 1.0 / T_id 

# 3. Create the balanced objective function
obj = (cp.sum(s_zon) 
       + LAMBDA_SIGMA * cp.trace(Sigma_kappa)
       + LAMBDA_LAMBDA * cp.sum([cp.norm(lambdas[:,k],1) for k in range(T_id)]))

print("Solving SDP...")
# 4. Slightly relax Clarabel's absolute tolerance to clear the numerical warning
cp.Problem(cp.Minimize(obj), cons).solve(
    solver=cp.CLARABEL, 
    tol_gap_abs=1e-6, 
    tol_gap_rel=1e-6, 
    max_iter=100, 
    verbose=False
)

# obj = (cp.sum(s_zon) + cp.trace(Sigma_kappa)
#        + cp.sum([cp.norm(lambdas[:,k],1) for k in range(1,T_id)]))
# print("Solving SDP...")
# cp.Problem(cp.Minimize(obj), cons).solve(solver=cp.CLARABEL, verbose=False)

A_hat, B_hat = C_A.value, C_B.value
s_star = np.maximum(s_zon.value, 0.0)
c_om, Sig = c_omega.value, Sigma_kappa.value
print(f"-> active s_i={int((s_star>1e-6).sum())}/{q}, sum(s*)={s_star.sum():.3e}")

# ---- effective-noise set Z_wtilde = <c_omega, G_omega, I_n, 0, Sigma_kappa>_MZ ----
mz_axis = (s_star[None,None,:]*np.abs(V)).sum(2).max(0)   # matrix-zonotope spread
dz_axis = np.abs(G_omega).sum(1)                          # bounded zonotope half-width
D_kappa = eta*np.sqrt(np.maximum(np.diag(Sig),0.0))       # confidence half-width
c_dd  = c_om.copy()
hw_dd = mz_axis + dz_axis + D_kappa
G_dd  = np.diag(hw_dd)
def residuals(X,U,A,B): return X[1:] - X[:-1]@A.T - U@B.T

# ======================================================================
# 4. Contraction metric P for THIS plant (App.-B SDP, eq. 32).
# ======================================================================
print("Solving App.-B SDP for P...")
THETA_mass, THETA_lc2 = 0.20, 0.12          # parametric uncertainty set (contains true mismatch)
thetas = [make_params(P_NOM["m1"]*f1, P_NOM["m2"]*f2, 0.5, 0.4, lc2f=0.55*f3)
          for f1 in (1-THETA_mass,1+THETA_mass)
          for f2 in (1-THETA_mass,1+THETA_mass)
          for f3 in (1-THETA_lc2,1+THETA_lc2)]
# worst-case accel-space model error over Theta x (X x U) -> state box
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

# ======================================================================
# 5. Analytical bound beta(x,a) of Proposition 1, recomputed for this plant.
#    a=max||P^.5 B Mtil||, b=max||P^.5 B Ctil||, c=max||P^.5(B gtil)+P^.5 ddisc||
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
# 6. P-norm comparison along the trajectory + coverage on validation set
# ======================================================================
W_id   = residuals(X_id,   U_id,   A_hat, B_hat)          # effective-noise residuals (measured)
W_idtr = Xtrue_id[1:]-Xtrue_id[:-1]@A_hat.T-U_id@B_hat.T   # noise-free
qd_true = Xtrue_id[:-1, 2:]
beta = a*np.linalg.norm(U_id,axis=1) + b*np.linalg.norm(qd_true,axis=1) + c   # Prop.1 along traj
wP   = pnorm(W_id)

# data-driven set P-radius (max over its box corners), and analytical max radius
corners = np.array(list(itertools.product([-1,1],repeat=4)))*hw_dd + c_dd
rad_dd_P = pnorm(corners).max()
ratio_max = beta.max()/rad_dd_P
ratio_med = np.median(beta)/rad_dd_P

# validation coverage of the data-driven set (should be ~ 1-delta)
W_va = residuals(X_va, U_va, A_hat, B_hat)
cov_dd = 100*np.all(np.abs(W_va - c_dd) <= hw_dd + 1e-12, axis=1).mean()
cov_beta = 100*np.mean(pnorm(W_id) <= beta)

lines = [
 "=== Theorem-1 set Z_wtilde  vs  analytical beta(x,a) [Prop.1], P-norm ===",
 f"omega_k model: discrepancy (no separate injection); meas-noise kept (ours)",
 f"SDP: active s_i = {int((s_star>1e-6).sum())}/{q}, sum(s*) = {s_star.sum():.3e}, eta = {eta:.3f}",
 f"P eigvals = {np.linalg.eigvalsh(P)}",
 f"Theta = +/-{int(THETA_mass*100)}% mass, +/-{int(THETA_lc2*100)}% lc2 (contains true mismatch)",
 f"analytical constants  a={a:.4e}  b={b:.4e}  c={c:.4e}",
 "",
 f"Z_wtilde half-widths [q1,q2,qd1,qd2] = {hw_dd}",
 f"  matrix-zonotope spread             = {mz_axis}",
 f"  bounded zonotope ||e_i^T G_omega||1 = {dz_axis}",
 f"  confidence eta*sqrt(Sigma_ii)      = {D_kappa}",
 f"  learned |lambda_k| mean/max        = {np.abs(lambdas.value).mean():.3f} / {np.abs(lambdas.value).max():.3f}",
 "",
 f"beta(x,a) along traj : med={np.median(beta):.4e}  max={beta.max():.4e}",
 f"||w_tilde||_P        : med={np.median(wP):.4e}  max={wP.max():.4e}",
 f"coverage ||w||_P <= beta            : {cov_beta:.1f}%",
 f"data-driven set valid. coverage     : {cov_dd:.1f}%  (target {100*(1-delta_conf):.0f}%)",
 "",
 f"P-radius  Z_wtilde = {rad_dd_P:.4e}",
 f"P-radius  beta_max = {beta.max():.4e}",
 f"conservatism ratio (beta/Z) : max {ratio_max:.2f}x , median {ratio_med:.2f}x",
]
summary = "\n".join(lines); print(summary)
with open(OUT_DIR/"summary.txt","w") as f: f.write(summary+"\n")



# ======================================================================
# 7. Figure: per-step P-norm trace.
#    ||w_tilde_k||_P  vs  beta(x_k,a_k), with the data-driven set P-radius as
#    a horizontal line.  Both sides are P-radii, so there is no projection
#    distortion: residuals < Z_wtilde radius < analytical bound at every step.
# ======================================================================
plt.rcParams.update({"font.size":11,"axes.grid":True,"grid.alpha":0.3,
                     "figure.dpi":160,"savefig.bbox":"tight"})
C_DD, C_ANL, C_PTS = "#1f77b4", "#d62728", "0.55"
 
fig, axT = plt.subplots(figsize=(9.0, 4.2))
kk = np.arange(len(wP))
axT.plot(kk, beta, color=C_ANL, lw=1.4, ls="--",
         label=r"analytical bound $\beta(x_k,u_k)$")
axT.plot(kk, wP, color=C_PTS, lw=0.9, alpha=0.9,
         label=r"residual $\Vert\tilde w_k\Vert_P$")
axT.axhline(rad_dd_P, color=C_DD, lw=1.8,
            label=r"$\mathcal{Z}_{\tilde w}$ $P$-radius")
axT.set_yscale("log")
axT.set_ylabel(r"$P$-weighted magnitude")
axT.set_xlabel(r"time step $k$")
axT.legend(loc="center right", frameon=False, fontsize=9)
axT.margins(x=0.01)
fig.savefig(OUT_DIR/"fig1_residual_sets.png")
print(f"\nFigure and summary written to: {OUT_DIR}")
