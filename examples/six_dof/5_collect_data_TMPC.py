import numpy as np
import pickle

from aux import (TimeStepIntegratorContinuous,
                 get_linear_double_integrator_discrete_dynamics)
from problem_scenario import ProblemScenarioMassAllPin
from examples.six_dof import P_EXAMPLE_6_DOF

# ---------------------------------------------------------------- scenario --
p_d = P_EXAMPLE_6_DOF / "data" / "dof_6_ef_0.02"
ps = ProblemScenarioMassAllPin.from_cached_dir(p_d)

A, B = get_linear_double_integrator_discrete_dynamics(
    ps.config_dim, dt=ps.dt, method="zoh")
n, m = A.shape[0], B.shape[1]          # n = 12, m = 6
nq = ps.config_dim                     # 6

integrator = TimeStepIntegratorContinuous(dt=ps.dt)
integrator.set_dyns(ps.get_nominal_dynamics(), ps.get_err_dyn_random())

SIGMA = 1e-5                           # measurement-noise STD
V_hat = SIGMA ** 2 * np.eye(n)         

# ------------------------------------------------------------- excitation ---
T = 8000                               # more samples for the higher-dim system
np.random.seed(42)
rng = np.random.default_rng(42)

# Low-frequency multisine POSITION reference on all nq joints.
KP, KD = 80.0, 18.0                    # PD gains on the feedback-linearised plant
DITHER = 1.5                           # independent input dither
try:
    U_MAX = float(ps.u_amp_nom)        # respect the torque/accel limit
except Exception:
    U_MAX = 20.0

t = np.arange(T + 1) * ps.dt

BASE  = np.array([0.08, 0.17, 0.31, 0.53, 0.89, 1.30])
AMPS  = 0.70 * np.array([0.55, 0.40, 0.28, 0.16, 0.09, 0.05])
phase = rng.uniform(0.0, 2.0*np.pi, (nq, BASE.size))

# each joint gets its own frequency comb, offset so no two joints share a line
q_ref  = np.zeros((T+1, nq)); qd_ref = np.zeros((T+1, nq)); qdd_ref = np.zeros((T+1, nq))
for i in range(nq):
    w_i = 2.0*np.pi*(BASE + 0.011*i)     
    q_ref[:, i]   = (AMPS * np.sin(np.outer(t, w_i) + phase[i])).sum(1)
    qd_ref[:, i]  = (AMPS * w_i * np.cos(np.outer(t, w_i) + phase[i])).sum(1)
    qdd_ref[:, i] = (-AMPS * w_i**2 * np.sin(np.outer(t, w_i) + phase[i])).sum(1)

# ------------------------------------------------------------- simulate -----
X = np.zeros((n, T + 1))               # TRUE states, noise-free
U0 = np.zeros((m, T))
X[:, 0] = np.concatenate([q_ref[0], qd_ref[0]])

for k in range(T):
    x = X[:, k]
    u = (qdd_ref[k]
         + KP * (q_ref[k] - x[:nq])
         + KD * (qd_ref[k] - x[nq:])
         + rng.uniform(-DITHER, DITHER, m))
    u = np.clip(u, -U_MAX, U_MAX)
    U0[:, k] = u
    X[:, k + 1] = integrator.solve_time_step(x, u)

# ONE i.i.d. noise sequence, shared:  Y0[:,k]=x_k+ups_k, Y1[:,k]=x_{k+1}+ups_{k+1}
ups = rng.normal(0.0, SIGMA, size=(n, T + 1))
Y0 = X[:, :T] + ups[:, :T]
Y1 = X[:, 1:] + ups[:, 1:]

# ------------------------------------------------------------------ save ----
data = {
    "U0": U0, "Y0": Y0, "Y1": Y1,
    "V_hat": V_hat,
    "A_lin": A, "B_lin": B,
    "n": n, "m": m, "dt": ps.dt,
    "config_dim": ps.config_dim,
}
with open(p_d / "id_dataset.pckl", "wb") as f:
    pickle.dump(data, f)
print(f"Dataset saved to {p_d / 'id_dataset.pckl'}   (config_dim={nq}, "
      f"n={n}, m={m}, T={T})")

# ------------------------------------------------------------ diagnostics ---
Z0 = np.vstack([Y0, U0])
sv = np.linalg.svd(Z0, compute_uv=False)
print("\n--- excitation ---")
print(f"  std(q)    per joint : {Y0[:nq].std(axis=1)}")
print(f"  std(qdot) per joint : {Y0[nq:].std(axis=1)}")
print(f"  max|q|              : {np.abs(X[:nq]).max():.4f} rad  (old design: ~0.01)")
print(f"  max|qdot|           : {np.abs(X[nq:]).max():.4f} rad/s")
print(f"  max|u|              : {np.abs(U0).max():.4f}  (limit {U_MAX})")
print(f"  rank(Z0)            : {np.linalg.matrix_rank(Z0, tol=1e-8)} / {n + m}")
print(f"  cond(Z0)            : {sv[0] / sv[-1]:.1f}         (old design: O(100))")

W_tilde = Y1 - A @ Y0 - B @ U0
print("\n--- effective uncertainty w_tilde (vs the IDEAL ZOH model) ---")
print(f"  max |.| per dim : {np.abs(W_tilde).max(axis=1)}")
print(f"  std  per dim    : {W_tilde.std(axis=1)}")
print(f"  pure-noise std ~ sqrt(1+||A||^2)*sigma = "
      f"{np.sqrt(1 + np.linalg.norm(A, 2) ** 2) * SIGMA:.3e}")


# ------------------------------------------- structural check (ZOH ratio) ---
Z_true = np.vstack([X[:, :T], U0])
Theta_ls = X[:, 1:] @ np.linalg.pinv(Z_true)        # best linear one-step map
dTheta = Theta_ls - np.hstack([A, B])
dA_pos, dA_vel = np.abs(dTheta[:nq, :n]).max(), np.abs(dTheta[nq:, :n]).max()
dB_pos, dB_vel = np.abs(dTheta[:nq, n:]).max(), np.abs(dTheta[nq:, n:]).max()
ratio_A = dA_pos / max(dA_vel, 1e-300)
ratio_B = dB_pos / max(dB_vel, 1e-300)

print("\n--- structural check: position vs velocity rows of the one-step map ---")
print(f"  |dA| position rows : {dA_pos:.3e}      |dB| position rows : {dB_pos:.3e}")
print(f"  |dA| velocity rows : {dA_vel:.3e}      |dB| velocity rows : {dB_vel:.3e}")
print(f"  ratio pos/vel  (A) : {ratio_A:.4f}     (B) : {ratio_B:.4f}")
print(f"  structural prediction Ts/2 = {ps.dt / 2:.4f}")
need = max(ratio_A, ratio_B) / (ps.dt / 2)
print(f"  => struct_safety >= {need:.1f} is supported by this data")
if need > 5.0:
    print("     *** ratio well above Ts/2: higher-order terms are significant,")
    print("     *** so the structured hull will buy less than projected. ***")