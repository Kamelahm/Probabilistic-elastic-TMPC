import numpy as np
import pickle
from aux import (TimeStepIntegratorContinuous,
                 get_linear_double_integrator_discrete_dynamics)
from problem_scenario import ProblemScenarioMassAllPin
from examples.six_dof import P_EXAMPLE_6_DOF

# -- Load scenario --------------------------------------------------------
p_d = P_EXAMPLE_6_DOF / "data" / "dof_6_ef_0.02"
ps  = ProblemScenarioMassAllPin.from_cached_dir(p_d)

A, B = get_linear_double_integrator_discrete_dynamics(
    ps.config_dim, dt=ps.dt, method="zoh")
n = A.shape[0]      # 12 for 6-DOF
m = B.shape[1]      # 6  for 6-DOF

integrator = TimeStepIntegratorContinuous(dt=ps.dt)
integrator.set_dyns(ps.get_nominal_dynamics(), ps.get_err_dyn_random())

V_hat = (1e-5) ** 2 * np.eye(n)

# -- Data collection ------------------------------------------------------
T = 5000                      # more samples for the higher-dim system
RESET_EVERY = 50              # reset state to keep it in the linear regime
np.random.seed(42)

# Balanced multisine excitation on ALL m joints, with independent random
# phases per joint so the input channels are uncorrelated (clean B id).
t = np.arange(T) * ps.dt
freqs = np.array([0.3, 0.7, 1.3, 2.1, 3.3, 5.0, 7.0])   # more frequencies
amp = 0.5                                       # per-channel amplitude
U0 = np.zeros((m, T))
for i in range(m):
    sig = np.zeros(T)
    for f in freqs:
        phase = np.random.uniform(0, 2 * np.pi)
        sig += np.sin(2 * np.pi * f * t + phase)
    U0[i] = amp * sig / np.sqrt(len(freqs))     # normalize power

# Respect the torque limits during data collection
u_lim = ps.u_amp_nom
U0 = np.clip(U0, -u_lim, u_lim)

Y0 = np.zeros((n, T))
Y1 = np.zeros((n, T))

x = np.zeros(n)
for k in range(T):
    if k % RESET_EVERY == 0:
        x = np.zeros(n)        # reset: keep state near the linearization point
    Y0[:, k] = x + np.random.multivariate_normal(np.zeros(n), V_hat)
    x_next   = integrator.solve_time_step(x, U0[:, k])
    Y1[:, k] = x_next + np.random.multivariate_normal(np.zeros(n), V_hat)
    x = x_next

# -- Save -----------------------------------------------------------------
data = {
    "U0": U0, "Y0": Y0, "Y1": Y1,
    "V_hat": V_hat,
    "A_lin": A, "B_lin": B,
    "n": n, "m": m, "dt": ps.dt,
    "config_dim": ps.config_dim,
}
out_path = p_d / "id_dataset.pckl"
with open(out_path, "wb") as f:
    pickle.dump(data, f)
print(f"Dataset saved to {out_path}")
print(f"  config_dim = {ps.config_dim}  ->  n = {n}, m = {m}")
print(f"  T = {T} samples, reset every {RESET_EVERY} steps")
print(f"  U0 shape: {U0.shape}, Y0 shape: {Y0.shape}, Y1 shape: {Y1.shape}")

# -- State range check (should stay small / linear) -----------------------
print(f"\n  max |position| reached: {np.abs(Y0[:ps.config_dim]).max():.4f}")
print(f"  max |velocity| reached: {np.abs(Y0[ps.config_dim:]).max():.4f}")

# -- Effective disturbance ------------------------------------------------
W_tilde = Y1 - A @ Y0 - B @ U0
print("\n--- Effective disturbance w_tilde_k ---")
print(f"  Max |.| per dim: {np.abs(W_tilde).max(axis=1)}")
print(f"  Std  per dim:    {W_tilde.std(axis=1)}")

# -- Persistence-of-excitation check (regressor rank) ---------------------
Phi = np.vstack([Y0, U0])              # (n+m, T) regressor
rank = np.linalg.matrix_rank(Phi)
print(f"\n--- Persistence of excitation ---")
print(f"  Regressor shape: {Phi.shape}, rank: {rank} / {n + m}")
if rank < n + m:
    print("  WARNING: regressor is rank-deficient; identification of B "
          "may be poor. Increase excitation richness or T.")