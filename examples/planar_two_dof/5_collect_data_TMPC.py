import numpy as np
import pickle
from aux import (TimeStepIntegratorContinuous,
                 get_linear_double_integrator_discrete_dynamics)
from problem_scenario import ProblemScenarioMassAllPin
from examples.planar_two_dof import P_EXAMPLE_2_DOF

# -- Load scenario --------------------------------------------------------
p_d = P_EXAMPLE_2_DOF / "data" / "dof_2_ef_0.1"
ps  = ProblemScenarioMassAllPin.from_cached_dir(p_d)

A, B = get_linear_double_integrator_discrete_dynamics(
    ps.config_dim, dt=ps.dt, method="zoh")
n = A.shape[0]
m = B.shape[1]

integrator = TimeStepIntegratorContinuous(dt=ps.dt)
integrator.set_dyns(ps.get_nominal_dynamics(), ps.get_err_dyn_random())

V_hat = (1e-5) ** 2 * np.eye(n)

# -- Improved data collection --------------------------------------------
T = 2000                      # 8x more samples
RESET_EVERY = 50              # reset state to keep it in the linear regime
np.random.seed(42)

# Balanced multisine excitation: same amplitude / spectrum on BOTH joints,
# independent random phases so the channels are uncorrelated.
t = np.arange(T) * ps.dt
freqs = np.array([0.5, 1.1, 2.0, 3.3, 5.0])   # rich frequency content (Hz)
amp = 0.5                                       # per-channel amplitude
U0 = np.zeros((m, T))
for i in range(m):
    sig = np.zeros(T)
    for f in freqs:
        phase = np.random.uniform(0, 2 * np.pi)
        sig += np.sin(2 * np.pi * f * t + phase)
    U0[i] = amp * sig / np.sqrt(len(freqs))     # normalize power

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