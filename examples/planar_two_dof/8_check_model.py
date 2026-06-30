import pickle
import numpy as np
from aux import get_linear_double_integrator_discrete_dynamics
from examples.planar_two_dof import P_EXAMPLE_2_DOF

p_d = P_EXAMPLE_2_DOF / "data" / "dof_2_ef_0.1"
with open(p_d / "identified_model.pckl", "rb") as f:
    M = pickle.load(f)

A_id, B_id = M["A_id"], M["B_id"]
A_lin, B_lin = get_linear_double_integrator_discrete_dynamics(2, 0.01)

np.set_printoptions(precision=6, suppress=True)
print("=== A_id (identified) ===");  print(A_id)
print("\n=== A_lin (textbook) ===");  print(A_lin)
print("\n=== A_id - A_lin ===");      print(A_id - A_lin)
print(f"\n||A_id - A_lin||_F = {np.linalg.norm(A_id - A_lin):.6e}")

print("\n=== B_id (identified) ===");  print(B_id)
print("\n=== B_lin (textbook) ===");   print(B_lin)
print("\n=== B_id - B_lin ===");       print(B_id - B_lin)
print(f"\n||B_id - B_lin||_F = {np.linalg.norm(B_id - B_lin):.6e}")

# Relative error in B (control effectiveness) is what matters most
rel_B = np.linalg.norm(B_id - B_lin) / np.linalg.norm(B_lin)
print(f"\nRelative error in B: {rel_B:.2%}")