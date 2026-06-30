"""
Data-Driven Elastic Tube MPC Controller (Algorithm 1 of the paper)

Implements the corridor-following MPC that uses:
  - Identified model (A_id, B_id, s*, Sigma_kappa, c_omega) from Theorem 1
  - Lyapunov-shaped polyhedral tube  E_k = { e : H_e e <= h_e_k }
  - Online contractive tube update from Theorem 2
  - Elastic TMPC nominal problem from Eq. (21)

At each MPC step:
   1. Solve nominal MPC (21) with current K_k     -> (y_bar^(1), u_bar^(1))
   2. Solve contractive tube (22) at y_bar^(1)    -> (K_k, lambda_k, h_e_next)
   3. Re-solve nominal MPC (21) with updated K_k  -> (y_bar^(2), u_bar^(2))
   4. Apply u_k = u_bar^(2)[:,0] + K_k @ e_k
"""

import numpy as np
import cvxpy as cp
import pickle


class DDElasticTubeController:

    # ================================================================== #
    #  Construction / loading                                            #
    # ================================================================== #
    def __init__(
        self,
        # Identified model (from Theorem 1)
        A_bar, B_bar,
        s_star, Sigma_kappa,
        c_omega, G_omega, generators,
        eta, kappa,
        # Tube design (from Theorem 2)
        H_e, h_e_0,
        K_init, lambda_init, rho_init,
        P_lyap, P_sqrt,
        M_e_k,
        # MPC weights and constraints
        Q, R, Q_terminal, N,
        u_lim, v_lim, p_amp,
        # Solver
        use_solver=cp.CLARABEL,
    ):
        # --- Model ---
        self.A_bar = A_bar
        self.B_bar = B_bar
        self.n = A_bar.shape[0]
        self.m = B_bar.shape[1]

        # # TEST: override with exact linear double integrator
        # from aux import get_linear_double_integrator_discrete_dynamics
        # A_lin, B_lin = get_linear_double_integrator_discrete_dynamics(
        #     self.n // 2, 0.01)
        # self.A_bar = A_lin
        # self.B_bar = B_lin

        # --- Zonotope / disturbance ---
        self.s_star      = s_star
        self.Sigma_kappa = Sigma_kappa
        self.c_omega     = c_omega
        self.G_omega     = G_omega
        self.generators  = generators
        self.q           = len(generators)
        self.s_omega     = G_omega.shape[1]
        self.eta         = eta
        self.kappa       = kappa

        # --- Tube ---
        self.H_e   = H_e
        self.s_f   = H_e.shape[0]
        self.P_lyap = P_lyap
        self.P_sqrt = P_sqrt
        self.M_e_k_init = M_e_k

        # --- MPC ---
        self.Q          = Q
        self.R          = R
        self.Q_terminal = Q_terminal
        self.N          = N
        self.u_lim      = u_lim
        self.v_lim      = v_lim
        self.p_amp      = p_amp
        self.config_dim = self.n // 2

        # --- Online state (re-updated at each MPC step) ---
        self.h_e_k    = h_e_0.copy()
        self.K_k      = K_init.copy()
        self.lambda_k = float(lambda_init)
        self.rho_k    = float(rho_init)
        self.M_e_k    = float(M_e_k)

        # --- Precompute fixed quantities used in (22b) ---
        self.D_kappa = self.eta * np.sqrt(
            np.maximum(np.diag(self.H_e @ self.Sigma_kappa @ self.H_e.T), 0)
        )
        self.He_c_omega = self.H_e @ self.c_omega

        # Per-facet sums of |H_e G_omega[:,i]|
        self.G_omega_term = np.sum(np.abs(self.H_e @ self.G_omega), axis=1)

        # Per-facet zonotope generator bounds (constant in y_bar, u_bar)
        self.sum_GA = np.zeros(self.s_f)
        self.sum_GB = np.zeros(self.s_f)
        for i in range(self.q):
            self.sum_GA += s_star[i] * np.linalg.norm(
                self.H_e @ generators[i][0], axis=1
            )
            self.sum_GB += s_star[i] * np.linalg.norm(
                self.H_e @ generators[i][1], axis=1
            )

        self.use_solver = use_solver

        # --- Storage of the most recent MPC solution (for the simulator) ---
        self.last_y_bar    = None     # (n, N+1)
        self.last_u_bar    = None     # (m, N)
        self.last_predicted_traj = None  # for plotting/replanning

        # --- Compatibility aliases for the simulator interface ---
        self.A = self.A_bar          # simulator reads cont.A.shape
        self.B = self.B_bar

    # ================================================================== #
    #  Public factory: load from saved pickle files                      #
    # ================================================================== #
    @classmethod
    def from_cached_dir(cls, p_cached_dir, Q, R, Q_terminal, N,
                        u_lim, v_lim, p_amp):
        """Load identified model + tube parameters from disk."""
        with open(p_cached_dir / "identified_model.pckl", "rb") as f:
            mod = pickle.load(f)
        with open(p_cached_dir / "tube_initial.pckl", "rb") as f:
            tube = pickle.load(f)

        return cls(
            A_bar=mod["A_id"], B_bar=mod["B_id"],
            s_star=mod["s_star"], Sigma_kappa=mod["Sigma_kappa"],
            c_omega=mod["c_omega"], G_omega=tube["G_omega"],
            generators=mod["generators"],
            eta=mod["eta"], kappa=mod["kappa"],
            H_e=tube["H_e"], h_e_0=tube["h_e_0"],
            K_init=tube["K_k"], lambda_init=tube["lambda_k"],
            # K_init=tube["K_lqr"], lambda_init=tube["lambda_k"],
            rho_init=tube["rho_k"],
            P_lyap=tube["P_lyap"], P_sqrt=tube["P_sqrt"],
            M_e_k=tube["M_e_k"],
            Q=Q, R=R, Q_terminal=Q_terminal, N=N,
            u_lim=u_lim, v_lim=v_lim, p_amp=p_amp,
        )

    # ================================================================== #
    #  Helpers                                                           #
    # ================================================================== #
    def _tube_radius_in_direction(self, h_vec, direction):
        """
        Support function of the polytope { e : H_e e <= h_vec } evaluated at
        a given direction vector.  Computed via LP duality:
            max  direction @ e   s.t.  H_e @ e <= h_vec
          = min  h_vec @ lambda  s.t.  H_e^T @ lambda = direction,  lambda>=0
        For a Lyapunov-shaped polytope H_e = [P_sqrt; -P_sqrt], a closed-form
        bound is  max_e |direction @ e| <= ||P_sqrt^-1 direction||_1 * h_vec_max
        For simplicity we use the worst-case bound:
            r_dir <= ||direction @ pinv(H_e)||_1  *  max(h_vec)
        which is conservative but cheap.
        """
        # Tight LP-based support: solve a small LP via dualization
        # Here we use a simple, conservative form for speed.
        H_pinv = np.linalg.pinv(self.H_e)
        coeffs = direction @ H_pinv
        return float(np.sum(np.abs(coeffs) * h_vec))

    def _tightening_state(self, A_x_row, h_vec):
        """For a state-constraint row A_x_row, return the tightening
        max_{e in E_k} A_x_row @ e  =  support function of E_k at A_x_row."""
        return self._tube_radius_in_direction(h_vec, A_x_row)

    def _tightening_input(self, A_u_row, K, h_vec):
        """For an input-constraint row A_u_row, return the tightening
        max_{e in E_k} A_u_row @ K @ e  =  support function of K E_k at A_u_row."""
        composed = A_u_row @ K
        return self._tube_radius_in_direction(h_vec, composed)

    def _tube_radius_position(self, h_vec):
        """Worst-case position-error magnitude over the tube  (for SOC corridor)."""
        # max_{e in E_k} ||e[:nq]||_2 ;  conservative bound via pinv
        H_pinv = np.linalg.pinv(self.H_e)
        # Each unit position direction maps to a row through pinv
        nq = self.config_dim
        max_r = 0.0
        for d in range(nq):
            direction = np.zeros(self.n)
            direction[d] = 1.0
            r = self._tube_radius_in_direction(h_vec, direction)
            max_r = max(max_r, r)
        return max_r



    # # ================================================================== #
    # #  Step 1 + 3 of Algorithm 1: solve nominal MPC  (Eq. 21)            #
    # # ================================================================== #
    # def _solve_nominal_mpc(self, y0, corridor_H, corridor_h, p_g, K):
    #     """
    #     corridor_H : list of (F_j, nq) facet-normal matrices, one per stage
    #     corridor_h : list of (F_j,)    facet offsets, one per stage
    #     p_g        : (nq,)  final/sub-goal position
    #     K          : (m, n) current feedback gain
    #     Returns (success, U_bar_value, Y_bar_value, predicted_traj)
    #     """
    #     n, m, N = self.n, self.m, self.N
    #     nq = self.config_dim
    #     h = self.h_e_k

    #     Y_bar = cp.Variable((n, N + 1))
    #     U_bar = cp.Variable((m, N))

    #     x_g_v = np.concatenate([p_g, np.zeros(nq)])

    #     cost = 0
    #     constraints = [Y_bar[:, 0] == y0]

    #     # Input tightening (state-independent for given K)
    #     u_tighten_pos = np.zeros(m)
    #     u_tighten_neg = np.zeros(m)
    #     for j_in in range(m):
    #         row = np.zeros(m); row[j_in] = 1.0
    #         u_tighten_pos[j_in] = self._tightening_input(row, K, h)
    #         row = np.zeros(m); row[j_in] = -1.0
    #         u_tighten_neg[j_in] = self._tightening_input(row, K, h)

    #     # Velocity tightening (state-independent)
    #     v_tighten_pos = np.zeros(nq)
    #     v_tighten_neg = np.zeros(nq)
    #     for j in range(nq):
    #         row = np.zeros(n); row[nq + j] = 1.0
    #         v_tighten_pos[j] = self._tightening_state(row, h)
    #         row = np.zeros(n); row[nq + j] = -1.0
    #         v_tighten_neg[j] = self._tightening_state(row, h)

    #     def _facet_tighten(H_stage):
    #         """Per-facet tube tightening: h^c_i -> h^c_i - support_{E_k}(H^c_i).
    #         Lifts each position-facet normal to the full state (zeros on the
    #         velocity rows) and evaluates the tube support function."""
    #         t = np.zeros(H_stage.shape[0])
    #         for i in range(H_stage.shape[0]):
    #             n_full = np.zeros(self.n)
    #             n_full[:nq] = H_stage[i]
    #             t[i] = self._tube_radius_in_direction(h, n_full)
    #         return t

    #     for j in range(N):
    #         # Stage cost
    #         cost += cp.quad_form(Y_bar[:, j] - x_g_v, self.Q) \
    #               + cp.quad_form(U_bar[:, j], self.R)

    #         # Nominal dynamics (21b)
    #         constraints += [
    #             Y_bar[:, j + 1] == self.A_bar @ Y_bar[:, j]
    #                              + self.B_bar @ U_bar[:, j]
    #         ]

    #         # Tightened polyhedral corridor (realizes 21c on the position part)
    #         if j >= 1:
    #             H_cj, h_cj = corridor_H[j], corridor_h[j]
    #             tighten = _facet_tighten(H_cj)
    #             h_eff = h_cj - tighten
    #             # If tightening makes a facet infeasible, the tube exceeds the
    #             # corridor here; keep the (now empty-ish) constraint so the
    #             # solver reports infeasibility rather than silently dropping it.
    #             constraints += [H_cj @ Y_bar[:nq, j] <= h_eff]

    #         # Tightened velocity constraints
    #         constraints += [
    #             Y_bar[nq:, j] <=  self.v_lim - v_tighten_pos,
    #             Y_bar[nq:, j] >= -self.v_lim + v_tighten_neg,
    #         ]

    #         # Tightened input constraints (21d)
    #         constraints += [
    #             U_bar[:, j] <=  self.u_lim - u_tighten_pos,
    #             U_bar[:, j] >= -self.u_lim + u_tighten_neg,
    #         ]

    #     # Terminal cost (corridor-terminal retained; terminal set deferred)
    #     cost += cp.quad_form(Y_bar[:, N] - x_g_v, self.Q_terminal)

    #     # Terminal polyhedral corridor constraint
    #     H_cN, h_cN = corridor_H[N], corridor_h[N]
    #     tighten_N = _facet_tighten(H_cN)
    #     constraints += [H_cN @ Y_bar[:nq, N] <= h_cN - tighten_N]

    #     prob = cp.Problem(cp.Minimize(cost), constraints)
    #     try:
    #         prob.solve(solver=self.use_solver)
    #     except Exception:
    #         return False, None, None, None

    #     if Y_bar.value is None or U_bar.value is None:
    #         return False, None, None, None

    #     predicted_traj = Y_bar.value[:nq, :].T
    #     return True, U_bar.value, Y_bar.value, predicted_traj
    
    
    # ================================================================== #
    #  Step 1 + 3 of Algorithm 1: solve nominal MPC  (Eq. 21)            #
    # ================================================================== #
    def _solve_nominal_mpc(self, y0, corridor_centers, corridor_radii, p_g, K):
        """
        corridor_centers : (N+1, nq)   ball centers along the horizon
        corridor_radii   : (N+1,)      ball radii
        p_g              : (nq,)       final goal position
        K                : (m, n)      current feedback gain
        Returns (success, U_bar_value, Y_bar_value, predicted_traj)
        """
        n, m, N = self.n, self.m, self.N
        nq = self.config_dim
        h = self.h_e_k

        Y_bar = cp.Variable((n, N + 1))
        U_bar = cp.Variable((m, N))

        # Goal state: position = p_g, velocity = 0
        x_g_v = np.concatenate([p_g, np.zeros(nq)])

        cost = 0
        constraints = [Y_bar[:, 0] == y0]

        # Precompute input tightening (state-independent for given K)
        nq_dim = m
        u_tighten_pos = np.zeros(m)
        u_tighten_neg = np.zeros(m)
        for j_in in range(m):
            row = np.zeros(m); row[j_in] = 1.0
            u_tighten_pos[j_in] = self._tightening_input(row, K, h)
            row = np.zeros(m); row[j_in] = -1.0
            u_tighten_neg[j_in] = self._tightening_input(row, K, h)

        # Velocity tightening (state-independent)
        v_tighten_pos = np.zeros(nq)
        v_tighten_neg = np.zeros(nq)
        for j in range(nq):
            row = np.zeros(n); row[nq + j] = 1.0
            v_tighten_pos[j] = self._tightening_state(row, h)
            row = np.zeros(n); row[nq + j] = -1.0
            v_tighten_neg[j] = self._tightening_state(row, h)

        # Position tube radius for the corridor SOC tightening
        pos_radius = self._tube_radius_position(h)

        for j in range(N):
            # Stage cost
            cost += cp.quad_form(Y_bar[:, j] - x_g_v, self.Q) \
                  + cp.quad_form(U_bar[:, j], self.R)

            # Nominal dynamics  (21b)
            constraints += [
                Y_bar[:, j + 1] == self.A_bar @ Y_bar[:, j]
                                 + self.B_bar @ U_bar[:, j]
            ]

            # corridor/pinning constraint there can conflict and cause infeasibility.
            if j >= 1:
                r_eff = corridor_radii[j] - pos_radius
                if r_eff > 1e-6:
                    constraints += [
                        cp.norm(Y_bar[:nq, j] - corridor_centers[j], 2) <= r_eff
                    ]
                else:
                    constraints += [
                        Y_bar[:nq, j] == corridor_centers[j]
                    ]

            # Tightened velocity constraints
            constraints += [
                Y_bar[nq:, j] <=  self.v_lim - v_tighten_pos,
                Y_bar[nq:, j] >= -self.v_lim + v_tighten_neg,
            ]

            # Tightened input constraints  (21d)
            constraints += [
                U_bar[:, j] <=  self.u_lim - u_tighten_pos,
                U_bar[:, j] >= -self.u_lim + u_tighten_neg,
            ]

        # Terminal cost
        cost += cp.quad_form(Y_bar[:, N] - x_g_v, self.Q_terminal)
        # Terminal corridor constraint
        r_eff_N = corridor_radii[N] - pos_radius
        if r_eff_N > 1e-6:
            constraints += [
                cp.norm(Y_bar[:nq, N] - corridor_centers[N], 2) <= r_eff_N
            ]
        else:
            constraints += [
                Y_bar[:nq, N] == corridor_centers[N]
            ]

        #######################################
        constraints += [Y_bar[nq:, N] == np.zeros(nq)]
        #######################################

        prob = cp.Problem(cp.Minimize(cost), constraints)
        try:
            prob.solve(solver=self.use_solver)
        except Exception:
            return False, None, None, None

        if Y_bar.value is None or U_bar.value is None:
            return False, None, None, None

        # Predicted trajectory (positions only, for next-step corridor planning)
        predicted_traj = Y_bar.value[:nq, :].T   # (N+1, nq)

        return True, U_bar.value, Y_bar.value, predicted_traj
    

    # ================================================================== #
    #  Step 2 of Algorithm 1: update contractive tube  (Eq. 22)          #
    # ================================================================== #
    def _update_tube(self, y_bar_k, u_bar_k):
        """Solve Eq. (22) at the current operating point to obtain
        new (K_k, lambda_k, h_{e,k+1})."""
        s_f = self.s_f
        h_e = self.h_e_k

        # Nominal trajectory contribution (depends on y_bar, u_bar)
        sum_nom = np.zeros(s_f)
        for i in range(self.q):
            sum_nom += self.s_star[i] * np.abs(
                self.H_e @ (self.generators[i][0] @ y_bar_k
                          + self.generators[i][1] @ u_bar_k)
            )

        P_k    = cp.Variable((s_f, s_f), nonneg=True)
        K_k    = cp.Variable((self.m, self.n))
        rho_k  = cp.Variable(nonneg=True)
        lam_k  = cp.Variable(nonneg=True)
        h_next = cp.Variable(s_f, nonneg=True)

        sigma_w = 1.0
        # sigma_w = 1e3
        constraints = [
            P_k @ h_e <= (
                lam_k * h_e
                - self.He_c_omega
                - self.G_omega_term
                - self.M_e_k * self.sum_GA
                - rho_k * self.M_e_k * self.sum_GB
                - sum_nom
                - self.D_kappa
            ),
            P_k @ self.H_e == self.H_e @ (self.A_bar + self.B_bar @ K_k),
            h_next == lam_k * h_e,
            cp.norm(K_k, 2) <= rho_k,
            lam_k <= 0.999,
            lam_k >= 0.001,
        ]
        obj = cp.Minimize(rho_k + sigma_w * lam_k)
        # obj = cp.Minimize(lam_k + (1.0 / sigma_w) * rho_k)
        prob = cp.Problem(obj, constraints)
    
        try:
            prob.solve(solver=self.use_solver)
        except Exception:
                self._next_h_e_k = self.h_e_k.copy()   # keep previous tube
                return False

        if K_k.value is None or prob.status not in (
                    "optimal", "optimal_inaccurate"):
                self._next_h_e_k = self.h_e_k.copy()   # keep previous tube
                return False

        # Accept the update
        self.K_k      = K_k.value
        self.lambda_k = float(lam_k.value)
        self.rho_k    = float(rho_k.value)


        h_min_abs = 1e-2 * self.h_e_k   # never drop below 1% of current tube
        self._next_h_e_k = np.maximum(h_next.value, h_min_abs)

        return True

    
    # ================================================================== #
    #  One full MPC step  (Algorithm 1, lines 4 - 9, FAITHFUL)           #
    # ================================================================== #
    def step(self, y_k, y_bar_k, corridor_centers, corridor_radii, p_g):
    # def step(self, y_k, y_bar_k, corridor_H, corridor_h, p_g):
        """
        Faithful Algorithm 1:
          5. solve nominal MPC with current K_k        -> y_bar^(1), u_bar^(1)
          6. solve contractivity SDP (22) at nominal   -> K_k, lambda_k, h_next
          7. re-solve nominal MPC with updated K_k     -> y_bar^(2), u_bar^(2)
          8. apply u_k = u_bar^(2)[:,0] + K_k e_k
          9. update tube scaling h_e_k <- h_next
        """
        # ---- One-time sign sanity check ----
        if not hasattr(self, "_sign_checked"):
            eig_plus  = np.max(np.abs(np.linalg.eigvals(
                self.A_bar + self.B_bar @ self.K_k)))
            eig_minus = np.max(np.abs(np.linalg.eigvals(
                self.A_bar - self.B_bar @ self.K_k)))
            print(f"[SIGN CHECK]  rho(A+BK) = {eig_plus:.4f}")
            print(f"[SIGN CHECK]  rho(A-BK) = {eig_minus:.4f}")
            print(f"[SIGN CHECK]  correct convention: "
                  f"{'u = u_nom + K·e' if eig_plus < 1 else 'u = u_nom − K·e'}")
            self._sign_checked = True
            self._step_counter = 0

        # ---- Line 5: first nominal MPC pass with current K_k ----------- #
        ok1, U1, Y1, _ = self._solve_nominal_mpc(
            y_bar_k, corridor_centers, corridor_radii, p_g, self.K_k
        )
        # ok1, U1, Y1, _ = self._solve_nominal_mpc(
        #     y_bar_k, corridor_H, corridor_h, p_g, self.K_k
        # )
        if not ok1:
            return False, None, None, None
        # if not ok1:
        #     try:
        #         from examples.planar_two_dof.diagnose_infeasible import diagnose_infeasible
        #         diagnose_infeasible(self, y_bar_k, corridor_H, corridor_h, p_g, self.K_k)
        #     except Exception as _e:
        #         print("  [diag] probe failed:", _e)
        #     return False, None, None, None

        # ---- Line 6: contractivity SDP (Eq. 22) at the nominal point --- #
        updated = self._update_tube(Y1[:, 0], U1[:, 0])
        # _update_tube sets self.K_k, self.lambda_k, self.rho_k, self._next_h_e_k

        # ---- Line 7: re-solve nominal MPC with the updated K_k --------- #
        ok2, U, Y, traj = self._solve_nominal_mpc(
            y_bar_k, corridor_centers, corridor_radii, p_g, self.K_k
        )
        # ok2, U, Y, traj = self._solve_nominal_mpc(
        #     y_bar_k, corridor_H, corridor_h, p_g, self.K_k
        # )
        if not ok2:
            # Fall back to the first-pass solution if the re-solve fails
            U, Y, traj = U1, Y1, Y1[:self.config_dim, :].T

        # ---- Line 8: apply control  u_k = u_bar[:,0] + K_k e_k --------- #
        e_k = y_k - y_bar_k
        # u_applied = U[:, 0] + self.K_k @ e_k
        fb = self.K_k @ e_k

        u_applied = U[:, 0] + fb

        # ---- Line 9: commit the updated tube scaling ------------------- #
        if updated:
            self.h_e_k = self._next_h_e_k.copy()

        # ---- Tube-collapse diagnostic --------------------------------- #
        self._step_counter += 1
        if self._step_counter % 50 == 0:
            print(f"  [tube] step={self._step_counter:4d}  "
                  f"h_e_min={self.h_e_k.min():.4e}  "
                  f"h_e_max={self.h_e_k.max():.4e}  "
                  f"lambda_k={self.lambda_k:.4f}  "
                  f"||K_k||={np.linalg.norm(self.K_k, 2):.1f}")

        # ---- Store solution for the simulator -------------------------- #
        self.last_y_bar = Y
        self.last_u_bar = U
        self.last_predicted_traj = traj

        y_bar_next = Y[:, 1]
        return True, u_applied, y_bar_next, traj

    # ================================================================== #
    #  Accessors used by the corridor simulator                          #
    # ================================================================== #
    def get_predicted_traj(self):
        """Returns the position trajectory shape (N+1, nq) for replanning."""
        return self.last_predicted_traj

    def get_solved_control(self, x, k=0):
        """The k-th nominal control of the last solve."""
        return self.last_u_bar[:, k]

    def get_predicted_state(self, k=1):
        return self.last_y_bar[:, k]

    def set_solver(self, solver):
        self.use_solver = solver