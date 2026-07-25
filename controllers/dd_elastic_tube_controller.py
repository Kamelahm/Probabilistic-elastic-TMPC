"""
Data-Driven Elastic Tube MPC Controller (Algorithm 1), region route.

"""

import numpy as np
import cvxpy as cp
import pickle


class DDElasticTubeController:

    # ------------------------------------------------------------------ #
    #  FEATURE FLAGS                                                      #
    # ------------------------------------------------------------------ #
    FIX_MEK         = True    # Remark 7 / Alg.1 line 10: M_ek <- lam_k M_ek
    FIX_LAM         = True    # lambda ceiling
    FIX_CORRIDOR    = True    # 2-norm tube radius (under-tightens if False)
    FIX_NESTING     = True    # [C5] (26g)-(26i)
    FIX_CONSISTENCY = True    # [C3] (23f), now cheap in psi form
    C_BAR_MARGIN    = 1.10
    SIGMA_W         = 1.0     # objective weight on lambda_k in (26a).
                              # Raise to buy contraction at the cost of ||K||.
    DEBUG           = True
    FIX_CONSISTENCY = True    # [C3] (23f), now cheap in psi form
    PIN_K           = True    # [C6] the facet template is built around K_des;


    # ================================================================== #
    def __init__(
        self,
        # --- region-route identification (Theorem 1 / Lemma 3) ---------- #
        A_bar, B_bar,                 # nominal = mid(F)
        Gamma_A, Gamma_B,             # mismatch half-widths
        h_wtilde, c_omega,            # confidence-region centre + half-widths
        # --- tube design (Theorem 2) ------------------------------------ #
        H_e, h_e_0,
        K_init, lambda_init, rho_init,
        P_lyap, P_sqrt, M_e_k,
        # --- MPC weights and constraints -------------------------------- #
        Q, R, Q_terminal, N,
        u_lim, v_lim, p_amp,
        c_bar_design=None,
        lam_max=0.99,
        use_solver=cp.CLARABEL,
    ):
        # --- model ---
        self.A_bar, self.B_bar = A_bar, B_bar
        self.n, self.m = A_bar.shape[0], B_bar.shape[1]
        self.config_dim = self.n // 2

        # --- uncertainty (region route) ---
        self.Gamma_A, self.Gamma_B = Gamma_A, Gamma_B
        self.h_wtilde, self.c_omega = h_wtilde, c_omega

        # --- tube ---
        self.H_e = H_e
        self.s_f = H_e.shape[0]
        self.P_lyap, self.P_sqrt = P_lyap, P_sqrt

        # --- MPC ---
        self.Q, self.R, self.Q_terminal, self.N = Q, R, Q_terminal, N
        self.u_lim, self.v_lim, self.p_amp = u_lim, v_lim, p_amp
        self.lam_max = float(lam_max)

        self.H_u = np.vstack([np.eye(self.m), -np.eye(self.m)])
        self.h_u = np.full(2 * self.m, u_lim)
        self.s_u = self.H_u.shape[0]

        # --- online state ---
        self.h_e_0 = np.asarray(h_e_0, float).copy()   # reference profile
        self.h_e_k = self.h_e_0.copy()
        self._h_scale = 1.0                            # [C4] h_e_k = scale*h_e_0
        self.K_k = K_init.copy()
        self.lambda_k = float(lambda_init)
        self.rho_k = float(rho_init)
        self.M_e_k = float(M_e_k)

        # --- [C2] precomputed facet quantities -------------------------- #
        absHe = np.abs(self.H_e)
        self.rowsA = absHe @ Gamma_A                   # |H_e^{l,:}| Gamma_A
        self.rowsB = absHe @ Gamma_B
        self.phi_A = np.linalg.norm(self.rowsA, axis=1)      # (19), L2
        self.phi_B = np.linalg.norm(self.rowsB, axis=1)
        self.additive = self.H_e @ c_omega + absHe @ h_wtilde  # (28)

        # --- [C4] support cache ----------------------------------------- #
        self._support_cache = {}
        self.use_solver = use_solver

        # --- certified input-facet bound cbar^{(-1)} -------------------- #
        if c_bar_design is None:
            c_bar_design = self._certify_gain(K_init, self.h_e_k)
            if c_bar_design is None:
                c_bar_design = np.array([
                    self._support(self.h_e_k, self.H_u[j] @ K_init)
                    for j in range(self.s_u)])
            c_bar_design = c_bar_design * self.C_BAR_MARGIN
        self.c_bar_design = np.asarray(c_bar_design, float)
        self.c_bar_prev = self.c_bar_design.copy()
        self.c_bar_k = self.c_bar_design.copy()

        self.last_y_bar = self.last_u_bar = self.last_predicted_traj = None
        self.A, self.B = self.A_bar, self.B_bar        # simulator aliases

    # ================================================================== #
    @classmethod
    def from_cached_dir(cls, p_cached_dir, Q, R, Q_terminal, N,
                        u_lim, v_lim, p_amp, lam_max=0.99):
        with open(p_cached_dir / "identified_model.pckl", "rb") as f:
            mod = pickle.load(f)
        with open(p_cached_dir / "tube_initial.pckl", "rb") as f:
            tube = pickle.load(f)
        return cls(
            A_bar=tube["A_bar"], B_bar=tube["B_bar"],          # [C1]
            Gamma_A=mod["Gamma_A"], Gamma_B=mod["Gamma_B"],
            h_wtilde=mod["h_wtilde"], c_omega=mod["c_omega"],
            H_e=tube["H_e"], h_e_0=tube["h_e_0"],
            K_init=tube["K_k"], lambda_init=tube["lambda_k"],
            rho_init=tube["rho_k"],
            P_lyap=tube["P_lyap"], P_sqrt=tube["P_sqrt"],
            M_e_k=tube["M_e_k"],
            Q=Q, R=R, Q_terminal=Q_terminal, N=N,
            u_lim=u_lim, v_lim=v_lim, p_amp=p_amp, lam_max=lam_max,
        )

    # ================================================================== #
    #  [C4] Support function of E = { e : H_e e <= h },  computed by LP    #
    #       once at h_e_0 and rescaled (h_e_k is always a multiple of it). #
    # ================================================================== #
    def _support(self, h_vec, direction):
        scale = float(h_vec[0] / self.h_e_0[0]) if self.h_e_0[0] != 0 else 1.0
        key = np.asarray(direction, float).tobytes()
        if key not in self._support_cache:
            mu = cp.Variable(self.s_f, nonneg=True)
            prob = cp.Problem(cp.Minimize(self.h_e_0 @ mu),
                              [self.H_e.T @ mu == direction])
            try:
                prob.solve(solver=self.use_solver)
            except Exception:
                return np.inf
            if mu.value is None:
                return np.inf
            self._support_cache[key] = float(self.h_e_0 @ mu.value)
        return scale * self._support_cache[key]

    def _tube_radius_position(self, h_vec):
        """max_{e in E} ||e_pos||_2 -- what the SOC corridor constraint needs."""
        nq = self.config_dim
        r = np.zeros(nq)
        for d in range(nq):
            e_d = np.zeros(self.n); e_d[d] = 1.0
            r[d] = max(self._support(h_vec, e_d), self._support(h_vec, -e_d))
        if self.FIX_CORRIDOR:
            return float(np.linalg.norm(r, 2))
        return float(r.max())

    def _certify_gain(self, K, h_vec):
        """(26g)-(26i) for a GIVEN K:  M_u H_e = H_u K, M_u >= 0, cbar = M_u h."""
        M_u = cp.Variable((self.s_u, self.s_f), nonneg=True)
        prob = cp.Problem(cp.Minimize(cp.sum(M_u @ h_vec)),
                          [M_u @ self.H_e == self.H_u @ K])
        try:
            prob.solve(solver=self.use_solver)
        except Exception:
            return None
        return None if M_u.value is None else M_u.value @ h_vec

    # ================================================================== #
    #  psi, the nominal-dependent facet term (20)                         #
    # ================================================================== #
    def _psi(self, y_bar, u_bar):
        return self.rowsA @ np.abs(y_bar) + self.rowsB @ np.abs(u_bar)

    # ================================================================== #
    #  Algorithm 1 lines 5 / 7: nominal MPC (23)                          #
    # ================================================================== #
    def _solve_nominal_mpc(self, y0, corridor_centers, corridor_radii, p_g,
                           c_bar, d_k=None):
        n, m, N = self.n, self.m, self.N
        nq = self.config_dim
        h = self.h_e_k

        Y_bar = cp.Variable((n, N + 1))
        U_bar = cp.Variable((m, N))
        x_g_v = np.concatenate([p_g, np.zeros(nq)])

        cost = 0
        cons = [Y_bar[:, 0] == y0]

        v_tp = np.zeros(nq); v_tn = np.zeros(nq)
        for j in range(nq):
            row = np.zeros(n); row[nq + j] = 1.0
            v_tp[j] = self._support(h, row)
            row = np.zeros(n); row[nq + j] = -1.0
            v_tn[j] = self._support(h, row)

        pos_radius = self._tube_radius_position(h)
        if self.DEBUG and not hasattr(self, "_corr_reported"):
            print(f"  [MPC] pos_radius={pos_radius:.4e}  corridor radii: "
                  f"min={np.min(corridor_radii):.4e} max={np.max(corridor_radii):.4e}")
            self._corr_reported = True

        for j in range(N):
            cost += cp.quad_form(Y_bar[:, j] - x_g_v, self.Q) \
                + cp.quad_form(U_bar[:, j], self.R)
            cons += [Y_bar[:, j + 1] == self.A_bar @ Y_bar[:, j]
                     + self.B_bar @ U_bar[:, j]]                     # (23b)
            if j >= 1:                                               # (23c)
                r_eff = corridor_radii[j] - pos_radius
                if r_eff > 1e-6:
                    cons += [cp.norm(Y_bar[:nq, j] - corridor_centers[j], 2)
                             <= r_eff]
                else:
                    if self.DEBUG:
                        print(f"  [MPC] stage {j}: r_eff={r_eff:.4e} <= 0 "
                              f"-- TUBE EXCEEDS CORRIDOR")
                    return False, None, None, None
            cons += [Y_bar[nq:, j] <= self.v_lim - v_tp,
                     Y_bar[nq:, j] >= -self.v_lim + v_tn]
            cons += [self.H_u @ U_bar[:, j] <= self.h_u - c_bar]     # (23d)

        cost += cp.quad_form(Y_bar[:, N] - x_g_v, self.Q_terminal)
        r_eff_N = corridor_radii[N] - pos_radius
        if r_eff_N > 1e-6:
            cons += [cp.norm(Y_bar[:nq, N] - corridor_centers[N], 2) <= r_eff_N]
        else:
            if self.DEBUG:
                print(f"  [MPC] terminal stage {N}: r_eff={r_eff_N:.4e} <= 0 "
                      f"-- TUBE EXCEEDS CORRIDOR")
            return False, None, None, None
        cons += [Y_bar[nq:, N] == np.zeros(nq)]

        # ---- [C3] consistency constraint (23f) in psi form -------------- #
        if d_k is not None and self.FIX_CONSISTENCY:
            zy = cp.Variable(n, nonneg=True)
            zu = cp.Variable(m, nonneg=True)
            cons += [zy >= Y_bar[:, 0], zy >= -Y_bar[:, 0],
                     zu >= U_bar[:, 0], zu >= -U_bar[:, 0],
                     self.rowsA @ zy + self.rowsB @ zu <= d_k + 1e-9]

        prob = cp.Problem(cp.Minimize(cost), cons)
        try:
            prob.solve(solver=self.use_solver)
        except Exception:
            return False, None, None, None
        if Y_bar.value is None or U_bar.value is None:
            return False, None, None, None
        return True, U_bar.value, Y_bar.value, Y_bar.value[:nq, :].T

    # ================================================================== #
    #  Algorithm 1 line 6: contractive tube (26)                          #
    # ================================================================== #
    def _update_tube(self, y_bar_k, u_bar_k):
        s_f = self.s_f
        h_e = self.h_e_k
        psi_k = self._psi(y_bar_k, u_bar_k)              # [C2]

        P_k = cp.Variable((s_f, s_f), nonneg=True)
        K_k = cp.Variable((self.m, self.n))
        rho_k = cp.Variable(nonneg=True)
        lam_k = cp.Variable(nonneg=True)
        h_next = cp.Variable(s_f, nonneg=True)
        M_u = cp.Variable((self.s_u, s_f), nonneg=True)

        cons = [
            P_k @ h_e <= (cp.multiply(lam_k, h_e)                    # (26b)
                          - self.additive
                          - self.M_e_k * self.phi_A
                          - rho_k * self.M_e_k * self.phi_B
                          - psi_k),
            P_k @ self.H_e == self.H_e @ (self.A_bar + self.B_bar @ K_k),  # (26c)
            h_next == cp.multiply(lam_k, h_e),                       # (26d)
            cp.norm(K_k, 2) <= rho_k,                                # (26f)
            lam_k <= (self.lam_max if self.FIX_LAM else 0.999),
            lam_k >= 0.001,
        ]

        if self.PIN_K:                                               # [C6]
            cons.append(K_k == self.K_k)

        if self.FIX_NESTING:                                         # (26g)-(26i)
            cons += [M_u @ self.H_e == self.H_u @ K_k,
                     M_u @ h_e <= self.c_bar_prev]

        # predicted floor on lambda from the disturbance budget alone: if this
        # already exceeds lam_max the SDP CANNOT succeed at this nominal pair.
        budget = (self.additive + self.M_e_k * self.phi_A
                  + self.rho_k * self.M_e_k * self.phi_B + psi_k)
        self._last_budget_frac = float((budget / h_e).max())

        prob = cp.Problem(cp.Minimize(rho_k + self.SIGMA_W * lam_k), cons)
        try:
            prob.solve(solver=self.use_solver)
        except Exception as exc:
            self._hold_tube(f"solver raised {type(exc).__name__}"); return False
        if (K_k.value is None
                or prob.status not in ("optimal", "optimal_inaccurate")):
            self._hold_tube(f"status={prob.status}"); return False

        self.K_k = K_k.value
        self.lambda_k = float(lam_k.value)
        self.rho_k = float(rho_k.value)

        if self.FIX_NESTING and M_u.value is not None:
            self.c_bar_k = np.maximum(M_u.value @ h_e, 0.0)
        else:
            self.c_bar_k = np.array([
                self._support(h_e, self.H_u[j] @ self.K_k)
                for j in range(self.s_u)])

        self._next_h_e_k = h_next.value
        self._next_M_e_k = (self.lambda_k * self.M_e_k
                            if self.FIX_MEK else self.M_e_k)     # Remark 7
        self._d_k = psi_k                                        # (23f) RHS
        self._n_tube_ok = getattr(self, "_n_tube_ok", 0) + 1
        return True

    def _hold_tube(self, reason="unknown"):
        """Tube update failed: keep the previous tube.  NOTE self.lambda_k then
        retains its previous value, so any printed lambda is STALE."""
        self._next_h_e_k = self.h_e_k.copy()
        self._next_M_e_k = self.M_e_k
        self._d_k = None
        self._n_tube_fail = getattr(self, "_n_tube_fail", 0) + 1
        self._last_fail_reason = reason


    # ================================================================== #
    #  One MPC step (Algorithm 1, lines 4-10)                             #
    # ================================================================== #
    def step(self, y_k, y_bar_k, corridor_centers, corridor_radii, p_g):
        if not hasattr(self, "_step_counter"):
            eig_p = np.max(np.abs(np.linalg.eigvals(
                self.A_bar + self.B_bar @ self.K_k)))
            print(f"[SIGN CHECK] rho(A+BK) = {eig_p:.4f} -> u = u_nom + K e  "
                  f"({'OK' if eig_p < 1 else 'WRONG SIGN'})")
            self._step_counter = 0

        ok1, U1, Y1, _ = self._solve_nominal_mpc(                    # line 5
            y_bar_k, corridor_centers, corridor_radii, p_g, self.c_bar_prev)
        if not ok1:
            if self.DEBUG:
                self._diagnose(y_bar_k, corridor_centers, corridor_radii, p_g)
            return False, None, None, None

        updated = self._update_tube(Y1[:, 0], U1[:, 0])              # line 6

        if updated:                                                  # line 7
            ok2, U, Y, traj = self._solve_nominal_mpc(
                y_bar_k, corridor_centers, corridor_radii, p_g,
                self.c_bar_k, d_k=self._d_k)
            if not ok2:
                U, Y, traj = U1, Y1, Y1[:self.config_dim, :].T
        else:
            U, Y, traj = U1, Y1, Y1[:self.config_dim, :].T

        e_k = y_k - y_bar_k                                          # line 9
        u_applied = U[:, 0] + self.K_k @ e_k

        if updated:                                                  # line 10
            self.h_e_k = self._next_h_e_k.copy()
            self._h_scale *= self.lambda_k          # [C4] keeps supports exact
            self.M_e_k = self._next_M_e_k
            self.c_bar_prev = self.c_bar_k.copy()

        self.last_y_bar, self.last_u_bar = Y, U
        self.last_predicted_traj = traj
        return True, u_applied, Y[:, 1], traj


    # ================================================================== #
    def get_predicted_traj(self):
        return self.last_predicted_traj

    def get_solved_control(self, x, k=0):
        return self.last_u_bar[:, k]

    def get_predicted_state(self, k=1):
        return self.last_y_bar[:, k]

    def set_solver(self, solver):
        self.use_solver = solver