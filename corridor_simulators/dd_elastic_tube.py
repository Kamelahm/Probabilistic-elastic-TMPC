"""
Simulator for the Data-Driven Elastic Tube MPC controller.
Drives DDElasticTubeController through an RRT-planned corridor,
paralleling FlexibleTubeSimulator.

Polyhedral-corridor version: the per-stage safe region is represented as a
convex polytope { p : H_c p <= h_c } (see corridor_polytopes.py) rather than a
ball.  The polytopes realize the output constraint set Y of the MPC problem,
restricted in position to the obstacle-free configuration space; see the
simulation remark in the paper.  The balls are still computed and used only to
pick the moving sub-goal via compute_goal_state.
"""
import time
import numpy as np

from aux import compute_goal_state, interpolate_equidistant
from corridor_simulators.base import BaseSimulator
from corridor_simulators.corridor_polytopes import get_corridor_polytopes


class DDElasticTubeSimulator(BaseSimulator):

    def __init__(self, cont, integrator, world=None, aux_controller_steps=1):
        super().__init__(cont, integrator, world)
        self.aux_controller_steps = aux_controller_steps

    def simulate(self, path_centers, path_radii,
                 nr_steps=1000, goal_tol=1e-2, verbose=False):
        cont       = self.cont
        world      = self.world
        integrator = self.integrator
        N          = cont.N

        m_x = cont.A.shape[0]          # state dim  (4)
        m_p = m_x // 2                 # position dim (2)

        p_g = path_centers[-1]         # final goal position
        x_g = np.r_[p_g, np.zeros(m_p)]
        x_0 = np.r_[path_centers[0], np.zeros(m_p)]

        # Reference trajectory used for corridor planning
        path_traj = x_0[:m_p][None].repeat(N + 1, axis=0)
        path_track = interpolate_equidistant(path_centers.copy(), delta=0.001)

        x      = x_0.copy()            # real (perturbed) state
        y_bar  = x_0.copy()            # nominal state
        ts     = 0
        status = "not_in_goal"
        all_results = []

        while ts < nr_steps:
            t_start = time.time()

            # ---- Corridor planning ------------------------------------- #
            # Balls: used only to select the moving sub-goal.
            cs, rs = self.get_corridor_balls(
                path_traj, path_centers, path_radii, p_g)
            # Polytopes: the safe regions actually enforced by the MPC.
            # Position is unbounded in Y (H_y_pos / h_y_pos = None); the corridor
            # is the only position constraint, and is contained in the
            # obstacle-free configuration space by construction.
            corr_H, corr_h, corr_centers = get_corridor_polytopes(
                self, path_traj, path_centers, path_radii, p_g,
                H_y_pos=None, h_y_pos=None)

            # Moving sub-goal: farthest reachable point along the corridor
            x_g_v, _ = compute_goal_state(
                cs, rs, path_track, p_g, return_index=True)
            p_g_local = x_g_v[:m_p]

            t_corridor = time.time() - t_start

            # ---- One step of Algorithm 1 -------------------------------- #
            t_mpc_s = time.time()
            try:
                # ok, u_applied, y_bar_next, predicted_traj = cont.step(
                #     y_k=x,
                #     y_bar_k=y_bar,
                #     corridor_H=corr_H,
                #     corridor_h=corr_h,
                #     p_g=p_g_local,              # moving sub-goal
                # )
                ok, u_applied, y_bar_next, predicted_traj = cont.step(
                    y_k=x,
                    y_bar_k=y_bar,
                    corridor_centers=cs,
                    corridor_radii=rs,
                    p_g=p_g_local,
                )
            except Exception as e:
                if verbose:
                    print(f"  step exception: {e}")
                ok = False
            t_mpc = time.time() - t_mpc_s

            if not ok:
                status = "infeasible"
                all_results.append(DDElasticTubeResults(
                    cont, x_t=x.copy(),
                    times=(time.time() - t_start, t_corridor, t_mpc)))
                break

            # ---- Advance the real nonlinear system --------------------- #
            x = integrator.solve_time_step(x, u_applied)
            # y_bar = x.copy()
            y_bar = y_bar_next
            ts += 1

            all_results.append(DDElasticTubeResults(
                cont, x_t=x.copy(),
                times=(time.time() - t_start, t_corridor, t_mpc),
                u=u_applied.copy(),
                h_e=cont.h_e_k.copy(),
                lambda_k=cont.lambda_k,
                corr_H=corr_H[0], corr_h=corr_h[0]))   # stage-0 corridor at this step

            # ---- Replan corridor reference for next step --------------- #
            X_shifted = self.get_shifted_trajectory(
                predicted_traj.T, nr_shifts=self.aux_controller_steps)
            path_traj = X_shifted[:, :m_p]

            # ---- Collision check (6-DOF only; world is None for 2-DOF) -- #
            if world is not None:
                if self.is_path_in_collision(path_traj):
                    status = "collision"
                    break

            # # ---- Goal check -------------------------------------------- #
            # if np.linalg.norm(x - x_g) < goal_tol:
            #     status = "success"
            #     break

            pos_err = np.linalg.norm(x[:m_p] - p_g)
            vel_err = np.linalg.norm(x[m_p:])
            if pos_err < 0.05 and vel_err < 0.02:
                status = "success"
                break

            # ---- Diagnostic: print progress every 50 steps ----
            if ts % 50 == 0:
                dist_to_goal_pos = np.linalg.norm(x[:m_p] - p_g)
                dist_y_bar = np.linalg.norm(y_bar[:m_p] - p_g)
                err_norm = np.linalg.norm(x - y_bar)
                print(f"  ts={ts:4d}  "
                      f"||x_pos-p_g||={dist_to_goal_pos:.4f}  "
                      f"||y_bar_pos-p_g||={dist_y_bar:.4f}  "
                      f"||e||={err_norm:.4f}  "
                      f"||u||={np.linalg.norm(u_applied):.2f}")

            self.step_write(ts, x, x_g, time.time() - t_start,
                            status, write=verbose)

        self.end_write(ts, x, x_g, status, write=verbose)
        return (status, ts), all_results


class DDElasticTubeResults:
    """Lightweight per-step result container."""
    def __init__(self, cont, x_t, times, u=None, h_e=None, lambda_k=None,
                 corr_H=None, corr_h=None):
        self.x_t      = x_t
        self.times    = times
        self.u        = u
        self.h_e      = h_e
        self.lambda_k = lambda_k
        self.corr_H   = corr_H        # stage-0 corridor facet normals (F, nq)
        self.corr_h   = corr_h        # stage-0 corridor facet offsets (F,)