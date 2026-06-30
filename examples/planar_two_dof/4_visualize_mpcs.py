# Copyright (c) 2025, ABB Schweiz AG
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


# import argparse
# import pickle

# import numpy as np
# from matplotlib.collections import PatchCollection
# from matplotlib.patches import Circle

# import matplotlib.pyplot as plt

# from examples.planar_two_dof import P_EXAMPLE_2_DOF
# from examples.planar_two_dof.world import DiskWorld

# fig, (ax1, ax2) = plt.subplots(1, 2)


# def render_nominal(world, ax1, ax2, inputs, outputs):
#     (x, x_g_v, cs, rs) = inputs
#     (X, U) = outputs
#     q, _ = np.split(x, 2)
#     world.render_world_space(ax1, q=q)
#     world.render_configuration(ax1, q=q_g, color="g")
#     world.render_configuration_space(ax2, q=q)
#     ax = ax2
#     ax.scatter(*x[:world.config_dim], c="tab:blue")
#     ax.scatter(*q_g, c="tab:green")
#     ax.plot(X[0], X[1], marker=".", color="r")
#     ax.plot(path_centers[:, 0], path_centers[:, 1], color="k")
#     ax.scatter(x_g_v[0], x_g_v[1])
#     ax.add_collection(
#         PatchCollection(
#             [Circle(c, r) for c, r in zip(cs.T, rs)]
#             , facecolors="none", edgecolors="k"
#         )
#     )



# def render_rigid_tube(world, ax1, ax2, inputs, outputs):
#     (x, z_g, cs, rs) = inputs
#     (Z, V, r_p) = outputs
#     q, _ = np.split(x, 2)
#     world.render_world_space(ax1, q=q)
#     world.render_configuration(ax1, q=q_g, color="g")
#     world.render_configuration_space(ax2, q=q)
#     ax = ax2
#     ax.scatter(*x[:world.config_dim], c="tab:blue")
#     ax.scatter(*q_g, c="tab:green")
#     ax.plot(Z[0], Z[1], marker=".", color="r")
#     ax.plot(path_centers[:, 0], path_centers[:, 1], color="k")
#     ax.scatter(z_g[0], z_g[1])
#     ax.add_collection(
#         PatchCollection(
#             [Circle(c, r_p) for c in Z.T], facecolors="none", edgecolors="r"
#         )
#     )
#     ax.add_collection(
#         PatchCollection(
#             [Circle(c, r) for c, r in zip(cs.T, rs)]
#             , facecolors="none", edgecolors="k"
#         )
#     )


# def render_flexible_tube(world, ax1, ax2, inputs, outputs):
#     (x, z_g, cs, rs, s_0) = inputs
#     (Z, U, S, r_p) = outputs
#     q, _ = np.split(x, 2)
#     world.render_world_space(ax1, q=q)
#     world.render_configuration(ax1, q=q_g, color="g")
#     world.render_configuration_space(ax2, q=q)
#     ax = ax2
#     ax.scatter(*x[:world.config_dim], c="tab:blue")
#     ax.scatter(*q_g, c="tab:green")
#     ax.plot(Z[0], Z[1], marker=".", color="r")
#     ax.plot(path_centers[:, 0], path_centers[:, 1], color="k")
#     ax.scatter(z_g[0], z_g[1])
#     ax.add_collection(
#         PatchCollection(
#             [Circle(c, r_p * s) for s, c in zip(S, Z.T)], facecolors="none", edgecolors="r"
#         )
#     )
#     ax.add_collection(
#         PatchCollection(
#             [Circle(c, r) for c, r in zip(cs.T, rs)]
#             , facecolors="none", edgecolors="k"
#         )
#     )

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     names = (
#         "nom_star",
#         "rt",
#         "ft"
#     )
#     parser.add_argument("--method", type=str, default="nom_star", choices=names)
#     args = parser.parse_args()
#     f_name = args.method
#     world = DiskWorld.from_example()
#     path_data = P_EXAMPLE_2_DOF / "data" / "dof_2_ef_0.1"
#     paths = list(sorted((path_data / "motion").glob(f"{f_name}*.pckl")))
#     for p in paths:
#         _, nr = p.stem.rsplit("~", 1)
#         nr = int(nr)
#         data_corr = np.load(path_data / "corridors" / f"corr_{nr}.npz")
#         path_centers = data_corr["path_centers"]
#         with p.open("rb") as fp:
#             all_results = pickle.load(fp)
#         q_s, q_g = path_centers[[0, -1]]
#         path_radii = world.sdf(path_centers)
#         for results in all_results:
#             if f_name == "nom_star" or f_name =="nom":
#                 render_nominal(world, ax1, ax2, results.inputs, results.outputs)
#             elif f_name == "rt":
#                 render_rigid_tube(world, ax1, ax2, results.inputs, results.outputs)
#             elif f_name == "ft":
#                 render_flexible_tube(world, ax1, ax2, results.inputs, results.outputs)
#             plt.pause(.01)
#             for ax in (ax1, ax2):
#                 ax.cla()


import argparse
import pickle
from itertools import combinations

import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Polygon as MplPolygon
from scipy.spatial import ConvexHull

import matplotlib.pyplot as plt

from examples.planar_two_dof import P_EXAMPLE_2_DOF
from examples.planar_two_dof.world import DiskWorld
# Needed so pickle can reconstruct the dd_elastic results objects:
from corridor_simulators.dd_elastic_tube import DDElasticTubeResults

fig, (ax1, ax2) = plt.subplots(1, 2)


def render_nominal(world, ax1, ax2, inputs, outputs):
    (x, x_g_v, cs, rs) = inputs
    (X, U) = outputs
    q, _ = np.split(x, 2)
    world.render_world_space(ax1, q=q)
    world.render_configuration(ax1, q=q_g, color="g")
    world.render_configuration_space(ax2, q=q)
    ax = ax2
    ax.scatter(*x[:world.config_dim], c="tab:blue")
    ax.scatter(*q_g, c="tab:green")
    ax.plot(X[0], X[1], marker=".", color="r")
    ax.plot(path_centers[:, 0], path_centers[:, 1], color="k")
    ax.scatter(x_g_v[0], x_g_v[1])
    ax.add_collection(
        PatchCollection(
            [Circle(c, r) for c, r in zip(cs.T, rs)]
            , facecolors="none", edgecolors="k"
        )
    )


def render_rigid_tube(world, ax1, ax2, inputs, outputs):
    (x, z_g, cs, rs) = inputs
    (Z, V, r_p) = outputs
    q, _ = np.split(x, 2)
    world.render_world_space(ax1, q=q)
    world.render_configuration(ax1, q=q_g, color="g")
    world.render_configuration_space(ax2, q=q)
    ax = ax2
    ax.scatter(*x[:world.config_dim], c="tab:blue")
    ax.scatter(*q_g, c="tab:green")
    ax.plot(Z[0], Z[1], marker=".", color="r")
    ax.plot(path_centers[:, 0], path_centers[:, 1], color="k")
    ax.scatter(z_g[0], z_g[1])
    ax.add_collection(
        PatchCollection(
            [Circle(c, r_p) for c in Z.T], facecolors="none", edgecolors="r"
        )
    )
    ax.add_collection(
        PatchCollection(
            [Circle(c, r) for c, r in zip(cs.T, rs)]
            , facecolors="none", edgecolors="k"
        )
    )


def render_flexible_tube(world, ax1, ax2, inputs, outputs):
    (x, z_g, cs, rs, s_0) = inputs
    (Z, U, S, r_p) = outputs
    q, _ = np.split(x, 2)
    world.render_world_space(ax1, q=q)
    world.render_configuration(ax1, q=q_g, color="g")
    world.render_configuration_space(ax2, q=q)
    ax = ax2
    ax.scatter(*x[:world.config_dim], c="tab:blue")
    ax.scatter(*q_g, c="tab:green")
    ax.plot(Z[0], Z[1], marker=".", color="r")
    ax.plot(path_centers[:, 0], path_centers[:, 1], color="k")
    ax.scatter(z_g[0], z_g[1])
    ax.add_collection(
        PatchCollection(
            [Circle(c, r_p * s) for s, c in zip(S, Z.T)], facecolors="none", edgecolors="r"
        )
    )
    ax.add_collection(
        PatchCollection(
            [Circle(c, r) for c, r in zip(cs.T, rs)]
            , facecolors="none", edgecolors="k"
        )
    )


def halfspaces_to_polygon(H, h):
    """Vertices of the 2D polytope {p : H p <= h}, ordered CCW.

    Intersects all facet pairs, keeps points satisfying every halfspace,
    then takes the convex hull. Returns (V, 2) array or None if degenerate.
    For the axis-aligned box corridors produced by corridor_polytopes.py this
    returns the four box corners."""
    H = np.asarray(H, float)
    h = np.asarray(h, float)
    F = H.shape[0]
    pts = []
    for i, j in combinations(range(F), 2):
        A = H[[i, j]]
        b = h[[i, j]]
        if abs(np.linalg.det(A)) < 1e-12:
            continue
        try:
            p = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            continue
        if np.all(H @ p <= h + 1e-7):
            pts.append(p)
    if len(pts) < 3:
        return None
    pts = np.array(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None
    return pts[hull.vertices]


def load_tube_He(path_data):
    """Load the (fixed) tube facet matrix H_e from the initial tube pickle.
    Only H_e is fixed across the run; the offsets h_e evolve per step and come
    from results.h_e."""
    with (path_data / "tube_initial.pckl").open("rb") as f:
        tube = pickle.load(f)
    return tube["H_e"]          # (2n, n)


def tube_polygon_from_h(H_e, h_e):
    """Project the polyhedral tube E_k = {e : H_e e <= h_e} onto the position
    plane for a given (per-step) offset vector h_e, and return its 2D vertices
    centered at the origin.

    Enumerates polytope vertices by intersecting all n-subsets of the facet
    hyperplanes, keeps the feasible ones, projects to position, and takes the
    convex hull. Falls back to a support-function sweep if vertex enumeration
    is degenerate."""
    H_e = np.asarray(H_e, float)
    h_e = np.asarray(h_e, float)
    n = H_e.shape[1]
    nq = n // 2
    s_f = H_e.shape[0]

    # --- Vertex enumeration in full state space ---
    verts = []
    for idx in combinations(range(s_f), n):
        A_sub = H_e[list(idx)]
        b_sub = h_e[list(idx)]
        if abs(np.linalg.det(A_sub)) < 1e-12:
            continue
        v = np.linalg.solve(A_sub, b_sub)
        if np.all(H_e @ v <= h_e + 1e-9):
            verts.append(v)

    if len(verts) >= 3:
        verts = np.array(verts)
        pos_verts = verts[:, :nq]
        try:
            hull = ConvexHull(pos_verts)
            poly = pos_verts[hull.vertices]
            if hull.volume > 1e-12:
                return poly
        except Exception:
            pass

    # --- Fallback: 2D support-function sweep on the position projection ---
    import cvxpy as cp
    thetas = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    poly = []
    e = cp.Variable(n)
    for th in thetas:
        d = np.zeros(n)
        d[0] = np.cos(th)
        d[1] = np.sin(th)
        prob = cp.Problem(cp.Maximize(d @ e), [H_e @ e <= h_e])
        prob.solve(solver=cp.CLARABEL)
        if e.value is not None:
            poly.append(e.value[:nq])
    if len(poly) >= 3:
        return np.array(poly)
    return None


def render_dd_elastic(world, ax1, ax2, results, H_e, tube_scale, traj_so_far):
    """Render one step of the dd_elastic trajectory with the box corridor
    (stage-0 polytope) and the per-step evolving polyhedral tube E_k built
    from results.h_e."""
    x = results.x_t
    q, _ = np.split(x, 2)
    world.render_world_space(ax1, q=q)
    world.render_configuration(ax1, q=q_g, color="g")
    world.render_configuration_space(ax2, q=q)
    ax = ax2
    ax.plot(path_centers[:, 0], path_centers[:, 1], color="k")
    ax.scatter(*x[:world.config_dim], c="tab:blue")
    ax.scatter(*q_g, c="tab:green")

    # --- Box corridor (stage-0 polytope) at this step ---
    if results.corr_H is not None and results.corr_h is not None:
        box = halfspaces_to_polygon(results.corr_H, results.corr_h)
        if box is not None:
            ax.add_patch(MplPolygon(
                box, closed=True, fill=False,
                edgecolor="tab:green", linestyle="--", alpha=0.8))

    if len(traj_so_far) > 1:
        tr = np.array(traj_so_far)
        ax.plot(tr[:, 0], tr[:, 1], color="r", marker=".", markersize=2)

    # --- Per-step evolving tube E_k, magnified, centered at current position ---
    if results.h_e is not None:
        tube_poly = tube_polygon_from_h(H_e, results.h_e)
        if tube_poly is not None:
            ax.add_patch(MplPolygon(
                tube_poly * tube_scale + x[:world.config_dim],
                closed=True, fill=False, edgecolor="r"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    names = (
        "nom_star",
        "rt",
        "ft",
        "dd_elastic",
    )
    parser.add_argument("--method", type=str, default="nom_star", choices=names)
    parser.add_argument("--save", action="store_true",
                        help="save a static trajectory figure per corridor")
    args = parser.parse_args()
    f_name = args.method

    TUBE_SCALE = 20.0      

    world = DiskWorld.from_example()
    path_data = P_EXAMPLE_2_DOF / "data" / "dof_2_ef_0.1"

    H_e = None
    if f_name == "dd_elastic":
        try:
            H_e = load_tube_He(path_data)
            print(f"Loaded tube H_e: {H_e.shape}")
        except Exception as ex:
            print(f"Warning: could not load tube H_e ({ex}); "
                  "tube will not be drawn.")

    paths = list(sorted((path_data / "motion").glob(f"{f_name}*.pckl")))
    for p in paths:
        _, nr = p.stem.rsplit("~", 1)
        nr = int(nr)
        data_corr = np.load(path_data / "corridors" / f"corr_{nr}.npz")
        path_centers = data_corr["path_centers"]
        with p.open("rb") as fp:
            all_results = pickle.load(fp)
        q_s, q_g = path_centers[[0, -1]]
        path_radii = world.sdf(path_centers)

        traj_so_far = []
        for results in all_results:
            if f_name == "nom_star" or f_name == "nom":
                render_nominal(world, ax1, ax2, results.inputs, results.outputs)
            elif f_name == "rt":
                render_rigid_tube(world, ax1, ax2, results.inputs, results.outputs)
            elif f_name == "ft":
                render_flexible_tube(world, ax1, ax2, results.inputs, results.outputs)
            elif f_name == "dd_elastic":
                traj_so_far.append(results.x_t[:world.config_dim])
                render_dd_elastic(world, ax1, ax2, results,
                                  H_e, TUBE_SCALE, traj_so_far)
            plt.pause(.01)
            for ax in (ax1, ax2):
                ax.cla()

        # Optional: save a static figure of the full trajectory for this corridor
        if args.save and f_name == "dd_elastic" and len(traj_so_far) > 1:
            fig_dir = path_data / "figures"
            fig_dir.mkdir(exist_ok=True)
            q_final, _ = np.split(all_results[-1].x_t, 2)
            world.render_configuration_space(ax2, q=q_final)
            ax2.plot(path_centers[:, 0], path_centers[:, 1], color="k",
                     label="corridor")

            # Box corridors sampled along the trajectory (the swept safe region)
            box_label_done = False
            for results in all_results[::20]:
                if results.corr_H is None or results.corr_h is None:
                    continue
                box = halfspaces_to_polygon(results.corr_H, results.corr_h)
                if box is not None:
                    ax2.add_patch(MplPolygon(
                        box, closed=True, fill=False,
                        edgecolor="tab:green", linestyle="--", alpha=0.3,
                        label=None if box_label_done else "box corridor"))
                    box_label_done = True

            tr = np.array(traj_so_far)
            ax2.plot(tr[:, 0], tr[:, 1], color="r", label="dd_elastic")

            # Per-step evolving tube sampled along the trajectory
            if H_e is not None:
                tube_label_done = False
                for results in all_results[::20]:
                    if results.h_e is None:
                        continue
                    tube_poly = tube_polygon_from_h(H_e, results.h_e)
                    if tube_poly is not None:
                        c = results.x_t[:world.config_dim]
                        ax2.add_patch(MplPolygon(
                            tube_poly * TUBE_SCALE + c,
                            closed=True, fill=False, edgecolor="r", alpha=0.4,
                            label=None if tube_label_done else "tube"))
                        tube_label_done = True

            ax2.scatter(*q_s, c="tab:blue", label="start")
            ax2.scatter(*q_g, c="tab:green", label="goal")
            ax2.legend(fontsize=7)
            ax2.set_title(f"dd_elastic - corridor {nr}  (tube x{TUBE_SCALE:.0f})")
            fig.savefig(fig_dir / f"dd_elastic_corridor_{nr}.png", dpi=150)
            for ax in (ax1, ax2):
                ax.cla()