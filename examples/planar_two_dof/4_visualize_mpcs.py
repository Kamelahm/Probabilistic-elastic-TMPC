import argparse
import pickle
from itertools import combinations

import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Polygon as MplPolygon
from scipy.spatial import ConvexHull
from scipy.optimize import linprog                     # [V1]

import matplotlib.pyplot as plt

from examples.planar_two_dof import P_EXAMPLE_2_DOF
from examples.planar_two_dof.world import DiskWorld
# Needed so pickle can reconstruct the dd_elastic results objects:
from corridor_simulators.dd_elastic_tube import DDElasticTubeResults

fig, (ax1, ax2) = plt.subplots(1, 2)

# --------------------------------------------------------------------------- #
#  [V3] With the region-route tube the position half-width is ~0.010 rad and   #
#  the corridor radii are ~0.063, i.e. ~15% -- already visible.  The old 20x   #
#  magnification drew a tube 3.6x WIDER than the corridor.                     #
# --------------------------------------------------------------------------- #
TUBE_SCALE = 1.0            # 1.0 = true size
N_DIRS = 64                 # directions in the support sweep
VERTEX_ENUM_MAX_SF = 12     # above this facet count, never enumerate vertices


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
    """Vertices of the 2D polytope {p : H p <= h}, ordered CCW."""
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


def load_tube(path_data):
    """[V2] Return (H_e, h_e_0) from the initial tube pickle.  H_e is fixed for
    the whole run; h_e evolves per step and comes from results.h_e."""
    with (path_data / "tube_initial.pckl").open("rb") as f:
        tube = pickle.load(f)
    return tube["H_e"], np.asarray(tube["h_e_0"], float)


# --------------------------------------------------------------------------- #
#  [V1] Tube projection.                                                       #
#  Vertex enumeration is O(comb(s_f, n)): 70 subsets at s_f = 2n = 8, but      #
#  13,633,830 at s_f = 136 (J = 16 augmented template), i.e. tens of seconds   #
#  per frame -- the renderer hangs rather than fails.  Use a support-function  #
#  sweep (n vars, s_f rows) instead whenever the facet count is large.         #
# --------------------------------------------------------------------------- #
def _support_point(H_e, h_e, d):
    """argmax { d'e : H_e e <= h_e }."""
    res = linprog(-np.asarray(d, float), A_ub=H_e, b_ub=h_e,
                  bounds=[(None, None)] * H_e.shape[1], method="highs")
    return res.x if res.success else None


def tube_polygon_from_h(H_e, h_e, n_dirs=N_DIRS):
    """Project E_k = {e : H_e e <= h_e} onto the position plane, centred at 0."""
    H_e = np.asarray(H_e, float)
    h_e = np.asarray(h_e, float)
    n = H_e.shape[1]
    nq = n // 2
    s_f = H_e.shape[0]

    if s_f <= VERTEX_ENUM_MAX_SF:                        # original path
        verts = []
        for idx in combinations(range(s_f), n):
            A_sub, b_sub = H_e[list(idx)], h_e[list(idx)]
            if abs(np.linalg.det(A_sub)) < 1e-12:
                continue
            v = np.linalg.solve(A_sub, b_sub)
            if np.all(H_e @ v <= h_e + 1e-9):
                verts.append(v)
        if len(verts) >= 3:
            pos = np.array(verts)[:, :nq]
            try:
                hull = ConvexHull(pos)
                if hull.volume > 1e-12:
                    return pos[hull.vertices]
            except Exception:
                pass

    pts = []
    for th in np.linspace(0.0, 2.0 * np.pi, n_dirs, endpoint=False):
        d = np.zeros(n)
        d[0], d[1] = np.cos(th), np.sin(th)
        e = _support_point(H_e, h_e, d)
        if e is not None:
            pts.append(e[:nq])
    if len(pts) < 3:
        return None
    pts = np.array(pts)
    try:
        hull = ConvexHull(pts)
        return pts[hull.vertices]
    except Exception:
        return pts


class TubeShapeCache:
    """[V2] The tube update is h_{k+1} = lambda_k h_k, so h_e is always a scalar
    multiple of h_e_0 and the projected polygon only SCALES.  Compute the shape
    once (0.17 s) and rescale per frame (0.04 ms).  Falls back to a per-frame
    solve, with a warning, if the ratio is not uniform."""

    def __init__(self, H_e, h_e_ref):
        self.H_e = np.asarray(H_e, float)
        self.h_ref = np.asarray(h_e_ref, float)
        self.poly_ref = tube_polygon_from_h(self.H_e, self.h_ref)
        self._warned = False

    def polygon(self, h_e):
        h_e = np.asarray(h_e, float)
        if self.poly_ref is None:
            return None
        ratio = h_e / np.where(self.h_ref == 0.0, np.inf, self.h_ref)
        finite = ratio[np.isfinite(ratio)]
        if finite.size and np.allclose(finite, finite[0], rtol=1e-6, atol=0.0):
            return self.poly_ref * float(finite[0])
        if not self._warned:
            print("  [viz] h_e is not a uniform multiple of h_e_0; "
                  "recomputing the tube polygon per frame (slower).")
            self._warned = True
        return tube_polygon_from_h(self.H_e, h_e)


def plot_he_trace(all_results, out_path, title="tube contraction"):
    """[V4] h_e(max) and the applied lambda_k against the step index.
    Geometric decay then flattening = the tube floor of Remark 4.
    A flat line = the tube update never fired (static tube)."""
    he = [np.max(r.h_e) for r in all_results if getattr(r, "h_e", None) is not None]
    if len(he) < 2:
        print("  [viz] no h_e history in the results; skipping the trace plot.")
        return
    he = np.array(he)
    lam = he[1:] / np.maximum(he[:-1], 1e-300)

    f2, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 5.0), sharex=True)
    a1.semilogy(he, color="tab:red")
    a1.set_ylabel(r"$\max_l\ h_{e_k}^l$")
    a1.grid(alpha=0.3)
    a1.set_title(title)
    a2.plot(lam, color="tab:blue", lw=0.9)
    a2.axhline(1.0, color="k", lw=0.8, ls=":")
    a2.set_ylabel(r"applied $\lambda_k$")
    a2.set_xlabel("step $k$")
    a2.grid(alpha=0.3)
    n_contract = int((lam < 1.0 - 1e-9).sum())
    a2.text(0.02, 0.06,
            f"contracting on {n_contract}/{len(lam)} steps "
            f"({100 * n_contract / len(lam):.0f}%);  "
            f"h_e {he[0]:.3e} -> {he[-1]:.3e}",
            transform=a2.transAxes, fontsize=8)
    f2.tight_layout()
    f2.savefig(out_path, dpi=150)
    plt.close(f2)
    print(f"  [viz] tube trace written to {out_path}")


def render_dd_elastic(world, ax1, ax2, results, tube_cache, tube_scale,
                      traj_so_far):
    """One step of the dd_elastic trajectory: box corridor (stage-0 polytope)
    plus the per-step evolving polyhedral tube E_k from results.h_e."""
    x = results.x_t
    q, _ = np.split(x, 2)
    world.render_world_space(ax1, q=q)
    world.render_configuration(ax1, q=q_g, color="g")
    world.render_configuration_space(ax2, q=q)
    ax = ax2
    ax.plot(path_centers[:, 0], path_centers[:, 1], color="k")
    ax.scatter(*x[:world.config_dim], c="tab:blue")
    ax.scatter(*q_g, c="tab:green")

    if results.corr_H is not None and results.corr_h is not None:
        box = halfspaces_to_polygon(results.corr_H, results.corr_h)
        if box is not None:
            ax.add_patch(MplPolygon(
                box, closed=True, fill=False,
                edgecolor="tab:green", linestyle="--", alpha=0.8))

    if len(traj_so_far) > 1:
        tr = np.array(traj_so_far)
        ax.plot(tr[:, 0], tr[:, 1], color="r", marker=".", markersize=2)

    if tube_cache is not None and results.h_e is not None:
        tube_poly = tube_cache.polygon(results.h_e)        # [V2]
        if tube_poly is not None:
            ax.add_patch(MplPolygon(
                tube_poly * tube_scale + x[:world.config_dim],
                closed=True, fill=False, edgecolor="r"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    names = ("nom_star", "rt", "ft", "dd_elastic")
    parser.add_argument("--method", type=str, default="nom_star", choices=names)
    parser.add_argument("--save", action="store_true",
                        help="save a static trajectory figure per corridor")
    parser.add_argument("--tube-scale", type=float, default=TUBE_SCALE,
                        help="magnification of the drawn tube (1.0 = true size)")
    args = parser.parse_args()
    f_name = args.method
    tube_scale = args.tube_scale

    world = DiskWorld.from_example()
    path_data = P_EXAMPLE_2_DOF / "data" / "dof_2_ef_0.1"

    tube_cache = None
    if f_name == "dd_elastic":
        try:
            H_e, h_e_0 = load_tube(path_data)
            tube_cache = TubeShapeCache(H_e, h_e_0)         # [V2] one solve
            hw = (np.abs(tube_cache.poly_ref).max()
                  if tube_cache.poly_ref is not None else float("nan"))
            print(f"Loaded tube H_e: {H_e.shape};  position half-width "
                  f"{hw:.4f} rad  (drawn at x{tube_scale:g})")
        except Exception as ex:
            print(f"Warning: could not load the tube ({ex}); "
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
                                  tube_cache, tube_scale, traj_so_far)
            plt.pause(.01)
            for ax in (ax1, ax2):
                ax.cla()

        if args.save and f_name == "dd_elastic" and len(traj_so_far) > 1:
            fig_dir = path_data / "figures"
            fig_dir.mkdir(exist_ok=True)

            # [V4] tube contraction trace for this corridor
            plot_he_trace(all_results, fig_dir / f"dd_elastic_he_{nr}.png",
                          title=f"tube contraction - corridor {nr}")

            q_final, _ = np.split(all_results[-1].x_t, 2)
            world.render_configuration_space(ax2, q=q_final)
            ax2.plot(path_centers[:, 0], path_centers[:, 1], color="k",
                     label="corridor")

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

            if tube_cache is not None:
                tube_label_done = False
                for results in all_results[::20]:
                    if results.h_e is None:
                        continue
                    tube_poly = tube_cache.polygon(results.h_e)
                    if tube_poly is not None:
                        c = results.x_t[:world.config_dim]
                        ax2.add_patch(MplPolygon(
                            tube_poly * tube_scale + c,
                            closed=True, fill=False, edgecolor="r", alpha=0.4,
                            label=None if tube_label_done else "tube"))
                        tube_label_done = True

            ax2.scatter(*q_s, c="tab:blue", label="start")
            ax2.scatter(*q_g, c="tab:green", label="goal")
            ax2.legend(fontsize=7)
            ax2.set_title(f"dd_elastic - corridor {nr}"
                          + (f"  (tube x{tube_scale:g})" if tube_scale != 1.0 else ""))
            fig.savefig(fig_dir / f"dd_elastic_corridor_{nr}.png", dpi=150)
            for ax in (ax1, ax2):
                ax.cla()