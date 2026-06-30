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


import pickle

import numpy as np
import trimesh

from examples.six_dof import P_EXAMPLE_6_DOF_DATA
from examples.six_dof.manipulator.man import DemoManipulator
from examples.six_dof.misc.vis import render_path
from examples.six_dof.world import DemoWorld

# p_d = P_EXAMPLE_6_DOF_DATA / "dof_6_ef_0.02" / "motion" / "ft.pckl"

p_d = P_EXAMPLE_6_DOF_DATA / "dof_6_ef_0.02" / "motion" / "dd_elastic.pckl"


with p_d.open("rb") as fp:
    results = pickle.load(fp)

Xs = []
for result in results:
    # X, U, S, r_p_0 = result.outputs
    # Xs.append(X)
    Xs.append(result.x_t) 

Xs = np.stack(Xs, axis=0)

cnt = 0

# def callback(s):
#     global cnt
#     scene.delete_geometry(["X"])
#     X = Xs[cnt]
#     qs, _ = np.split(X.T, 2, axis=1)
#     q = qs[0]
#     man.update_scene(scene, q, s_data)
#     ps = []
#     for q in qs:
#         T, = man.get_link_fk(q, links=["link_6"])
#         ps.append(T[:3, -1])
#     ps = np.vstack(ps)
#     render_path(scene, ps, color=[0, 0, 255], geom_name="X")
#     cnt = (cnt + 1) % Xs.shape[0]


trail = []
def callback(s):
    global cnt, trail
    scene.delete_geometry(["X"])
    x_t = Xs[cnt]                # (12,) single real state
    q = x_t[:6]                  # configuration
    man.update_scene(scene, q, s_data)
    # end-effector position for the trailing path
    T, = man.get_link_fk(q, links=["link_6"])
    trail.append(T[:3, -1])
    if len(trail) > 1:
        render_path(scene, np.vstack(trail), color=[0, 0, 255], geom_name="X")
    cnt = (cnt + 1) % Xs.shape[0]
    if cnt == 0:
        trail = []               # reset trail when the loop restarts


world = DemoWorld()
q_s, q_g = world.get_demo_query()
man = DemoManipulator()

############################################
q_final = Xs[-1][:6]
T_final, = man.get_link_fk(q_final, links=["link_6"])

q_g = np.r_[0.6, np.pi / 4, np.pi / 2, np.zeros(3,)]  
T_goal, = man.get_link_fk(q_g, links=["link_6"])

print("final EE position:", T_final[:3, -1])
print("goal  EE position:", T_goal[:3, -1])
print("EE distance to goal:", np.linalg.norm(T_final[:3, -1] - T_goal[:3, -1]))
############################################

scene = trimesh.Scene(
    [
        world.s_obst.obst,
        trimesh.creation.axis()
    ]
)
man.add_to_scene(scene, q_s, color=[255, 0, 0, 100])
man.add_to_scene(scene, q_g, color=[0, 255, 0, 100])
s_data = man.add_to_scene(scene, geom_name_suffix="t")
scene.show(callback=callback)

# out = P_EXAMPLE_6_DOF_DATA / "dof_6_ef_0.02" / "figures"
# out.mkdir(exist_ok=True)

# N = Xs.shape[0]
# frames = {"start": 0, "middle": N // 2, "goal": N - 1}

# # precompute the full end-effector trail once
# ee = np.vstack([man.get_link_fk(x_t[:6], links=["link_6"])[0][:3, -1]
#                 for x_t in Xs])

# scene.camera_transform = np.array([
#     [ 0.68747834,  0.        ,  0.72620488,  1.09297086],
#     [-0.11556671,  0.98725637,  0.10940385,  0.18136567],
#     [-0.71695039, -0.15913789,  0.67871737,  1.15673268],
#     [ 0.        ,  0.        ,  0.        ,  1.        ],
# ])

# for name, cnt in frames.items():
#     man.update_scene(scene, Xs[cnt][:6], s_data)   # pose the arm
#     scene.delete_geometry(["X"])                    # clear old trail
#     if cnt > 0:
#         render_path(scene, ee[:cnt + 1], color=[0, 0, 255], geom_name="X")
#     png = scene.save_image(resolution=(2400, 1800), visible=True)
#     with (out / f"dd_elastic_{name}.png").open("wb") as f:
#         f.write(png)