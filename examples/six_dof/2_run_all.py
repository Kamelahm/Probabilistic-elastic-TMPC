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


from collections import defaultdict
import pickle

import cvxpy as cp
import numpy as np
import pandas as pd
import tqdm

from aux import (
    TimeStepIntegratorContinuous,
    TimeStepIntegratorDiscrete,
    interpolate_equidistant
)
from controllers.common import load_all_controllers
from corridor_simulators.flexible_tube import FlexibleTubeSimulator
from corridor_simulators.nom import NomSimulator
from corridor_simulators.tube import TubeSimulator
from examples.six_dof import P_EXAMPLE_6_DOF
from examples.six_dof.manipulator.man import DemoManipulator
from examples.six_dof.world import DemoWorld
from problem_scenario import ProblemScenarioMassAllPin

from controllers.common import load_dd_elastic_controller
from corridor_simulators.dd_elastic_tube import DDElasticTubeSimulator

man = DemoManipulator()

p_d = P_EXAMPLE_6_DOF / "data" / "dof_6_ef_0.02"
ps = ProblemScenarioMassAllPin.from_cached_dir(
    p_d
)


goal_tol = 0.01
horizon_length = 20

fcont, rcont, nom_cont = load_all_controllers(
    ps, horizon_length=horizon_length
)


dd_cont = load_dd_elastic_controller(ps, horizon_length=horizon_length)
if dd_cont is not None:
    dd_cont.set_solver(cp.CLARABEL)

# Override default MOSEK solver (requires license) with free CLARABEL solver
import cvxpy as cp
for c in (fcont, rcont, nom_cont):
    c.set_solver(cp.CLARABEL)

verbose = True

np.random.seed(1)

world = DemoWorld()


simulators = [
    FlexibleTubeSimulator(fcont, TimeStepIntegratorContinuous(dt=ps.dt), world, aux_controller_steps=1),
    TubeSimulator(rcont, TimeStepIntegratorContinuous(dt=ps.dt), world, aux_controller_steps=1),
    NomSimulator(nom_cont, TimeStepIntegratorContinuous(dt=ps.dt), world),
    NomSimulator(nom_cont, TimeStepIntegratorDiscrete(A=nom_cont.A, B=nom_cont.B, ignore_me=True), world),
    DDElasticTubeSimulator(dd_cont, TimeStepIntegratorContinuous(dt=ps.dt), world, aux_controller_steps=1),
]

names = [
    "ft",
    "rt",
    "nom",
    "nom_star",
    "dd_elastic",
]


corridor_statistics = []
computation_times = defaultdict(list)


path = world.get_demo_path()
path_centers = interpolate_equidistant(path, delta=0.05)

N = path_centers.shape[0]
path_radii = world.sdf(path_centers[:, :3])

iter_obj = list(zip(simulators, names))
suffix = f"Running {p_d.name} horizon: {horizon_length}"
if verbose:
    iter_obj = tqdm.tqdm(iter_obj, total=len(iter_obj), desc=suffix)

dyn_nom = ps.get_nominal_dynamics()
dyn_err = ps.get_err_dyn_random()


for simulator, name in iter_obj:
    if simulator.cont is None:
        continue
    simulator.integrator.set_dyns(dyn_nom, dyn_err)
    if verbose:
        iter_obj.set_description(f"{suffix} running {str(name)}")
    (status, ts_goal), all_results = simulator.simulate(
        path_centers,
        path_radii,
        nr_steps=500,
        goal_tol=goal_tol,
        verbose=False
    )
    if len(all_results) > 1:
        # Skip first sample of comp times, due to compilation time of the controller
        computation_times[name].extend([results.times for results in all_results[1:]])
    nr_steps = len(all_results)

    p_r = p_d / "motion"
    p_r.mkdir(exist_ok=True)
    cont_f_name = f'{name}.pckl'
    with (p_r / cont_f_name).open("wb") as fp:
        pickle.dump(all_results, fp)

    corridor_statistics.append(
        {
            "name": name,
            "t2g": ts_goal,
            "status": status
        }
    )

df = pd.DataFrame(corridor_statistics)
print("Results from runs:")
print(df)

