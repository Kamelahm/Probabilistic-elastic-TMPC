# Probabilistic Elastic TMPC

Implementation accompanying the paper

> **Probabilistic Elastic Tube MPC for Robust Safe Trajectory Tracking**

Ahmed Kamel, Gokul S. Sankar, and Hamidreza Modares

---

## Overview

This repository contains the implementation of a **Probabilistic Elastic Tube Model Predictive Control (TMPC)** framework for robust safe trajectory tracking under model uncertainty and measurement noise.

The proposed approach integrates

- probabilistic matrix-zonotope identification,
- covariance-based uncertainty characterization,
- mixed deterministic–stochastic uncertainty representations,
- adaptive elastic tube construction,
- recursive-feasible tube MPC design.

The framework constructs uncertainty tubes directly from data-consistent model sets learned from measurements and updates them online via a contraction-based tube-design procedure.

---

## Repository Structure
text
Probabilistic-elastic-TMPC/
│
├── controllers/
│   └── dd_elastic_tube_controller.py
│
├── corridor_simulators/
│   └── dd_elastic_tube.py
│
├── examples/
│   ├── planar_two_dof/
│   │   ├── 2_run_all.py
│   │   ├── 5_collect_data_TMPC.py
│   │   ├── 6_run_theorem1_SDP.py
│   │   ├── 7_run_theorem2_tube.py
│   │   ├── 8_check_model.py
│   │   └── 9_compare_wtilde_vs_delta.py
│   │
│   └── six_dof/
│       ├── 3_show_world_space_motion.py
│       ├── 5_collect_data_TMPC.py
│       ├── 6_run_theorem1_SDP.py
│       └── 7_run_theorem2_tube.py
│
├── figures/
│   ├── planar_two_dof/
│   │   ├── Elastic_tube_2dof_Pic1.pdf
│   │   ├── Elastic_tube_2dof_Pic2.pdf
│   │   └── Elastic_tube_2dof_Pic3.pdf
│   │
│   └── six_dof/
│       ├── Elastic_tube_6dof_start.png
│       ├── Elastic_tube_6dof_middle.png
│       └── Elastic_tube_6dof_goal.png
│
├── paper/
│
├── README.md
│
├── requirements.txt
│
└── LICENSE
```

---

## Features

The repository provides implementations of

- Probabilistic matrix-zonotope identification
- Covariance upper-bound estimation
- Mixed-zonotope uncertainty propagation
- Polyhedral error-tube construction
- Adaptive elastic tube MPC
- Recursive-feasibility analysis
- Collision-free trajectory tracking
- 2-DOF manipulator simulations
- 6-DOF manipulator simulations
- Benchmark comparisons 

---

## Numerical Experiments

The simulations reported in Section V of the paper include

### Conservatism analysis

Comparison between

- analytical worst-case residual bounds

and

- learned effective uncertainty descriptions

obtained from measured trajectories.

### Trajectory tracking

Robust collision-free tracking for

- planar 2-DOF manipulators
- general 6-DOF manipulators

under parametric uncertainty, disturbances, and noisy measurements.

---

## Installation

Clone the repository

```bash
git clone https://github.com/Kamelahm/Probabilistic-elastic-TMPC.git

cd Probabilistic-elastic-TMPC
```

Create an environment

```bash
python -m venv venv

source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running Experiments

Example:

```bash
python examples/planar_two_dof/2_run_all.py
```

Comparison study

```bash
python examples/planar_two_dof/9_compare_wtilde_vs_delta.py
```

Results can be visualized using

```bash
python examples/planar_two_dof/3_print_results.py
```

---

## Acknowledgment

This repository builds upon the open-source framework

> Robust Convex Model Predictive Control with Collision Avoidance Guarantees for Robot Manipulators

available at

https://github.com/whiterabbitfollow/rob_cvx_mpc_rob_man

The original implementation is distributed under the BSD-3-Clause license.

This repository contains the additional modules, modifications, and simulation scripts required to reproduce the results presented in

> Probabilistic Elastic Tube MPC for Robust Safe Trajectory Tracking.

---

## License

This repository follows the BSD-3-Clause license of the original implementation.

Please retain attribution to the original repository when reusing or redistributing code.

---

## Contact

Ahmed Kamel

Michigan State University

Email: kamelahm@msu.edu

GitHub:

https://github.com/Kamelahm
