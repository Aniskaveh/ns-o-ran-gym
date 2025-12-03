# A Gymnasium Environment for ns-O-RAN

This repository contains a package for a [gymnasium](https://gymnasium.farama.org/) based reinforcement learning environment for the 5G O-RAN architecture through the [`ns-O-RAN`](https://openrangym.com/ran-frameworks/ns-o-ran) simulator.

## New Use Case: Handover Optimization (HO)

(Work in progress — to be completed once implementation is ready)

## An overview

At a high level: the system can be viewed through its different parts:

+ The `base` folder contains the abstract class `NsOranEnv`, as well as the two utility classes: `ActionController` and `Datalake`. `NsOranEnv` deals with the communication with the agent and the underlying simulation, `ActionController` writes the agent's action to a file shared with the simulation and `Datalake` acts as a wrapper to an *SQLite* database used to store the *KPMs* (*Key Performance Metrics*).
+ The `environments` folder contains `TrafficSteeringEnv`, `EnergySavingEnv` and `HandoverEnv`, environments derived from `NsOranEnv`, implementing the Traffic Steering, Energy Saving and handover use cases.

The primary goal of this work is to provide a Gymnasium-compliant environment for 5G Open RAN online reinforcement learning. To accommodate a wide range of use cases, we have developed `NsOranEnv`, an abstract environment that serves as the foundational building block for all new environments. `NsOranEnv` coordinates both the environment and the ns-3 simulation, offering several utilities as well. This structure simplifies the creation and testing of new environments, as the complexities of ns-3 and its simulations are managed by the existing `NsOranEnv`.

Briefly, every new environment that is built on NsOranEnv should provide a json file to the constructor (`scenario_configuration`) that will be used to fetch the necessary information for the environment to work. Of course, this json can be fine tuned to provide the necessary parameters for any specific new environment. Moreover, each newly created environment should include the following methods, which are require by `NsOranEnv`:

+ `_get_obs()`: returns the observation state;
+ `_compute_reward()`: computes the reward;
+ `_compute_action()`: converts the agent’s action defined in gym into the format required by ns-O-RAN,

Additionally, `_init_datalake_usecase()` and `_fill_datalake_usecase()`  may be implemented to capture additional data from ns-O-RAN and store it in the Datalake.
A full example can be found on `EnergySavingEnv`, which extends `NsOranEnv` for the O-RAN Energy Saving use case. For specific details on how each abstract method should be extended by new environments see the `docs/` and `CITATION.cff`.

![](./docs/environment.svg)

## References

If you use the Gymnasium Base Environment, please reference the following paper:

```text
@INPROCEEDINGS{10619796,
  author={Lacava, Andrea and Pietrosanti, Tommaso and Polese, Michele and Cuomo, Francesca and Melodia, Tommaso},
  booktitle={2024 IFIP Networking Conference (IFIP Networking)}, 
  title={Enabling Online Reinforcement Learning Training for Open RAN}, 
  year={2024},
  volume={},
  number={},
  pages={577-582},
  keywords={Training;Cellular networks;Open RAN;Computer architecture;Software;Quality of experience;Telemetry;Open RAN;ns-3;deep reinforcement learning;artificial intelligence;gymnasium},
  doi={10.23919/IFIPNetworking62109.2024.10619796}}
```

If you use the HandoverEnv Environment, please reference the following paper:

```text
@INPROCEEDINGS{anis,
AUTHOR="Fatemeh Kavehmadavani",
TITLE="GRL-HO",
BOOKTITLE="...",
ADDRESS="...",
PAGES="...",
DAYS=...,
MONTH=...,
YEAR=...
}
```
