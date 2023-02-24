# udaan

A collection of simulation and control scripts written/developed during my stay at Hybrid Robotics;

This package contains mathematical models for quadrotor(s) with suspended payload(s) and controllers for the same. Models are either written manually or developed using `pybullet` or `mujoco` simulators. 


### Models

#### Quadrotor

<p float="left">
  <img src=".media/quadrotor_mj.gif" width="400" />
</p>

Controllers
1. Geometric control on SE(3).
2. Geometric L1 Adaptive control on SO(3). TODO
3. MPC on variation linearized dynamics. TODO

#### Quadrotor with Cable-Suspended Payload

Controllers
1. Geometric control on SE(3)xS2. TODO
2. MPC on variation linearized dynamics. TODO
---



