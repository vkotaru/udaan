# udaan

A collection of simulation and control scripts written/developed during my stay at Hybrid Robotics;

This package contains mathematical models for quadrotor(s) with suspended payload(s) and controllers for the same. Models are either written manually or developed using `pybullet` or `mujoco` simulators. 


### Models

<table>
  <tr>
    <th>Name</th>
    <th>Remarks</th>
  </tr>
  <tr>
    <td>Quadrotor
      <p float="left">
        <img src=".media/quadrotor_mj.gif" width="400" />
      </p>
    </td>
    <td>
      <ul>
        <li>Geometric control on SE(3).  <a href="https://ieeexplore.ieee.org/document/5717652">[1]</a></li>
        <li>Geometric L1 Adaptive control on SO(3). TODO</li>
        <li>MPC on variation linearized dynamics. TODO</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td> Quadrotor with Cable Suspended Payload
      <p float="left">
        <img src=".media/quadrotor_cspayload_mj.gif" width="400" />
      </p>
    </td>
    <td>
      <ul>
        <li>Geometric control on SE(3).</li>
        <li>MPC on variation linearized dynamics. TODO</li>
      </ul>
    </td>
  </tr>
</table>

