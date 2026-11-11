# Changelog

## v1.4.0 (2026-11-10)

### Quadrotor (M1)

#### Added

- **Loop rates are constructor keywords** on `QuadrotorBase` and
  `QuadrotorMujoco`. `dt` is the position-loop period (default 5 ms, 200 Hz).
  `physics_rate_hz` (default `DEFAULT_PHYSICS_RATE_HZ = 1600`) sets
  `N = physics_rate_hz · dt` integration substeps of `dt/N` per `step()` on both
  backends; it must lie in `PHYSICS_RATE_RANGE_HZ` (500–2000 Hz) and give a
  whole `N`, otherwise `ValueError` lists the valid rates. `attitude_rate_hz`
  (default `DEFAULT_ATTITUDE_RATE_HZ = 800`; `None` is once per step) must be
  `k/dt` for a whole `k` dividing `N`.
- **Rotor parameters are constructor keywords**: `arm_length`, `rotor_angles`,
  `spin_directions`, `force_constant`, `torque_constant`, `prop_min_force`,
  `prop_max_force`, defaulting from `core/defaults.py` and validated, including
  a rank-4 check on the allocation matrix. `allocation_matrix` and
  `rotor_parameters` read them back. On `QuadrotorMujoco` the MJCF is their
  source (rotor sites, yaw-gear signs and size, `ctrlrange`); a passed value
  that disagrees with it raises `ValueError`.
- `control.quadrotor.quadrotor_allocation_matrix`, which builds the matrix;
  `DirectPropellerForceController` takes `allocation_matrix=` and has
  `from_model(model)`.
- `QuadrotorBase.yaw_setpoint`: a callable `ψ(t) -> (ψ, ψ̇, ψ̈)` driving the
  heading of the flatness reference attitude; `None` (the default) is ψ ≡ 0.
  `udaan.flatness.yaw_jet` turns it into the order-2 jet the flat map takes.
- Trajectories in `udaan.utils.trajectory`: `StepSetpoint` (a step, no
  `get_full`), `WaypointTrajectory` (take-off, waypoints with a stop at each,
  landing; 7th-order rest-to-rest legs) and `Figure8Trajectory` (a lemniscate
  whose `yaw()` swings the heading). `PolyTraj5.get_full`;
  `SpiralTrajectory.get_jet(t, order)`; `FlipTrajectory.reference_attitude(t)`,
  the closed-form rotation `R = exp(θ ê₂)` about the fixed pitch axis during the
  flip phase and `None` elsewhere, which `QuadrotorBase.simulate()` prefers over
  the flat map wherever it is not `None`.
- `udaan run quadrotor --traj` offers `step`, `mission` and `figure8`.
- `core/defaults.py`: `DEFAULT_M1_QUAD_MASS`, `DEFAULT_M1_QUAD_INERTIA`,
  `DEFAULT_ROTOR_ANGLES`, `DEFAULT_SPIN_DIRECTIONS`, `DEFAULT_ATTITUDE_RATE_HZ`,
  `DEFAULT_PHYSICS_RATE_HZ`, `PHYSICS_RATE_RANGE_HZ`, the actuator limits
  `DEFAULT_MIN_THRUST`, `DEFAULT_MAX_THRUST`, `DEFAULT_MIN_TORQUE`,
  `DEFAULT_MAX_TORQUE`, `DEFAULT_PROP_MIN_FORCE`, `DEFAULT_PROP_MAX_FORCE`, and
  `MIN_ATTITUDE_HINT_CONDITIONING`.
- `udaan.core.validation.check_positive`, rejecting non-positive and
  non-finite values.

#### Changed

- `QuadrotorBase.simulate()` flies the bound position setpoint on all three
  input paths (`acceleration`, `wrench`, `prop_forces`) with the model's own
  position and attitude loops and the same flatness feedforward. Every path
  closes the attitude loop inside `step()` at `attitude_rate_hz`, on both
  backends.
- The clock counts substeps, `t = n / physics_rate_hz`. `QuadrotorMujoco`
  writes `dt/N` into MuJoCo's timestep and the clock into `data.time`, so a
  `step()` advances exactly `dt`.
- The M1 airframe is `DEFAULT_M1_QUAD_MASS = 0.75` kg and
  `DEFAULT_M1_QUAD_INERTIA = diag(0.0053, 0.0049, 0.0098)` kg·m², the body of
  `quadrotor_mj.xml`, on both backends. The shared `DEFAULT_QUAD_MASS` and
  `DEFAULT_QUAD_INERTIA` used by the payload and multi-vehicle models are
  unchanged.
- `quadrotor_mj.xml`: the wrench `ctrlrange` equals the `core/defaults.py`
  actuator limits, and the rotor positions and yaw gears equal the allocation
  matrix.
- The flatness reference attitude composes with the thrust-derived attitude
  command: the thrust command sets `b₃`, and the reference's `b₁` is the
  heading hint, blended toward a fallback frame as `‖b₃ × hint‖` falls below
  `MIN_ATTITUDE_HINT_CONDITIONING`.
- `PositionL1Controller` is the piecewise-constant L1 law around
  `PositionPDController`, with an unbiased matched estimate. Its predictor uses
  the thrust the attitude delivers, `(u·b₃)b₃`, read from an optional third
  state element: position controllers take `(p, v)` or `(p, v, R)`.
  `QuadrotorBase.simulate()`, `DirectPropellerForceController` and
  `QuadrotorFleet` pass `R`.
- `GeometricL1AttitudeController`'s default `c` is half the Routh bound
  `c < k_Ω(k_R + Γ)/Γ` per axis and follows gain changes; an explicit `c` past
  the bound logs a warning.
- `PositionPDController` keeps user-supplied `kp`/`kd`.
- `SmoothTraj5`, `SmoothTraj3`, `SmoothTraj1` and `PolyTraj5` return
  derivatives in physical time for any `t_f`.
- `FlipTrajectory`: `n_loops` defaults to 1; the climb and recovery end at the
  displacement their boundary conditions imply, so `ż` never reverses, and
  carry jerk and snap; flip-rate ramps (`flip_ramp_time`, default
  `flip_duration / 4`) join the flip to them C⁴.
- `SpiralTrajectory` time-warps its phase through a degree-11 smoothstep over
  `ramp_time` (default 2 s), so the reference is C⁶ everywhere.
- `QuadrotorBase.reset()` and `simulate()` raise `ValueError` on an
  unrecognised keyword.
- `udaan run quadrotor`: `--model` is `base` or `mujoco`; an unrecognised
  `--model` or `--trajectory` exits with an error listing the choices. The
  step, circle, spiral, flip and figure-8 default to a start at `1,1,2`, and
  every trajectory starts the vehicle on its own reference.

#### Removed (breaking)

- `QuadrotorMujoco(attitude_zoh=...)`: raises `TypeError`; the attitude rate
  is `attitude_rate_hz`.
- `QuadrotorBase(inner_loop_steps=...)`: raises `TypeError`; the substep count
  follows from `physics_rate_hz`.
- `DirectPropellerForceController.compute_alloc_matrix()`.
- The `l1` extra. `PositionL1Controller` does not use the `control` package.
- `udaan run quadrotor -m vfx`. `QuadrotorVfx` remains importable as a library
  class.

## v1.0.0 (2025-04-08)

Initial public release.

### Models
- Base models: Quadrotor, QuadrotorCSPayload, FloatingPointmass, S2Pendulum, PointmassSuspendedPayload, MultiPointmassSuspendedPayload
- MuJoCo models: Quadrotor, QuadrotorCSPayload (tendon/links), MultiQuadrotorCSPointmass, MultiQuadRigidbody, QuadrotorComparison
- Built-in GLFW viewer with mouse interaction and real-time sync

### Controllers
- Geometric PD on SE(3) (Lee, Leok, McClamroch 2010)
- Geometric PD on SE(3) x S2 for payload (Sreenath, Lee, Kumar 2013)
- L1 adaptive position control (Kotaru, Wu, Sreenath 2020)
- Direct propeller force allocation

### Tools
- Manifold library (SO3, S2, hat/vee, Rodrigues, matrix exponential)
- Trajectory generators (smooth, polynomial, circular, Lissajous)
- Differential flatness maps
- Typer CLI (`udaan run`, `udaan generate-xml`)
- MuJoCo XML model generator
