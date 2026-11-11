# Changelog

## Unreleased

### Quadrotor with cable-suspended payload (M2)

#### Added

- `udaan.manif.zoh_step`: one zero-order-hold Euler substep on `S2` that steps
  `q` and then projects `ω` onto the tangent plane at the new `q`.
- **Loop rates are constructor keywords** on `QuadrotorCsPayloadBase` and
  `QuadrotorCsPayloadMujoco`, as on M1: `dt` is the payload-loop period
  (default 5 ms, 200 Hz), `physics_rate_hz` (default 1600 Hz, 500–2000 Hz)
  sets `N = physics_rate_hz · dt` substeps of `dt/N` on both backends, and
  `attitude_rate_hz` (default 800 Hz; `None` is once per step) must be `k/dt`
  for a whole `k` dividing `N`, otherwise `ValueError`. `attitude_rate_hz`,
  `physics_rate_hz` and `physics_steps` read them back.
- `QuadCSPayloadController` takes `mass_q`, `mass_l`, `cable_length` and
  `inertia` keywords (defaults: the M2 machine below) and has
  `sync_from_model(model)`; `mass_q`, `mass_l`, `cable_length`, `inertia`
  and `min_tension` read them back.
- The payload controller floors the tension it commands along the cable at
  `T_min = MIN_CABLE_TENSION_FRACTION · m_L · g` (`0.1 m_L g`), so a
  downward command keeps the cable taut and the commanded attitude upright.
- `QuadrotorCsPayloadBase.simulate()` passes the flatness reference attitude
  `(R, Ω, Ω̇)` to the attitude loop when the payload controller's setpoint is
  a bound method of a trajectory with `get_jet`: the payload jet through order
  6 goes through `QuadrotorCsPayload.from_model(model)`. The thrust command
  keeps `b₃` and the reference supplies the heading and rate feed-forward.
  `QuadrotorCsPayloadBase.yaw_setpoint` (`ψ(t) -> (ψ, ψ̇, ψ̈)`, `None` is
  ψ ≡ 0) drives its heading.
- `QuadrotorCsPayloadMujoco.step()` raises `RuntimeError` when MuJoCo reports a
  diverged integration (a bad acceleration, velocity or position) or its time
  runs backwards.
- `QuadrotorCsPayloadBase.allocation_matrix`.
- `core/defaults.py`: `DEFAULT_M2_PAYLOAD_MASS`, `DEFAULT_M2_CABLE_LENGTH`,
  `MIN_CABLE_TENSION_FRACTION`.
- `QuadrotorCsPayloadBase` validates its parameters at assignment: `mass`,
  `payload_mass` and `cable_length` must be finite and strictly positive, and
  `inertia` must be `(3,)` or `(3, 3)`, symmetric and positive definite;
  otherwise `ValueError`.
- `examples/quadrotor_cspayload.py`, and theory pages for the M2 dynamics and
  its payload-tracking controller.
- **Time-scale gain design** (`udaan.control.quadrotor_cspayload.gains`):
  `loop_frequencies(ℓ)` places the attitude loop at `ω_a = 40` rad/s, the cable
  loop at `ω_c = min(2√(g/ℓ), ω_a/5)`, the payload loop at `ω_p = ω_c/5`
  (x, y) and `ω_z = ω_a/25` (z), and `cable_cascade_gains(m_Q, m_L, ℓ, J)`
  turns them into gains with `ζ = 0.7` on every loop: `k_x = Mω²`,
  `k_v = 2ζMω` (`M = m_Q + m_L`), `k_q = m_Q ℓ ω_c²`, `k_ω = 2ζ m_Q ℓ ω_c`,
  `k_R = J ω_a²`, `k_Ω = 2ζ J ω_a` (roll, pitch; yaw keeps
  `DEFAULT_ATT_KP`/`DEFAULT_ATT_KD`). The constants are
  `CABLE_CASCADE_ATTITUDE_BANDWIDTH`, `CABLE_CASCADE_LOOP_RATIO`,
  `CABLE_CASCADE_PENDULUM_FACTOR` and `CABLE_CASCADE_DAMPING` in
  `core/defaults.py`.
- `QuadCSPayloadController` takes `position_kp`, `position_kd`, `cable_kp` and
  `cable_kd`; the gains not given are the design's for the controller's
  machine and are recomputed by `sync_from_model`. `design_gains`,
  `position_gains` and `cable_gains` read them back.
  `DEFAULT_PAYLOAD_POS_KP`, `DEFAULT_PAYLOAD_POS_KD`, `DEFAULT_PAYLOAD_CABLE_KP`
  and `DEFAULT_PAYLOAD_CABLE_KD` are fixed gains for these keywords; with the
  model's own attitude gains the default machine is stable with them at hover
  for 0.25 m ≤ ℓ ≤ 1.76 m.
- `QuadrotorCsPayloadBase.cable_mass` (0: the analytical cable is massless)
  and `QuadrotorCsPayloadMujoco.cable_mass` (0 on the tendon, the chain's 10 g
  on the link chain). The default payload controller and
  `QuadCSPayloadController.sync_from_model` take `m_L = payload_mass +
  cable_mass`, so on the link chain the hover feed-forward carries the chain's
  weight and the payload holds its setpoint's altitude.
- `GeometricAttitudeController` takes `kp` and `kd` (defaults `DEFAULT_ATT_KP`,
  `DEFAULT_ATT_KD`).
- **Reference feasibility** (`udaan.flatness.feasibility`): `is_upright`,
  `scan_reference`, `ReferenceFeasibility` and `infeasible_intervals` judge
  whether a reference keeps every airframe upright (`b₃·e₃ >
  MIN_UPRIGHT_THRUST_Z`). `QuadrotorCsPayloadRefState.upright` flags each
  sample the M2 flat map recovers, and
  `QuadrotorCsPayload.check_reference(jet, tf)` runs the map along a
  trajectory before flight and reports the infeasible intervals.
- `QuadrotorCsPayloadBase.simulate()` flies a step without the airframe
  feed-forward where the reference is not upright or the map refuses it,
  counts those steps in `feedforward_drops`, and logs one warning with the
  intervals at the end of the run.
- `CircularTraj.get_jet(t, order)` (every derivative in closed form) and
  `CircularTraj.get_full(t)`, so a circle drives the flatness feed-forward on
  M1 and M2, including `udaan run quadrotor --traj circle`.

#### Changed

- M2 is the M1 airframe (`DEFAULT_M1_QUAD_MASS` 0.75 kg,
  `DEFAULT_M1_QUAD_INERTIA`) with a 0.15 kg payload on a 1.000 m cable: the
  analytical model's defaults, the `QuadrotorCsPayload` flat map's defaults,
  and the bodies of `quad_payload_mj.xml` and `quad_flxcbl_mj.xml`. The link
  chain hangs the payload 1.000 m below the airframe and its airframe has an
  explicit `<inertial>`.
- `QuadrotorCsPayloadMujoco`'s default `cable_model` is `"tendon"`, as on the
  CLI. The tendon's limit is written as `solreflimit = (2 h, 1)` from the
  timestep `h = dt/N` (0.00125 s at 1600 Hz), so a taut cable holds its length.
- The link chain (`quad_flxcbl_mj.xml`) has 10 links of 0.1 m with 10 g of
  cable mass in all, each ball joint at its link's proximal end and the payload
  welded at the distal end.
- On `QuadrotorCsPayloadMujoco` (tendon and link chain) the MJCF is the source
  of `m_Q`, the full inertia tensor (principal moments rotated by
  `body_iquat`), `m_L` and `ℓ`; the payload and attitude controllers are
  rebuilt with them. On the link chain, setting `cable_length` to anything
  but the chain's own length raises `ValueError`.
- `simulate()` flies the payload controller on all three input paths
  (`acceleration`, `wrench`, `prop_forces`), closing the attitude loop inside
  `step()` at `attitude_rate_hz` on every path and both backends.
- The clock counts substeps, `t = n / physics_rate_hz`;
  `QuadrotorCsPayloadMujoco` writes `dt/N` into MuJoCo's timestep and the
  clock into `data.time`, so a `step()` advances exactly `dt`.
- The tendon and link-chain MJCFs' actuator ranges equal the model's limits
  (thrust 0–20 N, moments ±5, ±5, ±2 N·m), and the tendon MJCF's rotor sites
  and yaw gears are written to ten digits.
- Link chain: the payload's position and velocity are the payload body's own
  kinematics for the current state (`mj_objectVelocity`), and the cable rate
  is `ω = q × (v_L − v_Q)/ℓ` on both cable models. Its `reset()` places the
  airframe from `position`, or from `payload_position` and the chain's
  length, sets the airframe's pose and rates, and turns the straight chain about its first ball joint to `cable_attitude` at
  `cable_angular_velocity`; it raises `ValueError` on `payload_velocity`, which
  follows from those, and on `position` and `payload_position` together.
- `reset()` ties the velocities by `v_L = v_Q + ℓ (ω × q)` on the analytical
  model and both cable models: the airframe's is `velocity`; the payload's is
  `payload_velocity` when given (the cable rate is then
  `ω = q × (v_L − v_Q)/ℓ`), else `v_Q + ℓ (ω × q)` from
  `cable_angular_velocity`. `payload_velocity` and `cable_angular_velocity`
  together raise `ValueError`.
- `QuadrotorCsPayloadBase` integrates the cable with `zoh_step`, so
  `q · ω = 0` holds to machine precision.
- `QuadrotorCsPayloadBase.reset()` raises `ValueError` on a keyword other than
  `position`, `velocity`, `orientation`, `angular_velocity`,
  `payload_position`, `payload_velocity`, `cable_attitude` and
  `cable_angular_velocity`.
- `QuadrotorCsPayloadBase`'s actuator limits come from `core/defaults.py`.
- The M2 payload controller's gains and the M2 model's roll and pitch attitude
  gains are the time-scale design for the model's machine, so they follow the
  cable length, the masses and the inertia (on `QuadrotorCsPayloadMujoco`, the
  MJCF's). At the default machine: `k_x = (1.41, 1.41, 2.30)`,
  `k_v = (1.58, 1.58, 2.02)`, `k_q = 29.4`, `k_ω = 6.58`,
  `k_R = (8.48, 7.84, 1.35)`, `k_Ω = (0.297, 0.274, 0.225)`.
- `udaan run quad-payload -m base` runs the analytical model; `--model` is
  `base` or `mujoco`, and any other value exits with an error listing them.

#### Removed

- The composite cable: `cable_model="cable"` and `quad_cable_mj.xml` (MuJoCo
  4 has no `composite` element). `QuadrotorCsPayloadMujoco` and
  `udaan run quad-payload -c` refuse it, listing `tendon` and `links`.
- The `inner_loop_steps` keyword (`TypeError`; `physics_rate_hz` sets the
  substeps) and, on `QuadrotorCsPayloadMujoco`, `attitude_zoh` (`TypeError`;
  `attitude_rate_hz=None` is once per step).
- `QuadrotorCsPayloadBase`'s `_prop_controller`: `simulate()` flies the payload
  controller on the `prop_forces` path too.

## v1.4.1 (2026-11-11)

### Fixed

- The MuJoCo viewer's mouse drag and scroll work on MuJoCo 3.11 and later,
  whose `mjv_moveCamera` takes no `mjvScene` argument; the call matches the
  installed MuJoCo's signature.

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
