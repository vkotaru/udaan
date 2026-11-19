# Changelog

## Unreleased

### MuJoCo assets

#### Changed

- Every quadrotor marks its heading in a render: the rotor discs on the +x
  (forward, `b1`) side of the airframe are red (`FRONT_ROTOR_RGB`,
  `0.85 0.15 0.15`) and the other rotors carry the vehicle's colour. This
  covers every packaged MJCF and `MujocoAssetCreator.create_quadrotor0` (two
  front rotors in the X layout, one in the `+` layout). The change touches
  only `rgba`.
- Red is reserved for the front rotors; no other vehicle geom and no fleet
  palette entry lies in the red-to-pink hue band. The comparison plant's
  rotors are blue (`0.2 0.4 0.9`) and the reference ghost's cyan
  (`0.1 0.8 0.8`); the rear rotors of `quadrotor.xml`, `quadrotor_payload.xml`
  and the two-quad assets are orange (`0.95 0.45 0.1`, the generator's
  default); the generator's unmodelled-mass box is yellow and its
  `cylinder` default is grey. The fleet palette is blue, green, orange,
  purple (`0.5 0.25 0.85`), cyan, white, olive, navy.
- Generated fleet and multi-quad MJCF is cached under a name that does not
  encode colours: delete cached `fleet_*`, `cspayload_fleet_*` and
  `multi_quad_pointmass_*q` files in `udaan/models/assets/mjcf/` to pick up
  the colours.

### Quadrotor with cable-suspended payload (M2)

#### Added

- `udaan.utils.trajectory.HelixTrajectory`: a horizontal helix, the
  `SpiralTrajectory` jet with its axes permuted so the circle lies in the
  xy-plane and the advance is a climb along z (defaults: start `(0, 0, 1.5)`,
  radius 0.8 m, 2 m/s, two loops, 1 m climb, 1 s holds, 2 s ramps). It has
  `get`, `get_full` and `get_jet`, so flatness feed-forward engages.
- `CircularTraj(ramp_time=...)`: with `ramp_time > 0` the circle starts from
  rest and its angular rate ramps to `speed/radius` through the degree-11
  smoothstep `SpiralTrajectory` uses; `get_jet` stays exact. The default, 0,
  is the unramped circle.
- `udaan run quad-payload --traj {hover,circle,spiral}` (default `hover`):
  `circle` is a 1 m circle centred on `-p` that starts from rest and reaches
  1 m/s over a 2 s ramp, `spiral` the default
  `HelixTrajectory` starting at `-p`; both default `-p` to `0,0,1.5`.
  `--cable-length` sets the cable length in metres and rebuilds the default
  controllers for it; the link chain refuses any length but its MJCF's.
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

### Multi-quadrotor cable-suspended point-mass payload (M3)

#### Added

- **`udaan.models.multi_quad_cspayload`** — `MultiQuadCsPayloadBase` (analytical
  dynamics) and `MultiQuadCsPayloadMujoco` (MuJoCo backend): `nQ` rigid-body
  quadrotors, each on its own taut cable, carrying one point-mass payload. The
  per-vehicle input is a thrust magnitude and a body moment. Cables are
  integrated on `S2` with `udaan.manif.zoh_step`. Both backends fly the
  generated airframe (`DEFAULT_GENERATED_QUAD_MASS`,
  `DEFAULT_GENERATED_QUAD_INERTIA`), close the payload loop at 200 Hz, each
  attitude loop at 800 Hz (`attitude_rate_hz`) and the physics at 1600 Hz
  (`physics_rate_hz`), with a clock that counts substeps; `simulate()` flies
  the same law on the ACCELERATION, WRENCH and PROP_FORCES inputs. The
  analytical model counts the substeps on which a rigid cable pushes
  (`compressive_cable_steps`, `cable_tensions()`) and warns on the first.
  `MultiQuadCsPayloadState.cable_tensions` carries each cable's tension of the
  last physics substep on both backends (`nan` until the first substep after a
  reset); on the MuJoCo backend `cable_tensions()` is each tendon's limit
  force, zero while it is slack. The
  MuJoCo backend generates its MJCF from the construction parameters into the
  user asset cache, reads the body-frame inertia, cable lengths and actuator
  `ctrlrange` from it, writes MuJoCo's timestep as `dt/N` and each tendon's
  `solreflimit` as two timesteps, raises when MuJoCo reports a diverged
  integration, and its `reset()` places the taut ring at the current cable
  lengths and accepts `payload_position` and `payload_velocity` (a rigid
  velocity of the whole formation). Masses, cable lengths and inertias are
  validated at assignment.
- `udaan.utils.formation` (`check_formation`, `default_formation_radius`,
  `min_formation_radius`, `neighbour_spacing`, `default_spawn_height`):
  neighbouring airframes on a ring must be `DEFAULT_ROTOR_SPAN` plus
  `DEFAULT_FORMATION_CLEARANCE` apart and the radius below every cable length;
  the default radius grows with `nQ` and the spawn height follows the cable
  length. Both M3 models and the generator refuse a ring that breaks either.
- **`udaan.control.multi_quad_cspayload.MultiQuadCsPayloadController`** — a
  geometric cascade: payload PD in acceleration units, a `TensionSplitPolicy`
  distributing the net force into per-cable forces and their first two
  derivatives, per-cable `S2` tracking, then each vehicle's
  `GeometricAttitudeController`. The along-cable thrust is floored at
  `MULTI_MIN_CABLE_TENSION_FRACTION` of each cable's hover tension
  (`min_tensions`), and each vehicle's thrust vector is bounded to
  `MULTI_MAX_COMMANDED_TILT_DEG` (50°) from vertical before its attitude loop
  (`max_tilt_deg`, `bound_tilt`): the vertical part is floored at `T_min` and
  the horizontal part scaled down to the bound where it leans past it. Gains follow
  the time-scale design `multi_cable_cascade_gains` (`CABLE_CASCADE_*`),
  computed from the cable lengths and airframe inertias and recomputed on
  `sync_from_model`; `kp`, `kd`, `cable_kp` and `cable_kd` override it.
- **`udaan.flatness.MultiQuadCsPayload`** — flat output
  `(x_L, {ψ_i}, {internal cable forces})`; the internal forces come from a
  `TensionSplitPolicy` (`RingTensionSplit` by default, `AffineTensionSplit`,
  `LegacyFixedTensionSplit`). `simulate()` passes the controller's own policy
  object into the flat map, and enables feed-forward when the bound trajectory
  provides the jet the map needs. `MultiQuadCsPayloadBase.yaw_setpoint` sets the
  heading channel. `recover()` marks samples whose airframes are not all upright
  (`MultiQuadCsPayloadRefState.upright`), `check_reference()` scans a trajectory
  before flight, and `simulate()` flies those samples without the airframe
  feed-forward, counts them in `feedforward_drops` and warns once with the
  intervals.
- `udaan.flatness.cable` (`tension_direction_jet`, `point_mass_cable_force`,
  `cable_direction_jet`) and `udaan.flatness.derivatives` (`leibniz_product`,
  `leibniz_dot`, `leibniz_matmul`, `normalize_derivatives`).
- `udaan.utils.grasp` (`formation_cable_attitudes`) and
  `udaan.utils.assets.cache` (generated
  MJCF goes to `$UDAAN_ASSET_DIR` or the platform user cache, written
  atomically, with `scene.xml` beside it).
- `udaan.utils.assets.multi_quad_cspayload` and `payload_height`: the M3 MJCF
  generator, parameterised by `num_quadrotors`, `cable_length`, `payload_mass`,
  `quad_mass`, `formation_radius` and `spawn_height` (both derived when not
  given), with a `1/DEFAULT_PHYSICS_RATE_HZ` timestep, tendon `solreflimit` of
  two timesteps, each airframe's `ctrlrange` from `core.defaults` and undamped
  root joints. `MujocoAssetCreator.option()`,
  `create_quadrotor0(mass=, root_damping=, thrust_range=, torque_range=)`,
  `MujocoAssetCreator.spatial(solreflimit=)` and `MJCF_GENERATOR_VERSION`. `MujocoAssetCreator.save_to` writes `scene.xml`
  beside the asset when it is not already there.
- CLI: `udaan run multi-quad-payload` and `udaan generate-xml multi-quad-payload`.
  `udaan run multi-quad` and `udaan generate-xml multi-quad` keep running the
  `MultiQuadrotorCSPointmass` model and its generator.
- Docs: theory pages for the M3 dynamics, controller and flat map. Example:
  `examples/multi_quad_cspayload.py`.

### Multi-quadrotor cable-suspended rigid-body payload (M4)

#### Added

- `udaan.models.multi_quad_rigidbody`: `MultiQuadRigidbodyBase` (analytical
  dynamics of `nQ` quadrotors carrying one rigid-body payload on taut cables
  attached at body-frame points) and `MultiQuadRigidbodyMujoco` (its MuJoCo
  backend, on a generated, cached MJCF). Both run the payload cascade at
  200 Hz, the airframe attitude loops at `attitude_rate_hz` (800 Hz) and the
  physics at `physics_rate_hz` (1600 Hz), on every input path; the clock
  counts substeps; the MuJoCo backend writes its timestep as `dt/N`, its
  tendons' `solreflimit` as two timesteps, reads and writes the airframe
  inertia in the body frame, and raises `RuntimeError` on a diverged step.
  `slack_cable_steps` counts steps with a slack cable on both backends (a
  compressive tension on the analytical model, a tendon short of its length
  on MuJoCo). The analytical airframe defaults are the generated MJCF's.
- `udaan.control.multi_quad_rigidbody.MultiQuadRigidbodyController`: geometric
  cascade from payload pose to a wrench, cable tensions, cable directions and
  airframe attitudes, with the reference cable rates of the flat map in its
  cable loops (`compute(t, state, cable_reference)`, `cable_reference_rates`)
  and time-scale gains computed from the cable lengths and the payload inertia
  (`udaan.control.multi_quad_rigidbody.gains`; the payload attitude loop at
  `RIGID_PAYLOAD_ATTITUDE_FREQUENCY_RATIO` times the payload's lateral
  frequency, keywords `attitude_kp`/`attitude_kd`).
- `udaan.flatness.MultiQuadRigidbody`: flat-to-state map with flat output
  `(x_L, R_L, {ψ_i}, λ)`, an `upright` flag on its reference state and
  `check_reference` for a trajectory before flight; `simulate()` drops the
  airframe feed-forward on samples that need a non-upright airframe, counts
  them in `feedforward_drops` and warns once with the intervals.
  `udaan.flatness.allocation` (`WrenchAllocation`, `TensionDistributionPolicy`,
  `RingTensionDistribution`, `FixedTensionDistribution`,
  `MinimumTensionDistribution`, `null_space_basis`) for the wrench-to-tension
  split; the cascade's default policy floors every allocated tension at
  `RIGID_MIN_CABLE_TENSION_FRACTION` of its hover share.
- `udaan.utils.grasp`: ring attachment points, the grasp map, the hover wrench
  map, and the rigid-payload ring's default radius, clearance check and spawn
  height.
- `udaan.utils.assets.multi_quad_rigidbody` and `rigid_payload_height`: the M4
  MJCF generator, with a taut spawn, a default formation radius and spawn
  height derived from the geometry, and a refusal of rings whose airframes are
  closer than the rotor span plus a clearance.
- `udaan run multi-quad-rigidbody`: runs `MultiQuadRigidbodyMujoco` (`-n`,
  `--cable-length`, `--payload-mass`, `--attachment-radius`, `--radius`,
  `--height`) and `udaan generate-xml multi-quad-rigidbody`.
  `udaan run multi-quad-rigid` is unchanged.
- Docs: theory pages for the M4 dynamics and flat map. Example:
  `examples/multi_quad_rigidbody.py`.

### Quadrotor with a payload on a flexible n-link cable (M5)

#### Added

- `udaan.models.quadrotor_flexcable`: `QuadrotorFlexCableBase`, one quadrotor
  carrying a point-mass payload through a cable of `n` rigid links (lumped
  masses at the distal joints), integrated by zero-order-hold Euler at four
  1.25 ms substeps of the 5 ms step, and `QuadrotorFlexCableMujoco`, the same
  machine as a MuJoCo ball-joint chain generated on demand into the user asset
  cache. The MuJoCo backend reads the airframe's mass, its full inertia tensor
  in the body frame, the link lengths and the link masses from the MJCF, reads
  every link's direction and rate from the bodies at the current
  configuration, and raises `RuntimeError` when MuJoCo reports a diverged
  integration. `stability_growth_rate()` and the constructor's stiffness
  warning describe the analytical model's Euler substep; the MuJoCo backend
  does not warn. Both backends fly the generated airframe by default
  (`DEFAULT_GENERATED_QUAD_MASS`, and `DEFAULT_GENERATED_QUAD_INERTIA` scaled
  with the mass), and both clocks count physics substeps, so `k` steps read
  `k · dt` and `simulate(tf)` takes `tf / dt` steps.
- `udaan.control.quadrotor_flexcable.QuadrotorFlexCableController`: an
  infinite-horizon LQR on the variational error state of the whole chain,
  emitting a wrench, with the flat map's reference state and wrench as
  feed-forward. Its cost weights are `DEFAULT_FLEXCABLE_LQR_*` in
  `core/defaults.py`.
- `udaan.control.solve_lqr_gain`: a checked continuous-time Riccati solve
  (residual graded at `ARE_RESIDUAL_WARNING`, refused at
  `ARE_RESIDUAL_TOLERANCE`, closed loop required strictly stable).
- `udaan.flatness.QuadrotorFlexCable`: flat output `(x_L, ψ)`, payload jet to
  order `2n + 4`, built on the shared `flatness/derivatives.py` and
  `flatness/cable.py` and on `flatness/chain.py` (`cable_chain_cascade`,
  `CableChain`, `chain_jet_order`).
  `jet_discontinuities` finds the joints at which a trajectory's jet is
  discontinuous below the map's order, and `QuadrotorFlexCable.reference_steps`
  the steps they put into the reference airframe velocity and acceleration;
  `simulate()` warns when a step exceeds `MAX_FLEXCABLE_REFERENCE_ACCEL_STEP`
  (a fifth of gravity) and keeps them in `reference_steps`.
- `udaan.utils.assets.quad_flexcable`, the M5 MJCF generator, cached in the
  per-user generated-MJCF cache. `MujocoAssetCreator` gains
  `create_flexible_cable_payload(cable_mass=, joint_damping=,
  payload_joint_damping=, payload_joint_stiffness=, payload_radius=,
  link_radius=)` (defaults 0.01 kg, 1e-3, 1000, 1000, 0.05 m and 0.005 m).
- `udaan run quad-flexcable` (`-n` links, `-m base|mujoco`, `--traj
  hover|lissajous`), `examples/quadrotor_flexcable.py`, and the dynamics,
  controller and flat-map theory pages.

### Several quadrotors carrying one flexible hose (M6)

#### Added

- `udaan.models.multi_quad_hose`: `MultiQuadHoseBase`, `nQ` quadrotors bolted
  to chosen vertices of one hose of `n` rigid links (lumped vertex masses,
  ball-jointed links), integrated by zero-order-hold Euler at four 1.25 ms
  substeps of the 5 ms step, and `MultiQuadHoseMujoco`, the same machine as a
  MuJoCo ball-joint tree generated on demand into the user asset cache, each
  airframe on a ball joint at its own centre of mass. The MuJoCo backend reads
  the airframe masses, each airframe's full inertia tensor in its body frame,
  the vertex masses and the link lengths from the MJCF, reads every link and
  airframe at the current configuration, honours every `reset()` keyword the
  analytical model takes, and raises `RuntimeError` when MuJoCo reports a
  diverged integration. `stability_growth_rate()` and the constructor's
  stiffness warning describe the analytical model's Euler substep; the MuJoCo
  backend does not warn.
- `udaan.control.multi_quad_hose.MultiQuadHoseController`: an infinite-horizon
  LQR on the variational error state of the whole hose and every airframe,
  solved at the static catenary trim about the root setpoint, emitting one
  wrench per vehicle, with the flat map's reference state and wrenches as
  feed-forward. It exists when a vehicle sits at the last vertex. Its cost
  weights are `DEFAULT_HOSE_LQR_*` in `core/defaults.py`.
- `MultiQuadHoseBase` counts physics substeps for its clock on both backends
  (`simulate(tf)` takes `tf/dt` steps) and defaults to the generated MJCF's
  airframe (`DEFAULT_GENERATED_QUAD_MASS/INERTIA`, the inertia scaled with the
  mass). `MultiQuadHose.reference_steps` finds the joints where a root
  trajectory's jet is discontinuous below the flat map's order (M5's
  `jet_discontinuities`) and the step each puts into the airframes' reference
  acceleration; `simulate()` warns above `MAX_HOSE_REFERENCE_ACCEL_STEP` (0.2 g)
  and keeps them in `reference_steps`.
- `udaan.flatness.MultiQuadHose`: flat output `(x_0, {𝛕_a}, {ψ_k})`, the root
  jet to order `2n + 4` and one vector tension per vehicle but the last;
  `NotFlatError` when no vehicle sits at the last vertex.
- `udaan.utils.hose` (attachment validation and the taut spawn shape) and
  `udaan.utils.assets.multi_quad_hose`, the M6 MJCF generator.
  `MujocoAssetCreator.sphere` writes `pos` when it is not the body origin; its
  default emits the XML it emitted before.
- `udaan run hose` and `udaan generate-xml hose` (`-n` links, `-a`
  attachments, `-m base|mujoco`, `--controller lqr|station`, `--traj
  hover|lissajous`), `examples/multi_quad_hose.py`, and the dynamics,
  controller and flat-map theory pages.

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
