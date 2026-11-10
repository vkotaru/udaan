# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v2.0.0 (unreleased)

The release that takes the package from two systems to six, adds a
differential-flatness map for every one of them, and removes the legacy
standalone models. **This is a breaking release** — see *Removed* and *Changed*.

### Added

#### Systems

Four new systems, each laid out the same way as the existing two
(`models/<system>/base.py` for the pure analytical dynamics, `mujoco.py` for the
MuJoCo backend), each with a controller, a flat-to-state map, MJCF assets, a CLI
command and theory documentation:

- **`multi_quad_cspayload` (M3)** — `nQ` quadrotors carrying one cable-suspended
  point-mass payload. The quadrotors are **rigid bodies**, not point-mass force
  vectors as in the legacy model: the per-vehicle input is a thrust magnitude
  and a body moment, which is what makes per-quadrotor attitude feedforward
  recoverable. Verified against an independent Newton–Euler solve with explicit
  tension unknowns: `|a_L|` error ≤ 7e-15 and `|ω̇|` error ≤ 1.3e-9 across
  `nQ = 1..7` with randomised heterogeneous masses and cable lengths. At
  `nQ = 1` it reproduces M2 to 3.0e-14 in payload position.
- **`multi_quad_rigidbody` (M4)** — `nQ` quadrotors carrying one cable-suspended
  **rigid-body** payload: the load has attitude, cables attach at distinct
  body-frame points, and the team can exert a moment on it. The derivation turns on projecting each
  quadrotor equation onto its own cable direction, which annihilates the cable
  angular acceleration and makes the tension affine in `(a_L, Ω̇_L)` alone.
- **`quadrotor_flexcable` (M5)** — a quadrotor carrying a payload through a
  flexible cable of `n` rigid links, with an analytical model derived from the
  Lagrangian as a symmetric `(3+3n)` block solve and the full link-chain state
  observed on the MuJoCo backend. (M2's `cable_model="links"` option simulates a
  link chain but reports a point-mass state.)
- **`multi_quad_hose` (M6)** — `nQ` quadrotors bolted at chosen vertices of one
  flexible hose of `n` links. State is `6 + 6n + 12nQ`. Its mass matrix is
  exactly M5's: vehicles enter only through the right-hand side, so a system
  with three times M5's state needs no new solve machinery.

#### Differential flatness

- **`udaan.flatness`** — a new public subpackage holding flat-to-state maps for
  **all six** systems, replacing the legacy `flat2state` module. The central
  type is `Jet`, an `(order+1, dim)` array whose row *k* is the *k*-th time
  derivative (no factorial scaling). Each system subclasses `Flat2State` and
  declares `Flats` / `RefState` / `Inputs` dataclasses; `__init_subclass__`
  enforces this. Note the subpackage is **not** re-exported from
  `udaan/__init__.py` — import it explicitly.
  - `Quadrotor` — flat output `(x_Q, ψ)`.
  - `QuadrotorCsPayload` — flat output `(x_L, ψ)`.
  - `MultiQuadCsPayload` — flat output `(x_L, {ψ_i}, {internal cable forces})`.
    The payload balance `Σᵢ Tᵢ = F_net` is 3 equations in `3nQ` unknowns, so the
    tension split is not derived but *chosen*, and is part of the flat output;
    it is supplied by a pluggable `TensionSplitPolicy` (`RingTensionSplit` by
    default). Jet depth is independent of `nQ`.
  - `MultiQuadRigidbody` — flat output `(x_L, R_L, {ψ_i}, λ)`, with the free
    part of the split being the null space of the grasp map, resolved by a
    `TensionDistributionPolicy` through the `WrenchAllocation` primitive.
  - `QuadrotorFlexCable` — flat output `(x_L, ψ)`, the same as M2, of which it
    is the `n = 1` case. Unlike every other map its required jet depth grows
    with system size, as `2n + 4`.
  - `MultiQuadHose` — flat output `(x_0, {𝛕_a}, {ψ_k})`. The only map whose flat
    output is not built from configuration alone: the free coordinates are the
    *vector tensions* in the links hanging off all but the last vehicle.
    Positions cannot serve — every vehicle vertex with an onward link has six
    scalar unknowns against three equations. Flat only when a vehicle sits at
    the last vertex; raises `NotFlatError` otherwise.

- **Shared flatness primitives**, so chain rules are written once rather than
  re-derived per system:
  - `flatness/derivatives.py` — an arbitrary-order derivative engine:
    `leibniz_product`, `leibniz_dot`, `leibniz_matmul`, `normalize_derivatives`,
    with no cap on derivative order or link count.
  - `flatness/cable.py` — `cable_direction_jet`, `tension_direction_jet`,
    `point_mass_cable_force`.
  - `flatness/chain.py` — `cable_chain_cascade`, `CableChain`,
    `chain_jet_order`: sweeps a link chain from tip to top, propagating position
    and accumulating tension. Used by M5 and, on the index-reversed chain, M6.
  - `flatness/allocation.py` — `WrenchAllocation`, `TensionDistributionPolicy`,
    `RingTensionDistribution`, `FixedTensionDistribution`, `null_space_basis`:
    wrench-to-tension allocation for an over-actuated rigid-body payload.
  - `flatness/jet.py` — `yaw_jet`, the shared helper turning a `ψ(t)` callable
    into the order-2 scalar jet the flat maps take.

#### Controllers

- `control/multi_quad_cspayload` — a geometric cascade for M3: payload PD gives
  a desired net force, the `TensionSplitPolicy` distributes it and (because it
  differentiates in closed form) returns the desired cable direction and its
  first two derivatives, then per-cable S2 tracking and the existing
  `GeometricAttitudeController` close the loop per vehicle. `simulate()` passes
  the controller's own policy object into the flat map, so feedback and
  feedforward cannot allocate differently.
- `control/multi_quad_rigidbody` — M4 payload controller, allocating through the
  shared `flatness/allocation.py` rather than a private second copy. Pinned
  bit-identical to `allocate()` over 1200 random in-range wrenches across
  `nQ = 1..6`.
- `control/quadrotor_flexcable` — infinite-horizon LQR for M5. LQR rather than
  the geometric cascade every other system uses, and that is structural: a
  serial chain has one actuated body, so there is no per-link body and no loop
  to invert one level at a time.
- `control/multi_quad_hose` — infinite-horizon LQR for M6 on the variational
  linearisation about a catenary trim, generic in `n`, `nQ` and attachment
  pattern.

#### Other

- A yaw setpoint on the flatness feedforward path of M1, M3 and M4 —
  `QuadrotorBase.yaw_setpoint`, `MultiQuadCsPayloadBase.yaw_setpoint`,
  `MultiQuadRigidbodyBase.yaw_setpoint` — taking `ψ(t) -> (ψ, ψ̇, ψ̈)` and
  defaulting to `None`, which means ψ ≡ 0. It drives the heading channel of the
  reference attitude.
- `MIN_ATTITUDE_HINT_CONDITIONING` guards the commanded-attitude construction
  against a near-degenerate heading hint. The reference's own `b₁` tilts with
  the reference and can line up with the commanded thrust direction; as
  `‖b₃ × hint‖` falls the hint is blended continuously toward a fallback frame.
- `udaan.core.validation.check_positive`, shared by every model. M2 and M3 now
  reject non-positive and non-finite masses, cable lengths and inertias at
  assignment rather than failing inside the integrator.
- `udaan.manif.zoh_step` — the shared S2 substep used by M2–M6.
- `udaan.utils.grasp` and `udaan.utils.hose` — geometry helpers for the M4 grasp
  map and the M6 hose vertex layout.
- `udaan.utils.assets.cache` — the user-cache asset store described under
  *Changed*.
- **M1 rotor parameters are constructor keywords.** `QuadrotorBase` takes
  `arm_length`, `rotor_angles`, `spin_directions`, `force_constant`,
  `torque_constant`, `prop_min_force` and `prop_max_force` (defaults unchanged,
  from `core/defaults.py`, which gains `DEFAULT_ROTOR_ANGLES` and
  `DEFAULT_SPIN_DIRECTIONS`), each validated, including a rank-4 check on the
  allocation matrix; `allocation_matrix` and `rotor_parameters` read them back.
  On `QuadrotorMujoco` the MJCF is their source: the rotor sites give `ℓ` and
  `θᵢ`, the yaw gears' sign and size `dᵢ` and `k_M/k_F`, and `ctrlrange` the
  limits; a passed value that disagrees with the MJCF raises `ValueError`. The
  matrix is built by the new `control.quadrotor.quadrotor_allocation_matrix`,
  which `DirectPropellerForceController` also uses (its own copy is gone); it
  takes `allocation_matrix=` and has `from_model(model)`. Its
  `compute_alloc_matrix()` method was removed.
- **M1 physics rate is a constructor keyword**, `physics_rate_hz` (default
  `DEFAULT_PHYSICS_RATE_HZ = 1600`), on both backends: `N = physics_rate_hz · dt`
  substeps of `dt/N`, validated to 500–2000 Hz and a whole `N`, with a
  `ValueError` listing the valid rates otherwise. `QuadrotorMujoco` writes `dt/N`
  into MuJoCo's timestep instead of reading the MJCF's and rounding, so the
  `dt`-correction warning branch is gone; `attitude_rate_hz` is validated
  against the new `N`. `QuadrotorBase(inner_loop_steps=...)` is replaced by it
  and raises `TypeError`.
- **M1 loop rates: 200 / 800 / 1600 Hz by default** (position / attitude /
  physics). `DEFAULT_ATTITUDE_RATE_HZ = 800` and
  `DEFAULT_PHYSICS_RATE_HZ = 1600`: the attitude loop runs four times per
  position update. `attitude_rate_hz=None` gives once per step. A
  `physics_rate_hz` whose `N` four updates do not divide (600, 1000, 1400,
  1800, 2000 Hz at 5 ms) needs an explicit `attitude_rate_hz`; the error says
  so. `dt` is documented as the position-loop keyword and validated
  with the two rates.
- **M1 clock is exact.** Both backends counted time as `t += h` (MuJoCo:
  `data.time`), which fell short at rates whose `h` is not a binary fraction:
  a 10 s `simulate()` at 1000 or 1600 Hz took 2001 steps and ended at
  10.005 s, at `dt = 10 ms` 10.010 s. The clock now counts substeps,
  `t = n / physics_rate_hz`, and `QuadrotorMujoco` writes it into `data.time`.
- Three trajectories in `udaan.utils.trajectory`: `StepSetpoint` (a true step,
  no `get_full`), `WaypointTrajectory` (take-off, waypoints with a stop at each,
  landing; 7th-order rest-to-rest legs) and `Figure8Trajectory` (a lemniscate
  with a swinging heading through `yaw()`, bound as the model's
  `yaw_setpoint`). `udaan run quadrotor --traj` offers them as `step`,
  `mission` and `figure8`.
- CLI commands for the four new systems, plus `--cable-length`,
  `--payload-mass` and `--radius` options on `multi-quad`.
- Theory documentation for every system (dynamics, controllers and
  differential-flatness pages), a `tests/conftest.py` with a seeded `rng` and a
  calibrated tolerance ladder, and a Sphinx docs build script.

### Changed

- **`reset()` and `simulate()` reject an unrecognised keyword** with
  `ValueError` instead of discarding it and starting from the default state.
  A 1.x call such as `simulate(tf, xL=x0)` against M3 (which names that keyword
  `payload_position`) therefore raises. The rule is applied across every model that takes reset kwargs — M1–M6, both backends,
  and the three fleet models — through the shared `udaan.models._kwargs`
  helper; the three MuJoCo backends that already rejected unknown keywords keep
  their wording.
- **Actuator limits moved to `udaan.core.defaults`.** `_min_thrust`,
  `_max_thrust`, `_min_torque`, `_max_torque`, `_prop_min_force` and
  `_prop_max_force` were literals in six separate `base.py` constructors,
  identical in five of them. They are now `DEFAULT_MIN_THRUST` …
  `DEFAULT_PROP_MAX_FORCE` (plus `DEFAULT_MULTI_QUAD_MAX_THRUST` /
  `DEFAULT_MULTI_QUAD_MAX_TORQUE` for M4's larger limits). No value changed.
- **The `sdist` now ships `CHANGELOG.md`, `CITATION.cff`, `docs/`, `examples/`
  and `scripts/`**, in addition to `udaan/`, `tests/`, the README and the
  licence.
- **Generated MJCF is written to a user cache directory** (`$UDAAN_ASSET_DIR`,
  else the platform user cache) instead of into the installed package, so a
  read-only install works. Writes are atomic; cached files are published with
  ordinary (umask-derived) permissions rather than `mkstemp`'s `0600`, so a
  shared `$UDAAN_ASSET_DIR` and the warm-as-root-run-as-user container pattern
  both work; and `scene.xml` is copied beside every generated asset — including
  one written by `udaan generate-xml -o`, which previously could not be loaded
  from where it was written. A generator that writes nothing now raises instead
  of caching a 0-byte XML that no later run could recover from.
- `MJCF_GENERATOR_VERSION` 3 → 4: the M3 and fleet generators now emit
  `<option timestep="0.00125"/>`, so their backends advance exactly the `dt`
  they report. The fleets adopt the shared 5 ms outer loop (previously 2 ms).
- **The S2 substep is now `udaan.manif.zoh_step`**, shared by M2–M6, which keeps
  the cable/link rate exactly tangent instead of letting `q·ω` drift at O(h).

  This is a behaviour change: the gauge component is inert in the continuous
  dynamics, where `q̇ = ω × q` annihilates it, but the discrete step
  `q_next = expm(hat(ω h)) q` reads all of `ω`, so closed-loop trajectories
  differ slightly. For M5 and M6 it also removes a feedback error: their LQR
  error states build the link-rate block as `ω + q × (q × ω_d)`, i.e.
  `ω - (I - q qᵀ) ω_d`, which projects the reference rate but not the measured
  one, so any gauge drift entered the feedback directly.
- **The differential-flatness reference attitude composes with the
  thrust-derived attitude command** rather than being discarded: it supplies the
  heading, and the thrust command keeps `b₃`, which is the position loop's only
  path into the attitude command. On M4, passing `R_i = I` commands ψ = 0.
- `udaan.core.validation.check_positive` also rejects non-finite values, so an
  infinite mass, link length or inertia raises on M5 as well.
- The M3 asset generator is parameterised by formation radius, spawn height,
  cable length, payload mass and quad mass, all defaulting from
  `core/defaults.py`; previously only `nQ` was a parameter and the rest were
  literals in the function body.
- The per-quadrotor count kwarg was `num_quadrotors` on M3 and `nQ` on M4;
  both are now `num_quadrotors`, matching the CLI's `--num-quads`.

### Removed (breaking)

- **`udaan.models.base`** and the four standalone legacy models it held —
  `FloatingPointmass`, `S2Pendulum`, `PointmassSuspendedPayload`,
  `MultiPointmassSuspendedPayload`. They are **removed, not replaced**: none of
  the surviving systems reproduces them, and their tests were removed with them.

  The two payload models modelled the *quadrotor itself* as a point mass
  (`step()` took a raw force 3-vector, with no SO(3) state and no attitude
  loop), so no input mode of M2/M3, which take a thrust magnitude and a body
  moment, gives the same system. Also gone with them, with no equivalent anywhere:
  `get_rand_init_state`, `n_state` / `n_action` / `get_state_size` /
  `get_action_size`, `sim_timestep`, and the static `err_q` / `err_om` helpers.
  Anything depending on those interfaces has to be rewritten, not re-pointed.

  `BaseModel` itself survives as `udaan.models.fleet.BaseModel`, the shared base
  of the multi-agent models that are its only remaining users.
- **`MultiQuadrotorCSPointmass`** (`udaan/models/mujoco/multi_quad_cs_pointmass.py`)
  is deleted, superseded by the M3 package
  `udaan.models.multi_quad_cspayload.MultiQuadCsPayloadMujoco`. This is not a
  rename behind a compatibility alias — the replacement is a different model
  with a different vehicle abstraction (rigid bodies rather than point-mass
  force vectors) and a different state. Its `simulate()`/`reset()` payload
  keyword is `payload_position`, not `xL`. Defects in the retired model, fixed
  rather than carried over: `Cable.dq` stored a relative velocity rather than
  `q̇ = ω × q`; `ω` was recovered as `cross(q, dq)` with **no** `1/length`
  factor, so every cable rate was wrong by a factor of the cable length;
  `State.reset()` never reset the cables; `nQ` was re-derived from the MJCF
  rather than taken as authoritative; and the inlined feedforward applied
  `(mᵢ + m_L)·g` on **every** quadrotor, over-lifting the formation by
  `(nQ−1)·m_L·g` (now `mᵢ + m_L/nQ`). The payload spawn height was computed
  geometrically and then overwritten with a literal `1.34`, so every M3
  simulation began with 0.172 m of cable slack and a sag transient.
- **The old `MultiQuadRigidbody`** in `udaan/models/mujoco/`, superseded by the
  M4 package `udaan.models.multi_quad_rigidbody`. The retired version's
  `odefun_multiQuad_rigid` built every coupling block and then never assembled
  or solved them, returning six variables that were never assigned; the Python
  counterpart was a `pass` stub.
- **The legacy `flat2state` module**, replaced by `udaan.flatness`.
- `MujocoAssetCreator.create_cable_payload`, which had no callers.

### Moved

- MuJoCo backend infrastructure (`MujocoModel`, the GLFW viewer) is now
  `udaan.models.backend`; the multi-agent models (`QuadrotorFleet`,
  `QuadrotorCsPayloadFleet`, `QuadrotorComparison`) are now `udaan.models.fleet`.
  `udaan.models.mujoco` remains as a compatibility re-export of every name it
  previously exposed, including the `Quadrotor` alias. `udaan.models.mujoco` is
  a *module* now rather than a package, so its three public submodule paths —
  `udaan.models.mujoco.quadrotor_fleet`, `.quadrotor_cspayload_fleet`,
  `.quadrotor_comparison` — are aliased into `sys.modules`; imports written
  against them keep working and resolve to the same module objects.

### Fixed

- `udaan run quad-payload -m base` hung: any `--model` other than `mujoco`
  fell through to the vpython backend, which blocks even with `--no-render`.
- An unrecognised `--model` or `--trajectory` now exits with an error naming
  the valid choices instead of silently falling through to whichever backend
  or trajectory sat in the routing chain's `else`. Six such fall-throughs
  across the `run` commands are now errors.
- `QuadrotorMujoco(verbose=1)` raised `AttributeError`: it called
  `_logger.success(...)`, a leftover loguru idiom, on a stdlib `logging.Logger`.
- `QuadrotorVfx.simulate()` was a guaranteed `TypeError`: `step()` was declared
  `step(self, u)` while the inherited `simulate()` always calls
  `self.step(u, desired_att=...)`. Widening the signature also wires the VPython
  backend into the flatness feedforward path.
- Source docstrings that produced reStructuredText errors in the Sphinx build
  (`manif/S2.py`, `manif/SO3.py`, `models/quadrotor_cspayload/mujoco.py`,
  `utils/resize_gif.py`, `utils/trajectory/flip.py`), and the duplicate
  object descriptions emitted for every documented dataclass field and enum
  member. The documentation now builds without warnings. (GitHub issue #3.)
- `udaan/` is clean under `mypy` and gated in CI.
- The `-W` documentation CI job depended on network access: `-W` promotes an
  unreachable intersphinx inventory to a build failure. `conf.py` prefetches each inventory once into `docs/_build/inventories/`
  and drops any it cannot reach, so Sphinx reads local files and the gate is
  deterministic with or without network.

- **M1 quadrotor fixes:**
  - **Trajectories.** `SmoothTraj5`/`3`/`1` and `PolyTraj5` returned derivatives in
    normalised time (correct only at `t_f = 1`), and `PolyTraj5` solved its
    boundary conditions in those units too. `PolyTraj5` gains `get_full`.
  - **`FlipTrajectory`.**
    - Its flip phase ended after one loop while its end state was integrated over
      `n_loops`, so the reference jumped 4.9 m.
    - Its climb ended back at `start`, which made it dip 5.5 m below `start`. The
      climb and recovery now end at the Hermite displacement of their boundary
      conditions, so ż never reverses.
    - `n_loops` defaults to 1 (was 3, a 24 m manoeuvre).
    - Climb and recovery supply real jerk and snap.
    - Its reference attitude turned 180° about `b₃` for the inverted half of the
      loop: the flat map pins heading with `b₁d = (cos ψ, sin ψ, 0)`, which a
      pitch loop crosses (`b₃ = e₁`), so `b₂ = sign(cos θ)·e₂` flipped, and with
      it the heading hint and `R_d`. During the flip phase the reference is now the
      closed-form rotation about the fixed pitch axis, `R = exp(θ ê₂)`,
      `Ω = θ̇ e₂`, `Ω̇ = θ̈ e₂`, served by the new
      `FlipTrajectory.reference_attitude(t)` (`None` outside the flip phase).
      `QuadrotorBase.simulate()` prefers a trajectory's `reference_attitude`
      wherever it returns one and falls back to the flat map elsewhere; the
      hand-over is continuous because both agree while upright.
  - **`SpiralTrajectory`** stepped 2 m/s and 5 m/s² at both ends of the helix. Its
    phase is now time-warped through a degree-11 smoothstep (`ramp_time`, default
    2 s), so the reference is C⁶ everywhere, and it serves `get_jet(t, order)`. The helix phase
    is `ramp_time` longer. M4's flat map does not read `get_jet` and still
    zero-pads crackle and pop.
  - **Input paths.** `QuadrotorBase.simulate(input="prop_forces")` ignored the
    bound trajectory and flew to (0, 0, 1). All three input paths now fly the
    model's own position and attitude loops and the same flatness feedforward;
    the WRENCH path gains the feedforward too.
  - **Attitude-loop rate.** The three input paths closed the attitude loop at
    different rates: ACCELERATION inside `step()` (200 Hz on `QuadrotorBase`,
    800 Hz on `QuadrotorMujoco`, once per 1.25 ms MuJoCo step), WRENCH and
    PROP_FORCES in `simulate()` once per 5 ms step. Every path now closes it
    inside `step()` at one rate on both backends, set by the constructor
    keyword `attitude_rate_hz` (default
    `core.defaults.DEFAULT_ATTITUDE_RATE_HZ = 800`; `None` is once per step);
    it must be `k/dt` for a whole `k` dividing the substeps per step or the
    constructor raises `ValueError`. `QuadrotorMujoco`'s `attitude_zoh`
    keyword is removed and raises `TypeError`.
  - **`QuadrotorMujoco`** advanced 4 ms per `step()` while `dt` read 5 ms. It
    now writes MuJoCo's timestep as `dt/N` from `physics_rate_hz`, so a step
    advances exactly `dt`.
  - **`quadrotor_mj.xml`.** Its wrench `ctrlrange` re-clipped M_x and M_y to 3 N·m
    after the model's 5; the ranges now equal `core.defaults`. Its rotor
    positions and yaw gears were rounded; they now match the allocation matrix
    to 1e-10. Tests enforce both.
  - **`GeometricL1AttitudeController`.** Its default `c = 1` broke the Routh
    bound `c < k_Ω(k_R + Γ)/Γ` on every axis, and σ̂ hit its 30 N·m bound in
    0.18 s with no disturbance. `c` now defaults to half the bound per axis,
    tracks gain changes, and an explicit `c` past the bound logs a warning.
  - **`PositionL1Controller`** is rewritten as the piecewise-constant L1 law
    around `PositionPDController`. It drops the `control` dependency and the
    LQR, and no longer warns that it does not work. Its predictor uses the
    thrust the attitude actually delivers, `(u·b₃)b₃`, read from an optional
    third state element `(p, v, R)` that `QuadrotorBase.simulate()` and
    `QuadrotorFleet` now pass. Its matched estimate is unbiased: it is scaled
    by `(e^{aT}−1)/(aT)` so that a constant disturbance is estimated exactly.
    On MuJoCo it removes the steady error under a constant force and under a
    mass error to under 1 mm. `examples/quadrotor_geometric_l1.py` asserts
    that it beats PD.
  - **`PositionPDController`** silently replaced user-supplied `kp`/`kd` with
    the defaults.
  - **Airframe.** `QuadrotorBase` flew 0.9 kg while `QuadrotorMujoco` read
    0.75 kg from its MJCF. M1 now uses `DEFAULT_M1_QUAD_MASS`/`_INERTIA`, equal
    to the MJCF; the shared `DEFAULT_QUAD_*` that M2–M6 use are unchanged.
  - **M3 feedforward** reads `get_jet(t, 6)` from **any** trajectory that
    provides it (`SpiralTrajectory`, `CrazyTrajectory`, or a user's own),
    instead of zero-padding crackle and pop.
  - **The `l1` extra is removed** (from `pyproject.toml` and from `all`), along
    with the CLI's demo extras pre-flight check, which existed only for it.
    Nothing in `udaan` uses the `control` package any more:
    `PositionL1Controller` no longer solves an LQR, so
    `fleet --demo l1-comparison` and `examples/quadrotor_geometric_l1.py` run
    on a stock install, and `scripts/run_examples.sh` needs only `--extra dev`
    to run all seven examples. `pip install 'udaan[l1]'` now warns that the
    extra is unknown.
  - **Flip-rate ramps** (`flip_ramp_time`, default `flip_duration / 4`) make the
    flip's jerk and snap vanish at both ends, so the climb and recovery join it C⁴.
  - **CLI.** `udaan run quadrotor` flew the circle on the ground and started the
    spiral and flip there. Their default start is now 2 m up, and every
    trajectory starts the vehicle on its own reference.

### Known issues

- The composite `cable_model="cable"` backend for M2 remains experimental and
  its reset path is still a stub. The CLI warns when it is selected.
- **The `types` CI job is coupled to third-party stubs.** `mypy` is configured
  with `warn_unused_ignores = true`, and neither `mypy>=1.5` nor `numpy>=1.24`
  is pinned, so a release that makes any of the 43 `# type: ignore` comments in
  `udaan/` unnecessary turns the job red on an unrelated pull request. The
  setting stops stale ignores from accumulating; the remedy is to delete the
  unnecessary comment, not to weaken the setting.
- The `vfx` (VPython) backends are no longer offered by the CLI's `--model`.
  They block inside VPython and never return when driven headlessly,
  `--no-render` included. They remain importable as library classes.

## v1.3.0 (2026-04-22)

### Added
- `QuadrotorCsPayloadFleet` for side-by-side comparison of cable-suspended
  payload configurations, with a `cspayload-fleet` CLI command offering
  `same-gains` and `gain-sweep` demos.

### Changed
- `quadrotor_cspayload` restructured from a single module into a dedicated
  package (`base.py` / `mujoco.py` / `vfx.py`).
- Brighter default scene; GIFs re-recorded; plotting and recording dependencies
  added.

### Fixed
- `QuadrotorCsPayloadBase.reset()` mutated the caller's arrays.
- `create_flexible_cable_payload` namespaced; `rgba` kwargs bug in the asset
  creator.

## v1.2.1 (2026-04-11)

### Fixed
- `resize_gif` ceiling division; GIFs resized to a consistent 480×339.
- Lint and format: `os.path` replaced with `pathlib.Path`; `ruff format` clean.

## v1.2.0 (2026-04-11)

### Added
- Trajectory library (`udaan/utils/trajectory/`) with differential-flatness
  feedforward and CLI trajectory selection.
- Trail, start marker and target marker in the MuJoCo viewer, with a `--trail`
  flag and per-quad trails.
- SO(3) operators, a type-safe attitude controller, and integration tests.
- Bokeh debug tool.

### Changed
- Manifold types refactored from `ndarray` inheritance to composition.
- Quadrotor refactored into a clean class hierarchy; the old `Quadrotor` class
  removed in favour of `QuadrotorBase` / `QuadrotorMujoco`.

### Fixed
- L1 controller corrected for SO3/TSO3.
- Mutable default arguments in the manifold types; position PD gain scaling;
  Rodrigues small-angle accuracy.

## v1.1.1 (2026-04-09)

### Added
- `TSO3` and `TS2` tangent types with arithmetic operators, `SO3`/`S2`
  operators, static constructors, `SO3.config_error`, and `TS2.transport`.

### Changed
- Physical constants and control gains centralised in `core/defaults.py`.
- `printc` replaced with structured logging and a coloured formatter.
- Enums unified; dead code removed; naming conventions fixed.

### Fixed
- `SO3.__sub__` sign corrected to match the Lee (2010) convention.

## v1.0.2 (2026-04-09)

### Added
- Video recording and a tracking camera.
- PyPI publish workflow (trusted publisher, `release` environment).

### Changed
- README redesigned with logo, badges, feature highlights and GIFs; made
  PyPI-compatible.

## v1.0.1 (2026-04-08)

### Added
- Geometric L1 attitude controller.

### Changed
- Controllers split into individual files.
- Verbose fleet logging.

### Fixed
- Thrust computation.

## v1.0.0 (2026-04-08)

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
