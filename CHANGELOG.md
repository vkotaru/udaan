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
  body-frame points, and the team can exert a moment on it. Derived from
  scratch; no working reference existed. The derivation turns on projecting each
  quadrotor equation onto its own cable direction, which annihilates the cable
  angular acceleration and makes the tension affine in `(a_L, Ω̇_L)` alone.
- **`quadrotor_flexcable` (M5)** — a quadrotor carrying a payload through a
  flexible cable of `n` rigid links. Previously this existed only as M2's
  `cable_model="links"` backend option, which simulated a link chain and then
  collapsed it back to a point-mass state; the flexible cable was simulated but
  never observed as one, and there was no analytical model at all. Now derived
  from the Lagrangian as a symmetric `(3+3n)` block solve.
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
    `leibniz_product`, `leibniz_dot`, `leibniz_matmul`, `normalize_derivatives`.
    This replaces, at arbitrary order, a 9,422-line generated Mathematica dump
    (six copies, 47,958 lines, hard-capped at ten links) that the reference
    implementation used.
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
  pattern. Feedforward improves tracking 868–957× across seven configurations.

#### Other

- A yaw setpoint on the flatness feedforward path of M1, M3 and M4 —
  `QuadrotorBase.yaw_setpoint`, `MultiQuadCsPayloadBase.yaw_setpoint`,
  `MultiQuadRigidbodyBase.yaw_setpoint` — taking `ψ(t) -> (ψ, ψ̇, ψ̈)` and
  defaulting to `None`, which is the ψ ≡ 0 those paths hardcoded before. This is
  what makes the reference attitude's heading channel reachable at all. On a
  brisk spiral at ψ̇ = 3 rad/s the mean flown heading error is 1.4°, against
  91.1° with the reference discarded.
- `MIN_ATTITUDE_HINT_CONDITIONING` guards the commanded-attitude construction
  against a near-degenerate heading hint. The reference's own `b₁` tilts with
  the reference and can line up with the commanded thrust direction, which the
  fixed world-`x` hint it replaced could not; measured on an M1 spiral at 8 m/s,
  radius 2 m, ψ̇ = 3 rad/s, `‖b₃ × hint‖` fell to 0.0065 and the commanded
  attitude jumped 152° in one 5 ms step. Guarded, the same run floors
  conditioning at 0.31 and the worst jump at 31.2° — the well-conditioned hint's
  own figure.
- `udaan.core.validation.check_positive`, shared by every model. M2 and M3 now
  reject non-positive and non-finite masses, cable lengths and inertias at
  assignment rather than failing inside the integrator.
- `udaan.manif.zoh_step` — the shared S2 substep used by M2–M6.
- `udaan.utils.grasp` and `udaan.utils.hose` — geometry helpers for the M4 grasp
  map and the M6 hose vertex layout.
- `udaan.utils.assets.cache` — the user-cache asset store described under
  *Changed*.
- CLI commands for the four new systems, plus `--cable-length`,
  `--payload-mass` and `--radius` options on `multi-quad`.
- Theory documentation for every system (dynamics, controllers and
  differential-flatness pages), a `tests/conftest.py` with a seeded `rng` and a
  calibrated tolerance ladder, and a Sphinx docs build script.

### Changed

- **`reset()` and `simulate()` now reject an unrecognised keyword** instead of
  discarding it. Every model takes its initial condition through `**kwargs`, and
  a keyword no branch matched used to be dropped silently, so the run went ahead
  from the default state with a clean exit code. That is the worst failure mode
  a MAJOR break can have: v1.3.0's own CLI called `simulate(tf=time, xL=x0)` on
  the now-deleted `MultiQuadrotorCSPointmass`, and the M3 model that replaces it
  names the same quantity `payload_position`, so a 1.x user who only re-pointed
  the import got a silently different initial condition
  (`simulate(tf=0.5, xL=[-1,2,0.5])` left the payload at `[3.4e-05, 0, 1.079]`
  rather than near `[-1, 2, 0.5]`). It now raises `ValueError`. The rule is
  applied across every model that takes reset kwargs — M1–M6, both backends,
  and the three fleet models — through the shared `udaan.models._kwargs`
  helper; the three MuJoCo backends that already rejected unknown keywords keep
  their wording.
- **Actuator limits moved to `udaan.core.defaults`.** `_min_thrust`,
  `_max_thrust`, `_min_torque`, `_max_torque`, `_prop_min_force` and
  `_prop_max_force` were literals in six separate `base.py` constructors,
  identical in five of them. They are now `DEFAULT_MIN_THRUST` …
  `DEFAULT_PROP_MAX_FORCE` (plus `DEFAULT_MULTI_QUAD_MAX_THRUST` /
  `DEFAULT_MULTI_QUAD_MAX_TORQUE` for M4's larger limits), which is where the
  repo's own convention says constants live and where the theory docs already
  said to look. No value changed.
- **The `sdist` now ships `CHANGELOG.md`, `CITATION.cff`, `docs/`, `examples/`
  and `scripts/`.** It previously carried only `udaan/`, `tests/`, the README
  and the licence, so the citation metadata this release adds never reached a
  source install, and neither did `scripts/run_examples.sh` — the command the
  README presents as the way to verify an install.
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
  they report. The fleets adopt the shared 5 ms outer loop (they had inherited
  2 ms from a legacy default).
- **The S2 substep is now `udaan.manif.zoh_step`**, shared by M2–M6, which keeps
  the cable/link rate exactly tangent instead of letting `q·ω` drift at O(h).
  Drift falls from 3.2e-3…1.0e0 to ~1e-16 across M2 through M6.

  This **is** a behaviour change, and the "the gauge component is dynamically
  inert" argument does not make it a no-op: it holds for the continuous
  dynamics, where `q̇ = ω × q` annihilates the component, and not for the
  discrete step, where `q_next = expm(hat(ω h)) q` reads all of `ω`. Measured
  over a 3 s closed-loop tracking run, the payload position moves by 2.81e-04 m
  (M2), 9.97e-05 m (M3), 5.83e-04 m (M4) and 4.58e-05 m (M5); M6's runs from its
  default spawn never excite `q·ω` at all and are unchanged below 1e-13 m. Every
  tolerance in the suite absorbs this.

  For M5 and M6 it is a **fix**, not a wash. Their LQR error states build the
  link-rate block as `ω + q × (q × ω_d)`, i.e. `ω - (I - q qᵀ) ω_d`: the
  reference rate is projected and the measured one is not, so the drifting gauge
  component was entering the feedback directly as an error the controller could
  neither reduce nor interpret. It is now zero to machine precision.
- **The differential-flatness reference attitude now composes with the
  thrust-derived attitude command** (it supplies the heading; the thrust command
  keeps `b₃`) rather than being discarded. Feeding it through wholesale is not
  the fix — that `R_d` is the position loop's only path into the attitude
  command, and substituting the open-loop reference severs position feedback,
  20× worse tracking on three systems. The two now meet on orthogonal degrees of
  freedom instead. Documentation that said passing `R_i = I` gives a
  bit-identical run — M4's `simulate`, its payload controller, and
  `docs/theory/dynamics/multi-quad-rigidbody.md` — has been corrected:
  `R_i = I` now commands ψ = 0.
- `udaan.core.validation.check_positive` adds a finiteness check that M5's
  private copy lacked, so an infinite mass, link length or inertia now raises
  where it was previously accepted.
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
  the surviving systems reproduces them, and the 20 tests that covered them were
  deleted with no successor.

  The two payload models are often described as point-mass reductions of
  `quadrotor_cspayload` (M2) and `multi_quad_cspayload` (M3), and that is only
  half right. They modelled the *quadrotor itself* as a point mass: `step()`
  took a raw force 3-vector, there was no SO(3) state and no attitude loop, so
  no input mode of M2/M3 — which take a thrust magnitude and a body moment —
  gives the same system. Also gone with them, with no equivalent anywhere:
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

- `udaan run quad-payload -m base` hung. The command routed only two ways, so
  any `--model` other than `mujoco` — including the documented `base` — fell
  through to the vpython backend, which blocks even with `--no-render`. This
  was the only command with that defect: `quad-flexcable` and `hose` already
  built their analytical models, and `hose` already validated `--trajectory`;
  what they gained is the shared `--model` guard below.
- An unrecognised `--model` or `--trajectory` now exits with an error naming
  the valid choices instead of silently falling through to whichever backend
  or trajectory sat in the routing chain's `else`. Six such fall-throughs
  across the `run` commands are now errors.
- `udaan run fleet --demo l1-comparison` failed with a bare
  `ModuleNotFoundError: No module named 'control'` when the optional `l1` extra
  was not installed. The requirement is now checked up front — before the fleet
  or its viewer is built — and reported as a message naming the extra.
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
- `udaan/` is now clean under `mypy` (192 errors → 0) and gated in CI.
- The new `-W` documentation CI job was non-deterministic: `docs/conf.py` mapped
  three remote intersphinx inventories, and `-W` promotes an unreachable one to
  a build failure, so the job passed or failed on network weather (measured: 1,
  1, 0 over three runs of the literal CI command; a deterministic red offline).
  No `intersphinx_timeout` fixes this — it bounds the wait, not the warning.
  `conf.py` now prefetches each inventory once into `docs/_build/inventories/`
  and drops any it cannot reach, so Sphinx reads local files and the gate is
  deterministic with or without network.
- `scripts/run_examples.sh` reported `SKIP` for the L1 example under its own
  documented invocation: it ran `uv run --extra dev python`, which never
  installs the `l1` extra. It now asks for both extras, so all seven examples
  actually run.

### Known issues

- **`PositionL1Controller` is experimental and has never adapted.** Two
  independent defects, both established by instrumenting a live run:

  1. `compute()` assigns `self.last_t = t` at the *top* of the method, before
     `dt = t - self.last_t` is formed below it, so `dt` is always `0` and is
     clamped to `1e-6` — 5000x smaller than the real 5 ms step. The adaptation
     integrator, the MRAC reference model and the low-pass filter are frozen,
     and the controller reduces to its LQR baseline.
  2. The MRAC reference state was *bound* to the plant's own arrays rather than
     copied (`self.mrac_position = pos`). On `QuadrotorBase` —
     the backend the example, the README and the docs all use —
     `self.mrac_position is mdl.state.position` was `True` on every call, so
     `mrac_eta` was identically `eta`, `eta_tilde` identically zero and `delta`
     identically zero, **exactly**, for a whole run and at *any* `Gamma`. It
     also let the controller write into the plant: 4.2e-4 m of position and
     4.4e-2 m/s of velocity injected over a 25 s run. **This one is fixed** —
     the states are copied — because a controller mutating the plant's state is
     a correctness bug independently of the adaptation. (The MuJoCo backend was
     immune: `_query_latest_state()` rebuilds those arrays every step.)

  Defect 1 is deliberately **not** fixed, and the previous release note about it
  was wrong. Deleting the line does not unfreeze the adaptation: with that fix
  alone `delta` still ends a run at exactly `0`. The 0.2451 m -> 0.1606 m
  "improvement" this changelog previously quoted for it was defect 2 —- the
  controller corrupting the plant's velocity, by up to 0.143 m/s in one step —
  not tracking. With both defects fixed the run diverges to NaN
  (`Gamma*dt ~ 500` at the real 5 ms step, far outside the stability limit of
  the explicit Euler update; `udaan run fleet --demo l1-comparison` NaNs at
  t = 0.34 s), and no gain rescues it: sweeping `Gamma` over 1000 / 200 / 100
  stays finite but leaves 0.72-0.89 m of z-error in the fleet demo against the
  PD baseline's 0.001 m, and 0.55-0.64 m on the single quadrotor against PD's
  0.303 m.

  So this is not a working feature with a known one-line fix. The class is now
  labelled **experimental**, warns on construction, and
  `examples/quadrotor_geometric_l1.py` no longer asserts that it beats PD — it
  does not (0.3096 m against 0.3028 m). Finishing it means redesigning the
  adaptation update (most plausibly the piecewise-constant adaptation law in
  place of explicit Euler), retuning `Gamma`, and adding a regression test.
- The smooth projection in `PositionL1Controller.compute()` computes
  `proj = y - (df.y / ||df||^2) * df * f_` without clamping `f_` to `1`, so once
  `||delta||` exceeds `deltamax` the correction is scaled by `f_ > 1` and
  over-subtracts, which would accelerate rather than arrest a divergence.
- Neither `PositionL1Controller` nor `GeometricL1AttitudeController` has any
  test coverage.
- `position_l1.py` documents `lpf_cutoff` as `[Hz]` but uses it as a pole in
  rad/s; the comment is what is wrong.
- The composite `cable_model="cable"` backend for M2 remains experimental and
  its reset path is still a stub. The CLI warns when it is selected.
- **The `types` CI job is coupled to third-party stubs.** `mypy` is configured
  with `warn_unused_ignores = true`, and neither `mypy>=1.5` nor `numpy>=1.24`
  is pinned, so a release that makes any of the 43 `# type: ignore` comments in
  `udaan/` unnecessary turns the job red on an unrelated pull request. Accepted
  deliberately — the setting is what stops stale ignores from accumulating — but
  the fix when it happens is to delete the now-unnecessary comment, not to
  weaken the setting. The same fragility class is why the `docs` job no longer
  fetches intersphinx inventories during the build.
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
