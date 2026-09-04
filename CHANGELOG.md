# Changelog

## Unreleased

### Removed (breaking)
- `udaan.models.base` and the four standalone legacy models it held —
  `FloatingPointmass`, `S2Pendulum`, `PointmassSuspendedPayload`,
  `MultiPointmassSuspendedPayload`. They are **removed, not replaced**: none of
  the surviving systems reproduces them, and the 20 tests that covered them
  were deleted with no successor.

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

### Changed
- Generated MJCF is written to a user cache directory (`$UDAAN_ASSET_DIR`, else
  the platform user cache) instead of into the installed package, so a
  read-only install works. Writes are atomic, cached files are published with
  ordinary (umask-derived) permissions rather than `mkstemp`'s `0600` so a
  shared `$UDAAN_ASSET_DIR` and the warm-as-root-run-as-user container pattern
  both work, and `scene.xml` is copied beside every generated asset — including
  one written by `udaan generate-xml -o`, which previously could not be loaded
  from where it was written. A generator that writes nothing now raises instead
  of caching a 0-byte XML that no later run could recover from.
- `MJCF_GENERATOR_VERSION` 3 → 4: the M3 and fleet generators now emit
  `<option timestep="0.00125"/>`, so their backends advance exactly the `dt`
  they report. The fleets adopt the shared 5 ms outer loop (they had inherited
  2 ms from a legacy default).
- The S2 substep is now `udaan.manif.zoh_step`, shared by M2-M6, which keeps
  the cable/link rate exactly tangent instead of letting `q·ω` drift at O(h).

  This **is** a behaviour change, and the "the gauge component is dynamically
  inert" argument does not make it a no-op: it holds for the continuous
  dynamics, where `q̇ = ω × q` annihilates the component, and not for the
  discrete step, where `q_next = expm(hat(ω h)) q` reads all of `ω`. Measured
  over a 3 s closed-loop tracking run, the payload position moves by 2.81e-04 m
  (M2), 9.97e-05 m (M3), 5.83e-04 m (M4) and 4.58e-05 m (M5); M6's runs from
  its default spawn never excite `q·ω` at all and are unchanged below 1e-13 m.
  Every tolerance in the suite absorbs this.

  For M5 and M6 it is a **fix**, not a wash. Their LQR error states build the
  link-rate block as `ω + q × (q × ω_d)`, i.e. `ω - (I - q qᵀ) ω_d`: the
  reference rate is projected and the measured one is not, so the drifting
  gauge component was entering the feedback directly as an error the controller
  could neither reduce nor interpret. It is now zero to machine precision.
- The differential-flatness reference attitude now composes with the
  thrust-derived attitude command (it supplies the heading; the thrust command
  keeps `b₃`) rather than being discarded. Documentation that said passing
  `R_i = I` gives a bit-identical run — M4's `simulate`, its payload
  controller, and `docs/theory/dynamics/multi-quad-rigidbody.md` — has been
  corrected: `R_i = I` now commands ψ = 0.
- `udaan.core.validation.check_positive` adds a finiteness check that M5's
  private copy lacked, so an infinite mass, link length or inertia now raises
  where it was previously accepted.

### Added
- A yaw setpoint on the flatness feedforward path of M1, M3 and M4 —
  `QuadrotorBase.yaw_setpoint`, `MultiQuadCsPayloadBase.yaw_setpoint`,
  `MultiQuadRigidbodyBase.yaw_setpoint` — taking `ψ(t) -> (ψ, ψ̇, ψ̈)` and
  defaulting to `None`, which is the ψ ≡ 0 those paths hardcoded before. This
  is what makes the reference attitude's heading channel reachable at all:
  without it the flat map is always handed ψ ≡ 0 and the composition above is
  close to a no-op. On a brisk spiral at ψ̇ = 3 rad/s the mean flown heading
  error is 1.4°, against 91.1° with the reference discarded.
  `udaan.flatness.yaw_jet` is the shared helper that turns such a callable into
  the order-2 scalar jet the flat maps take.
- `MIN_ATTITUDE_HINT_CONDITIONING` guards the commanded-attitude construction
  against a near-degenerate heading hint. The reference's own `b₁` tilts with
  the reference and can line up with the commanded thrust direction, which the
  fixed world-`x` hint it replaced could not; measured on an M1 spiral at
  8 m/s, radius 2 m, ψ̇ = 3 rad/s, `‖b₃ × hint‖` fell to 0.0065 and the
  commanded attitude jumped 152° in one 5 ms step. Guarded, the same run floors
  conditioning at 0.31 and the worst jump at 31.2° — the well-conditioned
  hint's own figure.
- `udaan.core.validation.check_positive`, shared by every model. M2 and M3 now
  reject non-positive and non-finite masses, cable lengths and inertias at
  assignment rather than failing inside the integrator.

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
