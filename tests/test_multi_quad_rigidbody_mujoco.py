"""Tests for MultiQuadRigidbodyMujoco — MJCF wiring, state sync, backend agreement.

The MJCF is generated from the construction parameters now (see
``tests/test_multi_quad_rigidbody_xml.py`` for the generator itself), so what is
worth testing here is the *correspondence* between the loaded file and the
analytical model: the quadrotor count, the attachment sites, which tendon joins
which pair of them, and the inertial properties. Each of those was silently
wrong in the retired ``models/mujoco/multi_quad_rigidbody.py``, and each is now
checked at load time.

:class:`TestBackendAgreement` is the piece M4 had none of. It could not be
written against the retired hand-written asset: that one spawned 21% slack, so
it would have been comparing a taut analytical model against a simulated one in
free fall and measuring the drop rather than an integration difference. With
the generated asset both backends start from the same taut formation and the
comparison means something.
"""

import mujoco
import numpy as np
import pytest

from tests.conftest import TOL_ANALYTIC, TOL_TIGHT
from udaan.core.defaults import (
    DEFAULT_QUAD_MASS,
    DEFAULT_RIGID_PAYLOAD_INERTIA,
    GRAVITY,
)
from udaan.manif import S2, SO3, TS2, TSO3
from udaan.models.multi_quad_rigidbody import (
    MultiQuadRigidbodyBase,
    MultiQuadRigidbodyMujoco,
)

pytestmark = pytest.mark.mujoco

#: The generated asset's own count, used where the test is about the wiring
#: rather than about ``nQ``.
MJCF_NUM_QUADROTORS = 4


def attitude_angle_deg(R, R_d) -> float:
    """Rotation angle between two attitudes, in degrees."""
    err = np.asarray(R_d).T @ np.asarray(R)
    return float(np.degrees(np.arccos(np.clip((np.trace(err) - 1.0) / 2.0, -1.0, 1.0))))


@pytest.fixture
def mdl():
    return MultiQuadRigidbodyMujoco(render=False, num_quadrotors=MJCF_NUM_QUADROTORS)


class TestMujocoSetup:
    """Model loading, index resolution and parameters read back from the MJCF."""

    def test_instantiate(self, mdl):
        assert mdl.num_quadrotors == MJCF_NUM_QUADROTORS
        assert mdl._mjMdl.model.nu == 4 * MJCF_NUM_QUADROTORS
        assert len(mdl.state.positions) == MJCF_NUM_QUADROTORS

    @pytest.mark.parametrize("nQ", [2, 3, 5, 6])
    def test_the_quadrotor_count_is_a_parameter_now(self, nQ):
        """``num_quadrotors`` used to be fixed at the hand-written asset's four."""
        mdl = MultiQuadRigidbodyMujoco(render=False, num_quadrotors=nQ)
        assert mdl.num_quadrotors == nQ
        assert mdl._mjMdl.model.nu == 4 * nQ
        assert mdl._mjMdl.model.ntendon == nQ
        assert len(mdl.state.positions) == nQ

    def test_attachment_points_are_validated_against_the_sites(self, mdl):
        """The grasp geometry must agree with the ``pyld_poc`` sites.

        The retired model's hardcoded grasp matrix had ``(0.5, -0.5)`` where
        ``pyld_poc2`` sits at ``(-0.5, -0.5)``, so two of its four columns were
        duplicates and one corner of the slab was never represented. Nothing
        compared the two. The sites come from the same ring the model uses now,
        but the comparison stays: a stale cached asset would otherwise put the
        analytical grasp and the physics on different geometries in silence.
        """
        model = mdl._mjMdl.model
        sites = np.array([model.site_pos[i] for i in mdl._attachment_site_id])
        np.testing.assert_allclose(sites, mdl.attachment_points, atol=1e-9)
        # All four attachments distinct — the duplicate-column bug, as a property.
        assert len({tuple(np.round(p, 9)) for p in mdl.attachment_points}) == 4

    def test_mismatched_attachment_points_are_caught(self, mdl):
        """The check is constructive: move a site and the model must object."""
        model = mdl._mjMdl.model
        original = model.site_pos[mdl._attachment_site_id[0]].copy()
        try:
            model.site_pos[mdl._attachment_site_id[0]] += np.array([0.1, 0.0, 0.0])
            with pytest.raises(RuntimeError, match="disagree with the MJCF"):
                mdl._read_params_from_mjcf()
        finally:
            model.site_pos[mdl._attachment_site_id[0]] = original

    def test_quad_mass_is_the_whole_body_not_the_centre_box(self, mdl):
        """The body mass is the sum of every geom, and it is the requested total.

        The MJCF's ``quad{i}_geom`` box is only the chassis; each quadrotor also
        carries four arms and four propeller discs. The retired model hardcoded
        the chassis mass alone (0.75 kg against a 0.99 kg body), so every
        weight feedforward was 24% light. The generator now scales all eight
        geoms to hit the requested total, which is
        :data:`~udaan.core.defaults.DEFAULT_QUAD_MASS` here.
        """
        np.testing.assert_allclose(mdl.masses, np.full(4, DEFAULT_QUAD_MASS), atol=1e-9)
        model = mdl._mjMdl.model
        for i, body_id in enumerate(mdl._quad_body_id):
            assert model.body_mass[body_id] == pytest.approx(mdl.masses[i])
            # ... and it really is the sum of eight geoms, not one.
            geoms = [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
                for g in range(
                    model.body_geomadr[body_id],
                    model.body_geomadr[body_id] + int(model.body_geomnum[body_id]),
                )
            ]
            assert f"quad{i}_geom" in geoms
            assert len(geoms) > 1

    def test_payload_inertia_matches_the_analytical_default(self, mdl):
        """MuJoCo's ``inertiafromgeom`` and ``DEFAULT_RIGID_PAYLOAD_INERTIA`` agree."""
        assert mdl.payload_mass == pytest.approx(1.0)
        np.testing.assert_allclose(
            mdl.payload_inertia, DEFAULT_RIGID_PAYLOAD_INERTIA, atol=TOL_ANALYTIC
        )

    def test_params_read_from_mjcf(self, mdl):
        nQ = mdl.num_quadrotors
        assert mdl.masses.shape == (nQ,)
        assert np.all(mdl.masses > 0.0)
        assert mdl.inertias.shape == (nQ, 3, 3)
        assert mdl.cable_lengths.shape == (nQ,)
        assert np.all(mdl.cable_lengths > 0.0)

    def test_inertias_cannot_be_passed_to_the_constructor(self):
        with pytest.raises(ValueError, match="derives inertias"):
            MultiQuadRigidbodyMujoco(render=False, inertias=np.eye(3))

    def test_tendon_wiring_is_checked(self, mdl):
        """Tendon ``i`` joins ``quad{i}_end1`` to ``pyld_poc{i}``.

        Nothing in the MJCF enforces the pairing, and it is what makes
        ``attachment_points[i]`` belong to cable ``i``. The check is
        constructive: swapping two tendons in a copy of the model must trip it.
        """
        model = mdl._mjMdl.model
        assert model.ntendon == mdl.num_quadrotors
        original = model.wrap_objid.copy()
        try:
            start_a = int(model.tendon_adr[0])
            start_b = int(model.tendon_adr[1])
            model.wrap_objid[start_a + 1], model.wrap_objid[start_b + 1] = (
                model.wrap_objid[start_b + 1],
                model.wrap_objid[start_a + 1],
            )
            with pytest.raises(RuntimeError, match="tendon 0 joins"):
                mdl._check_tendon_wiring()
        finally:
            model.wrap_objid[:] = original
        mdl._check_tendon_wiring()

    def test_grasp_map_has_full_rank(self, mdl):
        """Four attachments on a ring can produce any wrench."""
        assert np.linalg.matrix_rank(mdl.grasp_map) == 6

    def test_the_loaded_geometry_can_hover(self, mdl):
        """Both hover residuals at round-off — and they are different numbers.

        ``hover_residual`` is the *controller's*, solved from its constructor
        arguments; ``mjcf_hover_residual`` asks whether the loaded file hovers
        under that trim. Only the second can see a stale or hand-edited asset,
        which is the case both were written for. See
        ``test_a_stale_asset_with_the_wrong_formation_radius_is_reported``.
        """
        assert mdl.payload_controller.hover_residual < 1e-12
        assert np.all(mdl.payload_controller.hover_tensions > 0.0)
        assert mdl.mjcf_hover_residual < 1e-9

    def test_the_file_derived_hover_residual_reads_the_file(self, mdl):
        """Move the file's quadrotors and the file-derived residual must move.

        The controller's own residual is blind to this by construction — it
        never touches the MJCF — so this is the only one of the two that is a
        check on the loaded geometry at all.
        """
        before = mdl.mjcf_hover_residual
        assert before < 1e-9
        model, data = mdl._mjMdl.model, mdl._mjMdl.data
        adr = mdl._quad_qposadr[0]
        original = data.qpos.copy()
        try:
            data.qpos[adr] += 0.25
            mujoco.mj_forward(model, data)
            mdl._query_latest_state()
            assert mdl.mjcf_hover_residual > 0.1
            # ... while the controller's is unchanged, which is the point.
            assert mdl.payload_controller.hover_residual < 1e-12
        finally:
            data.qpos[:] = original
            mujoco.mj_forward(model, data)
            mdl._query_latest_state()

    def test_a_substituted_controller_without_hover_tensions_is_tolerated(self, mdl):
        """``mjcf_hover_residual`` is ``None`` rather than an error there."""
        mdl._payload_controller = object()
        assert mdl.mjcf_hover_residual is None
        mdl._check_spawn_is_taut()


class TestMujocoStateSync:
    """``_query_latest_state`` pulls the right quantities out of MuJoCo."""

    def test_state_types(self, mdl):
        mdl.reset()
        assert isinstance(mdl.state.payload_orientation, SO3)
        assert isinstance(mdl.state.payload_angular_velocity, TSO3)
        for i in range(mdl.num_quadrotors):
            assert isinstance(mdl.state.orientations[i], SO3)
            assert isinstance(mdl.state.angular_velocities[i], TSO3)
            assert isinstance(mdl.state.cable_attitudes[i], S2)
            assert isinstance(mdl.state.cable_angular_velocities[i], TS2)

    def test_reset_places_the_payload_absolutely(self, mdl):
        """``payload_position`` is where the payload ends up, not a displacement.

        The retired model added it to every body's ``qpos``, so the same flag
        under the same CLI help text meant an offset there and an absolute
        position for ``multi-quad``.
        """
        target = np.array([-1.0, 2.0, 1.5])
        mdl.reset(payload_position=target)
        np.testing.assert_allclose(mdl.state.payload_position, target, atol=TOL_ANALYTIC)
        assert mdl.t == 0.0
        # Idempotent: resetting to the same target twice must not drift.
        mdl.reset(payload_position=target)
        np.testing.assert_allclose(mdl.state.payload_position, target, atol=TOL_ANALYTIC)
        for i in range(mdl.num_quadrotors):
            assert mdl.state.positions[i][2] > mdl.state.payload_position[2]

    def test_reset_rejects_the_overrides_it_cannot_apply(self, mdl):
        """A documented base kwarg used to be accepted and then ignored.

        ``simulate(tf, payload_orientation=...)` ran the MJCF's own level
        formation and reported nothing, so the run answered a different
        question from the one asked. The initial pose here is fixed by the
        tendon constraints, so the honest answer is an error.
        """
        for kwargs in (
            {"payload_orientation": np.eye(3)},
            {"payload_velocity": np.zeros(3)},
            {"cable_attitudes": [np.array([0.0, 0.0, -1.0])] * MJCF_NUM_QUADROTORS},
            {"not_a_state_at_all": 1.0},
        ):
            with pytest.raises(ValueError, match="cannot honour"):
                mdl.reset(**kwargs)
        with pytest.raises(ValueError, match="cannot honour"):
            mdl.simulate(tf=0.05, payload_orientation=np.eye(3))
        # The one it can apply still works.
        mdl.reset(payload_position=np.array([0.0, 0.0, 1.2]))

    def test_reset_translates_the_whole_formation(self, mdl):
        mdl.reset()
        base = [p.copy() for p in mdl.state.positions]
        base_payload = mdl.state.payload_position.copy()
        target = np.array([0.5, -0.5, 2.0])
        mdl.reset(payload_position=target)
        shift = target - base_payload
        for i in range(mdl.num_quadrotors):
            np.testing.assert_allclose(mdl.state.positions[i], base[i] + shift, atol=TOL_ANALYTIC)

    def test_cables_are_measured_from_the_attachment_not_the_payload_centre(self, mdl):
        """The whole difference from M3, and the thing the sites exist for."""
        mdl.reset(payload_position=np.array([0.0, 0.0, 1.2]))
        for i in range(mdl.num_quadrotors):
            q = np.asarray(mdl.state.cable_attitudes[i])
            assert float(np.linalg.norm(q)) == pytest.approx(1.0, abs=TOL_TIGHT)
            direction = mdl.attachment_position(i) - mdl.state.positions[i]
            np.testing.assert_allclose(q, direction / np.linalg.norm(direction), atol=TOL_ANALYTIC)
            # ... and it is genuinely a different vector from the M3 answer.
            centre = mdl.state.payload_position - mdl.state.positions[i]
            assert float(np.linalg.norm(q - centre / np.linalg.norm(centre))) > 1e-3

    def test_payload_pose_is_read_back(self, mdl):
        mdl.reset(payload_position=np.array([0.0, 0.0, 1.2]))
        np.testing.assert_allclose(
            np.asarray(mdl.state.payload_orientation), np.eye(3), atol=TOL_ANALYTIC
        )
        np.testing.assert_allclose(
            np.asarray(mdl.state.payload_angular_velocity), np.zeros(3), atol=TOL_ANALYTIC
        )

    def test_attachment_positions_match_the_mujoco_sites(self, mdl):
        """``x_L + R_L r_i`` and MuJoCo's own ``site_xpos`` are the same point."""
        mdl.reset(payload_position=np.array([0.3, -0.2, 1.5]))
        for i in range(mdl.num_quadrotors):
            np.testing.assert_allclose(
                mdl.attachment_position(i),
                mdl._mjMdl.data.site_xpos[mdl._attachment_site_id[i]],
                atol=TOL_ANALYTIC,
            )


class TestMujocoSetters:
    """Property setters write through into the MuJoCo model."""

    def test_mass_setters(self, mdl):
        mdl.masses = [1.1, 1.2, 1.3, 1.4]
        for i, body_id in enumerate(mdl._quad_body_id):
            assert mdl._mjMdl.model.body_mass[body_id] == pytest.approx(mdl.masses[i])
        mdl.payload_mass = 0.42
        assert mdl._mjMdl.model.body_mass[mdl._payload_body_id] == pytest.approx(0.42)

    def test_payload_inertia_setter(self, mdl):
        mdl.payload_inertia = np.array([0.2, 0.3, 0.4])
        np.testing.assert_allclose(
            mdl._mjMdl.model.body_inertia[mdl._payload_body_id], [0.2, 0.3, 0.4], atol=TOL_ANALYTIC
        )

    def test_cable_length_setter_moves_the_tendon_limit(self, mdl):
        mdl.cable_lengths = [0.9, 0.95, 1.0, 1.05]
        np.testing.assert_allclose(
            mdl._mjMdl.model.tendon_range[:, 1], [0.9, 0.95, 1.0, 1.05], atol=TOL_ANALYTIC
        )


class TestMujocoClosedLoop:
    """The cascade runs against the physics backend."""

    @pytest.mark.slow
    def test_simulate_holds_the_payload_at_its_setpoint(self, mdl):
        """The formation spawns taut now, so this is a real tracking assertion.

        Against the retired slack asset the payload dropped ~0.2 m before the
        cables caught and the only honest assertion was "nothing went
        non-finite". With the generated spawn the cascade holds station from
        ``t = 0``.
        """
        target = np.array([0.3, -0.2, 1.2])
        mdl.payload_controller.setpoint = lambda t: (target, np.zeros(3), np.zeros(3))
        mdl.simulate(tf=8.0, payload_position=target)

        assert np.all(np.isfinite(mdl.state.payload_position))
        assert float(np.linalg.norm(mdl.state.payload_position - target)) < 0.05
        R = np.asarray(mdl.state.payload_orientation)
        angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))
        assert angle < 5.0
        for i in range(mdl.num_quadrotors):
            assert mdl.state.positions[i][2] > mdl.state.payload_position[2]

    @pytest.mark.slow
    def test_the_flatness_feedforward_runs_through_this_backend(self, monkeypatch):
        """``simulate``'s feedforward path, exercised against the physics.

        Every other feedforward test in the suite runs on
        :class:`MultiQuadRigidbodyBase`, where ``step`` is the analytical
        integrator the flat map was derived against — so the whole chain could
        have been inert here without a single failure. This backend overrides
        ``step`` and loops over ``_step_iter`` MuJoCo substeps, and
        ``desired_att`` has to reach ``_repackage_input`` on each of them.
        """
        from udaan.utils.trajectory import SpiralTrajectory

        traj = SpiralTrajectory(
            start=np.array([0.0, 0.0, 1.4]),
            radius=0.5,
            speed=1.2,
            n_loops=1,
            advance=0.6,
            hold_time=0.25,
        )
        applied: list[int] = []
        original = MultiQuadRigidbodyBase._desired_attitudes

        def spy(self, flat_map, flat_fn, t):
            out = original(self, flat_map, flat_fn, t)
            applied.append(0 if out is None else len(out))
            return out

        monkeypatch.setattr(MultiQuadRigidbodyBase, "_desired_attitudes", spy)

        def run(feedforward):
            mdl = MultiQuadRigidbodyMujoco(render=False, num_quadrotors=MJCF_NUM_QUADROTORS)
            mdl.payload_controller.setpoint = (
                traj.get_full if feedforward else (lambda t: traj.get_full(t)[:3])
            )
            mdl.simulate(tf=traj.duration, payload_position=traj.get(0.0)[0])
            return mdl, float(np.linalg.norm(mdl.state.payload_position - traj.get(mdl.t)[0]))

        mdl_ff, err_ff = run(True)
        assert len(applied) > 100, "simulate() never asked for a feedforward attitude"
        assert set(applied) == {MJCF_NUM_QUADROTORS}, "some step ran without the attitudes"
        assert mdl_ff.feedforward_drops == 0

        applied.clear()
        _, err_fb = run(False)
        assert not applied, "the plain-closure setpoint must not unlock the feedforward"

        assert err_fb > 0.1, "trajectory too easy — the comparison proves nothing"
        assert err_ff < err_fb

    def test_step_accepts_the_base_input_pipeline(self, mdl):
        mdl.reset(payload_position=np.array([0.0, 0.0, 1.2]))
        weight = (mdl.payload_mass / mdl.num_quadrotors + mdl.masses[0]) * GRAVITY
        u = np.tile(np.array([0.0, 0.0, weight]), (mdl.num_quadrotors, 1))
        for _ in range(20):
            mdl.step(u)
        assert mdl.t > 0.0
        assert np.all(np.isfinite(mdl.state.payload_velocity))


@pytest.mark.slow
@pytest.mark.integration
class TestBackendAgreement:
    """The analytical model and MuJoCo describe the same machine.

    M4 had no such test at all, and one could not have been written before the
    MJCF was generated: the retired asset spawned 0.7887 m of cable against a
    1.0 m tendon limit, so a comparison would have measured the 0.2 m drop of
    a slack formation against an analytical model that has no slack mode. Both
    backends now start from the same taut ring.

    The two are *not* expected to agree to round-off, and the tolerance says
    which differences are being tolerated. They are different integrators of
    the same equations: the base model runs ZOH Euler at 1.25 ms on a
    constraint-eliminated ``6 x 6`` reduction where the cable is exact and
    inextensible by construction, while MuJoCo runs its own integrator with the
    cable as a *tendon limit constraint* — soft, with a solver-dependent
    penetration. Divergence is therefore expected to grow with time, and the
    honest form of the assertion is a bounded discrepancy over a short horizon,
    not equality.

    Two things that used to be absorbed into the tolerance are not any more,
    because neither was an integration difference:

    * The generated MJCF used to inherit ``MujocoAssetCreator.joint``'s default
      ``damping=0.1`` on the quadrotors' and the payload's *free* joints — a
      viscous drag on every degree of freedom that
      :class:`MultiQuadRigidbodyBase` has no term for. Removing it dropped the
      peak position discrepancy over the step manoeuvre from 2.12e-2 m to
      7.29e-3 m and the peak attitude discrepancy from 2.34 deg to 0.85 deg.
    * MuJoCo's default 2 ms timestep does not divide the model's 5 ms ``dt``,
      so the backend took ``round(2.5) = 2`` substeps and advanced 4 ms per
      ``step()`` while reporting 5 ms. Over a 2 s run MuJoCo ended at
      t = 2.00000 against the base model's t = 2.00500 and the two were
      compared at different simulation times. The asset now sets
      :data:`~udaan.core.defaults.DEFAULT_MJCF_TIMESTEP`, so both backends
      march on exactly the same grid and the traces below are compared sample
      for sample.

    The assertions run over the **whole horizon**, not at the endpoint. The
    endpoint is not the worst case and is not even a conservative proxy for it:
    on the step manoeuvre the position discrepancy peaks at 6.8e-3 m around
    t = 1.24 against 3.1e-3 m at t = 2.0, and before the damping was removed
    the attitude discrepancy peaked at 2.34 deg — over the 2.0 deg tolerance
    this class declared — while its endpoint value read 0.87 deg and passed.
    """

    HORIZON = 2.0
    #: Peak payload position discrepancy tolerated over a 1 s hover, in metres.
    #: Measured at 1.1e-3 m (``nQ = 3``) and 5.6e-4 m (``nQ = 4``); the bound
    #: is set a few times above so solver-version noise does not fail the
    #: build, and two orders below the ~0.2 m the slack asset produced.
    HOVER_POSITION_TOL = 4e-3
    #: Peak payload attitude discrepancy tolerated over the hover, in degrees.
    #: Measured at 0.02 deg (``nQ = 3``) and 0.01 deg (``nQ = 4``).
    HOVER_ATTITUDE_TOL = 0.2
    #: Peak position discrepancy tolerated over the :data:`HORIZON` step
    #: manoeuvre, in metres. Measured at 6.8e-3 m over a 0.58 m move — 1.2% of
    #: the distance travelled, and it is a *divergence* bound, so it is larger
    #: than the hover one by design rather than by slack.
    STEP_POSITION_TOL = 2e-2
    #: Peak payload attitude discrepancy tolerated over :data:`HORIZON`, in
    #: degrees. Measured at 0.76 deg.
    ATTITUDE_TOL = 2.0

    @staticmethod
    def _trace(model, target, start, tf):
        """Run the cascade and return ``(t, payload position, payload R)`` traces.

        Re-implements ``simulate``'s loop only to sample every step; the
        stepping and the input are the model's own.
        """
        model.payload_controller.setpoint = lambda t: (target, np.zeros(3), np.zeros(3))
        model.reset(payload_position=start)
        times = [model.t]
        positions = [model.state.payload_position.copy()]
        attitudes = [np.asarray(model.state.payload_orientation).copy()]
        while model.t < tf:
            model.step(model.compute_default_input())
            times.append(model.t)
            positions.append(model.state.payload_position.copy())
            attitudes.append(np.asarray(model.state.payload_orientation).copy())
        return np.array(times), np.array(positions), np.array(attitudes)

    @staticmethod
    def _matched_pair(nQ):
        """A MuJoCo model and an analytical one carrying the file's parameters."""
        mj = MultiQuadRigidbodyMujoco(render=False, num_quadrotors=nQ)
        base = MultiQuadRigidbodyBase(num_quadrotors=nQ)
        base.masses = mj.masses
        base.inertias = mj.inertias
        base.payload_mass = mj.payload_mass
        base.payload_inertia = mj.payload_inertia
        base.cable_lengths = mj.cable_lengths
        base.attachment_points = mj.attachment_points
        return mj, base

    @classmethod
    def _discrepancies(cls, nQ, start, target, tf):
        """Per-sample position (m) and attitude (deg) gaps over ``[0, tf]``.

        Both backends step at 1.25 ms under a 5 ms outer loop, so their sample
        times coincide — asserted here rather than assumed, because the whole
        point of comparing over the horizon is lost if the two traces are on
        different grids.
        """
        mj, base = cls._matched_pair(nQ)
        t_m, x_m, R_m = cls._trace(mj, target, start, tf)
        t_b, x_b, R_b = cls._trace(base, target, start, tf)
        assert len(t_m) == len(t_b)
        np.testing.assert_allclose(t_m, t_b, atol=1e-12)

        position = np.linalg.norm(x_m - x_b, axis=1)
        attitude = np.array([attitude_angle_deg(a, b) for a, b in zip(R_m, R_b, strict=True)])
        return mj, base, t_m, position, attitude

    @pytest.mark.parametrize("nQ", [3, 4])
    def test_backends_agree_on_a_short_hover(self, nQ):
        target = np.array([0.0, 0.0, 1.2])
        mj, base, _, position, attitude = self._discrepancies(nQ, target, target, 1.0)

        assert position.max() < self.HOVER_POSITION_TOL
        # Attitude too: a rigid-body payload can hold station while tumbling,
        # and holding station is all this test used to check.
        assert attitude.max() < self.HOVER_ATTITUDE_TOL
        # Both are actually holding the setpoint, so the agreement is not two
        # models failing in the same direction.
        assert float(np.linalg.norm(mj.state.payload_position - target)) < 0.05
        assert float(np.linalg.norm(base.state.payload_position - target)) < 0.05

    def test_backends_agree_on_a_step_manoeuvre(self):
        """A commanded move, so the comparison is of dynamics rather than a trim."""
        nQ = 4
        start = np.array([0.0, 0.0, 1.2])
        target = np.array([0.4, -0.3, 1.5])

        mj, _, times, position, attitude = self._discrepancies(nQ, start, target, self.HORIZON)

        assert position.max() < self.STEP_POSITION_TOL, (
            f"peak {position.max():.4g} m at t={times[position.argmax()]:.3f}"
        )
        assert attitude.max() < self.ATTITUDE_TOL, (
            f"peak {attitude.max():.4g} deg at t={times[attitude.argmax()]:.3f}"
        )

        # And the manoeuvre was real: without this the test would pass on two
        # models that both never left the start.
        travelled = float(np.linalg.norm(mj.state.payload_position - start))
        assert travelled > 0.2

    def test_the_endpoint_is_not_where_the_backends_disagree_most(self):
        """Why the assertions above run over the horizon rather than at ``tf``.

        The transient is what diverges; the endpoint is past it. Pinning this
        keeps the endpoint form of the test from creeping back as an
        "equivalent, cheaper" version of it.
        """
        start = np.array([0.0, 0.0, 1.2])
        target = np.array([0.4, -0.3, 1.5])
        _, _, times, position, attitude = self._discrepancies(4, start, target, self.HORIZON)

        assert position.max() > 1.5 * position[-1]
        assert attitude.max() > 1.5 * attitude[-1]
        assert times[position.argmax()] < 0.9 * self.HORIZON

    def test_the_two_backends_start_from_the_same_formation(self):
        """The precondition the agreement above rests on.

        Every cable spans its full length in both, so neither backend opens
        with a transient the other does not have.
        """
        nQ = 4
        mj = MultiQuadRigidbodyMujoco(render=False, num_quadrotors=nQ)
        base = MultiQuadRigidbodyBase(num_quadrotors=nQ)
        base.cable_lengths = mj.cable_lengths
        base.attachment_points = mj.attachment_points
        target = np.array([0.0, 0.0, 1.2])
        mj.reset(payload_position=target)
        base.reset(payload_position=target)

        for i in range(nQ):
            mj_span = np.linalg.norm(mj.attachment_position(i) - mj.state.positions[i])
            base_span = np.linalg.norm(base.attachment_position(i) - base.state.positions[i])
            assert mj_span == pytest.approx(mj.cable_lengths[i], abs=TOL_ANALYTIC)
            assert base_span == pytest.approx(base.cable_lengths[i], abs=TOL_ANALYTIC)
            np.testing.assert_allclose(
                mj.state.positions[i], base.state.positions[i], atol=TOL_ANALYTIC
            )
            np.testing.assert_allclose(
                np.asarray(mj.state.cable_attitudes[i]),
                np.asarray(base.state.cable_attitudes[i]),
                atol=TOL_ANALYTIC,
            )
