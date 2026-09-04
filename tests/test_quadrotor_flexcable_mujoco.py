"""Tests for QuadrotorFlexCableMujoco — MJCF wiring, state sync, backend agreement.

The MJCF is generated from the construction parameters, so what is worth
testing here is the *correspondence* between the loaded file and the analytical
model: how many links there are, how long each is, what everything weighs, and
— the point of promoting M5 out of ``QuadrotorCsPayloadMujoco`` — that the
per-link state comes back **measured** rather than reconstructed.

:class:`TestPerLinkStateIsMeasured` is that contract.  Under
``cable_model="links"`` M2 loads the same kind of N-link chain and then
collapses it: one direction from the airframe to the payload, and a payload
velocity obtained by finite-differencing consecutive positions — which reads
exactly zero on the first query and is first order thereafter.  Here every
link's direction comes out of its body frame and every rate out of
``mj_objectVelocity``, so a ``reset()`` round-trips to machine precision.

:class:`TestBackendAgreement` compares against the analytical model.  It is a
*qualitative* agreement, not the near-exact one M4's tendons allow: MuJoCo's
links are cylinders with distributed mass and the analytical model lumps a
point mass at each joint, which is a physics difference and not an integration
one.  Measured over a 1 s free swing from a tilted chain the peak airframe gap
is 1.1e-2 - 1.7e-2 m at ``dt = 5 ms``, on a chain 1 m long; the horizon is
load-bearing there, and the gap-against-horizon table is in that test's
docstring.

:class:`TestResetRepairsOffManifoldInput` is the other cross-backend contract:
both backends normalise a link direction and project a link rate onto its
tangent plane, so an off-manifold ``reset()`` has to land on the *same* state
in both.  It exists because a helper that only ever feeds already-valid input
(:func:`swinging_state`) cannot see a divergence in the repair.
"""

from pathlib import PurePath

import numpy as np
import pytest

from tests.conftest import TOL_ANALYTIC, TOL_INTEGRATION, TOL_TIGHT
from udaan.core.defaults import DEFAULT_MJCF_TIMESTEP, GRAVITY
from udaan.manif import S2, SO3, TS2, TSO3, rodrigues_expm
from udaan.models.quadrotor_flexcable import (
    QuadrotorFlexCableBase,
    QuadrotorFlexCableMujoco,
)

pytestmark = pytest.mark.mujoco

LINK_COUNTS = [1, 2, 3, 5]

_E3 = np.array([0.0, 0.0, 1.0])

#: Horizon and bound of :meth:`TestBackendAgreement.test_free_swing_agrees`.
#: The two are pinned together — see that method's docstring for the measured
#: gap-against-horizon table the pair is derived from.
_AGREEMENT_HORIZON = 1.0  # s
_AGREEMENT_BOUND = 5e-2  # m


@pytest.fixture
def mdl():
    return QuadrotorFlexCableMujoco(render=False, num_links=3)


def swinging_state(n, seed=5, spread=0.15):
    """A tilted, swinging initial condition both backends can be reset to."""
    rng = np.random.default_rng(seed)
    directions = []
    for _ in range(n):
        v = np.array([0.0, 0.0, -1.0]) + rng.normal(scale=spread, size=3)
        directions.append(S2(v / np.linalg.norm(v)))
    return {
        "position": np.array([0.0, 0.0, 2.5]),
        "velocity": np.array([0.1, -0.05, 0.0]),
        "orientation": SO3(rodrigues_expm(np.array([0.05, -0.03, 0.1]))),
        "angular_velocity": TSO3(np.array([0.1, -0.05, 0.02])),
        "link_attitudes": directions,
        "link_angular_velocities": [
            TS2(np.cross(np.asarray(directions[i]), rng.normal(scale=0.3, size=3)))
            for i in range(n)
        ],
    }


class TestMujocoSetup:
    """Model loading, index resolution and parameters read back from the MJCF."""

    @pytest.mark.parametrize("n", LINK_COUNTS)
    def test_instantiate(self, n):
        mdl = QuadrotorFlexCableMujoco(render=False, num_links=n)
        assert mdl.num_links == n
        assert mdl._mjMdl.model.nu == 4
        assert len(mdl.state.link_attitudes) == n
        assert len(mdl.state.link_angular_velocities) == n

    @pytest.mark.parametrize("n", LINK_COUNTS)
    def test_link_lengths_come_from_the_file(self, n):
        """Read from the joint anchors, which carry full double precision.

        The cylinder geoms do not: ``MujocoAssetCreator.cylinder`` formats
        sizes with ``%g``, so a three-link cable's 1/3 m links would come back
        as 0.333334.  The spawn itself is unaffected — body and joint ``pos``
        are both written with ``str()`` — so what reading the geoms would break
        is the ``_GEOMETRY_TOL`` consistency check, which would reject a
        perfectly good model.
        """
        mdl = QuadrotorFlexCableMujoco(render=False, num_links=n)
        np.testing.assert_allclose(mdl.link_lengths, np.full(n, 1.0 / n), atol=TOL_TIGHT)
        assert mdl.cable_length == pytest.approx(1.0, abs=TOL_TIGHT)

    def test_masses_come_from_the_file(self, mdl):
        model = mdl._mjMdl.model
        quad_mass = float(model.body_mass[mdl._quad_body_id])
        assert mdl.mass == pytest.approx(quad_mass)
        # Every cylinder's mass lands on its own link; the payload sphere is
        # added to the last. Total mass is therefore exact.
        bodies = sum(
            float(model.body_mass[i])
            for i in (mdl._quad_body_id, *mdl._link_body_id, mdl._payload_body_id)
        )
        assert mdl.total_mass == pytest.approx(bodies, abs=TOL_TIGHT)

    def test_inertia_comes_from_the_file(self, mdl):
        diag = np.diag(mdl._mjMdl.model.body_inertia[mdl._quad_body_id])
        np.testing.assert_allclose(mdl.inertia, diag, atol=TOL_TIGHT)

    def test_timestep_divides_dt(self, mdl):
        assert mdl._mjDt == pytest.approx(DEFAULT_MJCF_TIMESTEP)
        assert mdl._step_iter * mdl._mjDt == pytest.approx(mdl.dt)

    def test_no_damping_anywhere(self, mdl):
        """The analytical model has no drag term, so neither may the asset."""
        np.testing.assert_allclose(mdl._mjMdl.model.dof_damping, 0.0, atol=TOL_TIGHT)
        np.testing.assert_allclose(mdl._mjMdl.model.jnt_stiffness, 0.0, atol=TOL_TIGHT)

    def test_spawn_hangs_straight_down(self, mdl):
        for i in range(mdl.num_links):
            np.testing.assert_allclose(
                np.asarray(mdl.state.link_attitudes[i]), -_E3, atol=TOL_ANALYTIC
            )
        assert np.linalg.norm(mdl.state.position - mdl.state.payload_position) == pytest.approx(
            mdl.cable_length, abs=TOL_ANALYTIC
        )
        assert mdl.state.position[2] == pytest.approx(mdl.spawn_height, abs=TOL_ANALYTIC)

    def test_state_types(self, mdl):
        assert isinstance(mdl.state.orientation, SO3)
        assert isinstance(mdl.state.angular_velocity, TSO3)
        for i in range(mdl.num_links):
            assert isinstance(mdl.state.link_attitudes[i], S2)
            assert isinstance(mdl.state.link_angular_velocities[i], TS2)

    def test_link_and_length_kwargs_are_rejected(self):
        for kwarg in ("link_masses", "link_lengths"):
            with pytest.raises(ValueError, match=kwarg):
                QuadrotorFlexCableMujoco(render=False, num_links=2, **{kwarg: [0.1, 0.1]})

    def test_the_asset_name_carries_every_shaping_parameter(self):
        """A changed parameter must not reuse a warm cache entry."""
        args = {
            "num_links": 2,
            "cable_length": 1.0,
            "cable_mass": 0.1,
            "payload_mass": 0.2,
            "quad_mass": 0.9,
            "spawn_height": 2.0,
        }
        baseline = QuadrotorFlexCableMujoco._ensure_xml(**args)
        assert QuadrotorFlexCableMujoco._ensure_xml(**args) == baseline
        assert PurePath(baseline).name.startswith("quad_flexcable_2l_")
        variants = {
            "num_links": 3,
            "cable_length": 0.5,
            "cable_mass": 0.2,
            "payload_mass": 0.4,
            "quad_mass": 1.2,
            "spawn_height": 3.0,
        }
        for key, value in variants.items():
            assert QuadrotorFlexCableMujoco._ensure_xml(**{**args, key: value}) != baseline

    def test_a_changed_parameter_reaches_the_model(self):
        first = QuadrotorFlexCableMujoco(render=False, num_links=2)
        second = QuadrotorFlexCableMujoco(render=False, num_links=2, payload_mass=0.35)
        assert second.link_masses[-1] > first.link_masses[-1]
        assert second.total_mass == pytest.approx(first.total_mass + 0.15, abs=TOL_TIGHT)


class TestPerLinkStateIsMeasured:
    """The reason M5 is its own system rather than an M2 backend option."""

    @pytest.mark.parametrize("n", LINK_COUNTS)
    def test_reset_round_trips_the_whole_chain(self, n):
        """Directions and rates come back exactly as they went in.

        Not approximately: ``q_i`` is the link body's own ``-e_3`` axis and
        ``omega_i`` is ``mj_objectVelocity``'s answer with the (unobservable)
        spin about ``q_i`` projected out, so the round-trip through the ball
        joints is algebra, not estimation.
        """
        mdl = QuadrotorFlexCableMujoco(render=False, num_links=n)
        kwargs = swinging_state(n)
        mdl.reset(**kwargs)
        for i in range(n):
            np.testing.assert_allclose(
                np.asarray(mdl.state.link_attitudes[i]),
                np.asarray(kwargs["link_attitudes"][i]),
                atol=TOL_ANALYTIC,
            )
            np.testing.assert_allclose(
                np.asarray(mdl.state.link_angular_velocities[i]),
                np.asarray(kwargs["link_angular_velocities"][i]),
                atol=TOL_ANALYTIC,
            )
        np.testing.assert_allclose(mdl.state.position, kwargs["position"], atol=TOL_ANALYTIC)
        np.testing.assert_allclose(mdl.state.velocity, kwargs["velocity"], atol=TOL_ANALYTIC)

    @pytest.mark.parametrize("n", LINK_COUNTS)
    def test_measured_state_agrees_with_the_chain_geometry(self, n):
        """``x_L = x_Q + sum l_i q_i`` holds on the *measured* state.

        MuJoCo computes the payload's position from the tree and this class
        reads the link directions from the bodies; nothing forces the two to
        agree, so their agreement is evidence that ``q_i`` really is link
        ``i``'s direction and the ordering runs airframe-to-payload.
        """
        mdl = QuadrotorFlexCableMujoco(render=False, num_links=n)
        mdl.reset(**swinging_state(n))
        offset = sum(
            mdl.link_lengths[i] * np.asarray(mdl.state.link_attitudes[i]) for i in range(n)
        )
        np.testing.assert_allclose(
            mdl.state.payload_position, mdl.state.position + offset, atol=TOL_ANALYTIC
        )
        rate = sum(mdl.link_lengths[i] * mdl.state.dq(i) for i in range(n))
        np.testing.assert_allclose(
            mdl.state.payload_velocity, mdl.state.velocity + rate, atol=TOL_ANALYTIC
        )

    def test_payload_velocity_is_not_finite_differenced(self, mdl):
        """The M2 links path returns exactly zero here; this one returns the truth."""
        mdl.reset(
            position=np.array([0.0, 0.0, 2.0]),
            velocity=np.array([0.3, -0.2, 0.1]),
        )
        # A straight chain translating rigidly: the payload moves with it.
        np.testing.assert_allclose(
            mdl.state.payload_velocity, np.array([0.3, -0.2, 0.1]), atol=TOL_ANALYTIC
        )

    def test_link_directions_stay_unit_norm(self, mdl):
        mdl.reset(**swinging_state(mdl.num_links))
        mdl._input_type = mdl.INPUT_TYPE.WRENCH
        for _ in range(int(1.0 / mdl.dt)):
            mdl.step(mdl.hover_input())
        for i in range(mdl.num_links):
            assert float(np.linalg.norm(np.asarray(mdl.state.link_attitudes[i]))) == pytest.approx(
                1.0, abs=TOL_ANALYTIC
            )

    def test_link_rates_stay_exactly_tangent(self, mdl):
        """MuJoCo has no ``q . omega`` to drift: the projection is applied on read.

        The analytical model integrates ``omega_i`` as a free 3-vector and lets
        tangency drift at O(h); here the state is *derived* from the body
        frames every query, so the invariant is exact by construction.
        """
        mdl.reset(**swinging_state(mdl.num_links))
        mdl._input_type = mdl.INPUT_TYPE.WRENCH
        for _ in range(int(1.0 / mdl.dt)):
            mdl.step(mdl.hover_input())
        for i in range(mdl.num_links):
            assert float(
                np.dot(
                    np.asarray(mdl.state.link_angular_velocities[i]),
                    np.asarray(mdl.state.link_attitudes[i]),
                )
            ) == pytest.approx(0.0, abs=TOL_ANALYTIC)


class TestReset:
    """What the MuJoCo reset can and cannot honour."""

    def test_payload_position_translates_the_whole_model(self, mdl):
        target = np.array([1.0, -2.0, 1.5])
        mdl.reset(payload_position=target)
        np.testing.assert_allclose(mdl.state.payload_position, target, atol=TOL_ANALYTIC)
        np.testing.assert_allclose(
            mdl.state.position, target + mdl.cable_length * _E3, atol=TOL_ANALYTIC
        )

    def test_payload_velocity_is_rejected(self, mdl):
        with pytest.raises(ValueError, match="payload_velocity"):
            mdl.reset(payload_velocity=np.zeros(3))

    def test_unknown_kwarg_is_rejected(self, mdl):
        with pytest.raises(ValueError, match="bogus"):
            mdl.reset(bogus=1.0)

    def test_reset_clears_time(self, mdl):
        mdl._input_type = mdl.INPUT_TYPE.WRENCH
        for _ in range(10):
            mdl.step(np.zeros(4))
        assert mdl.t > 0.0
        mdl.reset(position=np.array([0.0, 0.0, 2.0]))
        assert mdl.t == 0.0


class TestDynamics:
    """Trim and free fall, in MuJoCo."""

    @pytest.mark.parametrize("n", LINK_COUNTS)
    def test_hover_trim_holds(self, n):
        """The first-principles trim from the base model, fed open loop."""
        mdl = QuadrotorFlexCableMujoco(render=False, num_links=n)
        mdl._input_type = mdl.INPUT_TYPE.WRENCH
        mdl.reset(position=np.array([0.0, 0.0, 2.0]))
        start = mdl.state.payload_position.copy()
        wrench = mdl.hover_input()
        for _ in range(int(3.0 / mdl.dt)):
            mdl.step(wrench.copy())
        np.testing.assert_allclose(mdl.state.payload_position, start, atol=TOL_INTEGRATION)
        np.testing.assert_allclose(
            mdl.state.position, start + mdl.cable_length * _E3, atol=TOL_INTEGRATION
        )

    def test_zero_input_is_free_fall(self, mdl):
        """Nothing pulls, so the whole chain falls together at g.

        The velocity is exact; the position carries MuJoCo's semi-implicit
        half-step bias of ``g h t / 2`` — 1.8e-3 m here at ``h = 1.25 ms`` over
        0.3 s — which is integration, not dynamics.
        """
        mdl._input_type = mdl.INPUT_TYPE.WRENCH
        mdl.reset(position=np.array([0.0, 0.0, 5.0]))
        start = mdl.state.position.copy()
        offsets = mdl.link_positions() - start
        for _ in range(int(0.3 / mdl.dt)):
            mdl.step(np.zeros(4))
        np.testing.assert_allclose(mdl.state.velocity, -GRAVITY * mdl.t * _E3, atol=TOL_ANALYTIC)
        drop = 0.5 * GRAVITY * mdl.t**2
        bias = 0.5 * GRAVITY * mdl._mjDt * mdl.t
        np.testing.assert_allclose(mdl.state.position, start - drop * _E3, atol=1.5 * bias)
        # The chain falls rigidly: nothing stretches, nothing swings.
        np.testing.assert_allclose(
            mdl.link_positions() - mdl.state.position, offsets, atol=TOL_ANALYTIC
        )


class TestBackendAgreement:
    """The analytical model and MuJoCo, from one initial condition."""

    @staticmethod
    def _pair(n, **kwargs):
        backend = QuadrotorFlexCableMujoco(render=False, num_links=n, **kwargs)
        analytic = QuadrotorFlexCableBase(num_links=n, **kwargs)
        # MuJoCo is the authority on the inertial properties; adopt them so the
        # comparison is about the dynamics rather than about the parameters.
        analytic.mass = backend.mass
        analytic.inertia = backend.inertia
        analytic.link_masses = backend.link_masses
        analytic.link_lengths = backend.link_lengths
        backend._input_type = backend.INPUT_TYPE.WRENCH
        analytic._input_type = analytic.INPUT_TYPE.WRENCH
        return backend, analytic

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_initial_conditions_match_exactly(self, n):
        """Before any stepping, the two states must be the same state.

        This is what a backend comparison rests on, and it is not free: the
        airframe's body rate is stored in the *body* frame while the chain is
        set up in world coordinates, and skipping that rotation used to leave
        every link's ``omega_i`` off by an O(Omega) amount that no amount of
        integration tolerance would have explained.
        """
        backend, analytic = self._pair(n)
        kwargs = swinging_state(n)
        backend.reset(**kwargs)
        analytic.reset(**kwargs)
        np.testing.assert_allclose(
            backend.state.payload_position, analytic.state.payload_position, atol=TOL_ANALYTIC
        )
        np.testing.assert_allclose(
            backend.state.payload_velocity, analytic.state.payload_velocity, atol=TOL_ANALYTIC
        )
        for i in range(n):
            np.testing.assert_allclose(
                np.asarray(backend.state.link_angular_velocities[i]),
                np.asarray(analytic.state.link_angular_velocities[i]),
                atol=TOL_ANALYTIC,
            )

    @pytest.mark.slow
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_free_swing_agrees(self, n):
        """A free swing under the hover wrench, tracked in both backends.

        The tolerance absorbs a real physics difference (distributed cylinder
        mass against lumped joint masses) on top of the integration one, so it
        is centimetres rather than the sub-millimetre M4 manages.

        **The horizon is part of the claim.**  Two models this close but not
        identical separate at an accelerating rate, so no single bound survives
        an arbitrary extension of the loop and the constants are pinned
        together on purpose.  Peak gap (airframe / payload, in m, at
        ``dt = 5 ms``) against horizon:

        =======  ===============  ===============  ===============
        horizon  n = 1            n = 2            n = 3
        =======  ===============  ===============  ===============
        0.5 s    1.0e-2 / 1.7e-2  5.2e-3 / 1.3e-2  7.5e-3 / 1.2e-2
        1.0 s    1.7e-2 / 1.9e-2  1.1e-2 / 2.6e-2  1.6e-2 / 1.6e-2
        1.5 s    1.7e-2 / 2.0e-2  1.4e-2 / 2.8e-2  2.7e-2 / 2.3e-2
        2.0 s    3.3e-2 / 3.3e-2  2.4e-2 / 2.9e-2  5.8e-2 / 4.3e-2
        3.0 s    1.4e-1 / 1.2e-1  1.2e-1 / 1.1e-1  2.4e-1 / 2.1e-1
        =======  ===============  ===============  ===============

        Growth is close to linear out to ~1.5 s and then takes off, which is
        why :data:`_AGREEMENT_HORIZON` is 1 s: :data:`_AGREEMENT_BOUND` = 5e-2
        then holds with 1.9x margin on the asserted horizon and still holds
        (1.8x) at 1.5 s, so a slightly longer run does not break it.  A run to
        2 s or beyond does, and legitimately — extending the horizon means
        re-deriving the bound from this table, not nudging the constant.
        """
        backend, analytic = self._pair(n)
        kwargs = swinging_state(n)
        backend.reset(**kwargs)
        analytic.reset(**kwargs)
        wrench = analytic.hover_input()
        peak_quad = 0.0
        peak_payload = 0.0
        for _ in range(int(_AGREEMENT_HORIZON / analytic.dt)):
            backend.step(wrench.copy())
            analytic.step(wrench.copy())
            peak_quad = max(
                peak_quad, float(np.linalg.norm(backend.state.position - analytic.state.position))
            )
            peak_payload = max(
                peak_payload,
                float(
                    np.linalg.norm(backend.state.payload_position - analytic.state.payload_position)
                ),
            )
        assert peak_quad < _AGREEMENT_BOUND
        assert peak_payload < _AGREEMENT_BOUND


#: The off-manifold ``reset()`` input both backends have to repair identically:
#: a direction of norm 1.0488 and a rate with ``q . omega = -0.23``.
_BAD_DIRECTION = np.array([0.3, 0.1, -1.0])
_BAD_RATE = np.array([0.2, 0.1, 0.3])


class TestResetRepairsOffManifoldInput:
    """Both backends repair an off-manifold ``reset()`` the same way.

    MuJoCo has no choice: a ball joint cannot represent a stretched link, and a
    spin about the link axis is unobservable, so its ``reset()`` normalises
    ``q_i`` and projects ``omega_i`` onto the tangent plane. The base model was
    made to do the same, because storing the raw input there instead put the
    two backends in *different states before a single step* — with these
    values, a 4.9e-2 m gap in the payload position at ``t = 0``, larger than
    the whole 1 s free-swing disagreement the suite tolerates.

    :func:`swinging_state` cannot catch that: it builds ``omega_i`` as
    ``q_i x r`` from an already normalised ``q_i``, so everything it feeds is
    on the manifold already and both repairs are no-ops.
    """

    @staticmethod
    def _kwargs(n):
        return {
            "position": np.array([0.0, 0.0, 2.5]),
            "link_attitudes": [_BAD_DIRECTION.copy() for _ in range(n)],
            "link_angular_velocities": [_BAD_RATE.copy() for _ in range(n)],
        }

    @pytest.mark.parametrize("n", [1, 3])
    def test_both_backends_land_on_the_same_state(self, n):
        backend = QuadrotorFlexCableMujoco(render=False, num_links=n)
        analytic = QuadrotorFlexCableBase(num_links=n)
        analytic.mass = backend.mass
        analytic.inertia = backend.inertia
        analytic.link_masses = backend.link_masses
        analytic.link_lengths = backend.link_lengths

        kwargs = self._kwargs(n)
        backend.reset(**kwargs)
        analytic.reset(**kwargs)

        np.testing.assert_allclose(
            backend.state.position, analytic.state.position, atol=TOL_ANALYTIC
        )
        np.testing.assert_allclose(
            backend.state.payload_position, analytic.state.payload_position, atol=TOL_ANALYTIC
        )
        for i in range(n):
            np.testing.assert_allclose(
                np.asarray(backend.state.link_attitudes[i]),
                np.asarray(analytic.state.link_attitudes[i]),
                atol=TOL_ANALYTIC,
            )
            np.testing.assert_allclose(
                np.asarray(backend.state.link_angular_velocities[i]),
                np.asarray(analytic.state.link_angular_velocities[i]),
                atol=TOL_ANALYTIC,
            )

    @pytest.mark.parametrize("n", [1, 3])
    def test_both_backends_land_on_the_manifold(self, n):
        """Whatever went in, ``|q_i| = 1`` and ``q_i . omega_i = 0`` come out."""
        models = [
            QuadrotorFlexCableMujoco(render=False, num_links=n),
            QuadrotorFlexCableBase(num_links=n),
        ]
        # The input is genuinely off-manifold, or the test proves nothing.
        assert abs(float(np.linalg.norm(_BAD_DIRECTION)) - 1.0) > 1e-2
        assert abs(float(_BAD_DIRECTION @ _BAD_RATE)) > 1e-2

        for mdl in models:
            mdl.reset(**self._kwargs(n))
            for i in range(n):
                q_i = np.asarray(mdl.state.link_attitudes[i])
                omega_i = np.asarray(mdl.state.link_angular_velocities[i])
                assert float(np.linalg.norm(q_i)) == pytest.approx(1.0, abs=TOL_TIGHT)
                assert float(q_i @ omega_i) == pytest.approx(0.0, abs=TOL_TIGHT)
                np.testing.assert_allclose(
                    q_i, _BAD_DIRECTION / np.linalg.norm(_BAD_DIRECTION), atol=TOL_ANALYTIC
                )
