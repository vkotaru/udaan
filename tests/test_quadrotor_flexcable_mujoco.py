"""Tests for QuadrotorFlexCableMujoco — MJCF wiring, state sync, backend agreement.

The MJCF is generated from the construction parameters, so what is worth
testing here is the *correspondence* between the loaded file and the analytical
model: how many links there are, how long each is, what everything weighs, and
that the per-link state comes back **measured** rather than reconstructed.

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
one.  The agreement is centimetres over a 1 s free swing from a tilted chain
1 m long; the horizon is load-bearing there, as that test's docstring explains.

:class:`TestResetRepairsOffManifoldInput` is the other cross-backend contract:
both backends normalise a link direction and project a link rate onto its
tangent plane, so an off-manifold ``reset()`` has to land on the *same* state
in both.  It exists because a helper that only ever feeds already-valid input
(:func:`swinging_state`) cannot see a divergence in the repair.
"""

import logging
from pathlib import PurePath

import mujoco
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
#: The two are pinned together — see that method's docstring for why the bound
#: holds only on this horizon.
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
        """The airframe's tensor in its own body frame, read from the MJCF file.

        MuJoCo stores principal moments in the principal frame ``body_iquat``,
        and for this airframe that frame is rotated (the moments come out in
        the order ``J_z, J_y, J_x``), so the diagonal of ``body_inertia`` is
        not the body-frame tensor. The reference is computed from a fresh load
        of the file, not from the live model the setter writes into.
        """
        fresh = mujoco.MjModel.from_xml_path(mdl._mjMdl.full_path)
        body = mdl._quad_body_id
        flat = np.zeros(9)
        mujoco.mju_quat2Mat(flat, fresh.body_iquat[body])
        rotation = flat.reshape(3, 3)
        expected = rotation @ np.diag(fresh.body_inertia[body]) @ rotation.T
        np.testing.assert_allclose(mdl.inertia, expected, atol=TOL_TIGHT)
        # A flat airframe yaws hardest: J_z is its largest moment.
        assert mdl.inertia[2, 2] > mdl.inertia[0, 0]
        assert mdl.inertia[2, 2] > mdl.inertia[1, 1]

    def test_inertia_setter_writes_the_body_frame_tensor(self, mdl):
        """An assigned tensor is the one MuJoCo integrates, off-diagonals included."""
        tensor = np.array([[6e-3, 4e-4, -2e-4], [4e-4, 7e-3, 3e-4], [-2e-4, 3e-4, 1.1e-2]])
        mdl.inertia = tensor
        model = mdl._mjMdl.model
        body = mdl._quad_body_id
        flat = np.zeros(9)
        mujoco.mju_quat2Mat(flat, model.body_iquat[body])
        rotation = flat.reshape(3, 3)
        np.testing.assert_allclose(
            rotation @ np.diag(model.body_inertia[body]) @ rotation.T, tensor, atol=TOL_TIGHT
        )

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_a_body_moment_spins_the_airframe_up_at_the_model_inertia(self, mdl, axis):
        """From rest, one step of a body moment gives ``Ω ≈ J⁻¹ M dt``.

        The cable hangs from a ball joint at the airframe's centre of mass, so
        it pulls through that point and adds no moment. ``J`` is the model's
        own :attr:`inertia`, so a model that reads a different tensor from the
        one MuJoCo integrates predicts the wrong rate.
        """
        mdl.reset(position=np.array([0.0, 0.0, 2.0]))
        mdl._input_type = mdl.INPUT_TYPE.WRENCH
        moment = np.zeros(3)
        moment[axis] = 2e-3
        wrench = mdl.hover_input()
        wrench[1:] = moment
        mdl.step(wrench)
        predicted = np.linalg.solve(mdl.inertia, moment) * mdl.dt
        measured = np.asarray(mdl.state.angular_velocity)
        assert measured[axis] == pytest.approx(predicted[axis], rel=1e-2)

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

    @pytest.mark.parametrize("n", [1, 3])
    def test_state_after_a_step_is_the_current_configuration(self, n):
        """After ``step()`` every body is read at the configuration MuJoCo holds now.

        ``mj_step`` leaves ``xpos``/``xmat``/``cvel`` at the configuration the
        step started from, while the airframe is read from ``qpos``/``qvel``.
        The reference is a forward pass on a copy of the current data.
        """
        mdl = QuadrotorFlexCableMujoco(render=False, num_links=n)
        mdl.reset(**swinging_state(n))
        mdl._input_type = mdl.INPUT_TYPE.WRENCH
        for _ in range(20):
            mdl.step(mdl.hover_input())
        model = mdl._mjMdl.model
        fresh = mujoco.MjData(model)
        fresh.qpos[:] = mdl._mjMdl.data.qpos
        fresh.qvel[:] = mdl._mjMdl.data.qvel
        mujoco.mj_forward(model, fresh)
        np.testing.assert_allclose(
            mdl.state.payload_position, fresh.xpos[mdl._payload_body_id], atol=TOL_TIGHT
        )
        for k in range(n):
            q_k = -np.asarray(fresh.xmat[mdl._link_body_id[k]]).reshape(3, 3)[:, 2]
            np.testing.assert_allclose(np.asarray(mdl.state.link_attitudes[k]), q_k, atol=TOL_TIGHT)
        # The chain closes on the measured state after stepping, not only at reset.
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
        set up in world coordinates, and skipping that rotation would leave
        every link's ``omega_i`` off by an O(Omega) amount that no integration
        tolerance explains.
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
        together on purpose.  The peak gap grows close to linearly out to
        ~1.5 s and then takes off, which is why :data:`_AGREEMENT_HORIZON` is
        1 s: :data:`_AGREEMENT_BOUND` holds there with margin, so a slightly
        longer run does not break it, while a run to 2 s or beyond legitimately
        can — extending the horizon means re-deriving the bound, not nudging
        the constant.
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
    ``q_i`` and projects ``omega_i`` onto the tangent plane. The base model does
    the same, because storing the raw input there would put the two backends in
    *different states before a single step* — with these values, a gap in the
    payload position at ``t = 0`` comparable to the whole 1 s free-swing
    disagreement the suite tolerates.

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


class TestDivergence:
    """A diverged MuJoCo integration raises instead of flying on from MuJoCo's reset."""

    def test_step_raises_when_mujoco_diverges(self, mdl):
        mdl.reset(position=np.array([0.0, 0.0, 2.0]))
        mdl._input_type = mdl.INPUT_TYPE.WRENCH
        mdl.step(mdl.hover_input())
        mdl._mjMdl.data.qvel[0] = np.nan
        with pytest.raises(RuntimeError, match="diverged"):
            mdl.step(mdl.hover_input())

    def test_simulate_stops_at_the_divergence(self, mdl):
        """MuJoCo resets its clock on divergence; ``simulate()`` must not run on."""
        target = np.array([0.0, 0.0, 1.0])
        data = mdl._mjMdl.data
        injected = []

        def setpoint(t):
            if t >= 0.1 and not injected:
                injected.append(t)
                data.qvel[0] = np.nan
            return target, np.zeros(3), np.zeros(3)

        mdl.payload_controller.setpoint = setpoint
        with pytest.raises(RuntimeError, match="diverged"):
            mdl.simulate(tf=1.0, payload_position=target)
        assert mdl.t < 0.2


class TestStiffnessWarningScope:
    """The ZOH-Euler stiffness warning belongs to the analytical integrator only."""

    def test_the_mujoco_backend_does_not_warn(self, caplog):
        with caplog.at_level(logging.WARNING):
            mdl = QuadrotorFlexCableMujoco(render=False, num_links=5, cable_mass=0.01)
        assert mdl.stability_growth_rate() > 1.0
        assert "stiff for the ZOH-Euler integrator" not in caplog.text

    def test_the_analytical_model_still_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            mdl = QuadrotorFlexCableBase(num_links=5, cable_mass=0.01)
        assert mdl.stability_growth_rate() > 1.0
        assert "stiff for the ZOH-Euler integrator" in caplog.text


class TestDefaultAirframe:
    """The analytical model's default airframe is the one the generated MJCF builds."""

    @staticmethod
    def _mjcf_airframe(mj):
        """Mass and body-frame inertia ``R diag(I) Rᵀ`` read from the MJCF file itself."""
        model = mujoco.MjModel.from_xml_path(mj._mjMdl.full_path)
        body = mj._quad_body_id
        w, x, y, z = model.body_iquat[body]
        R = np.empty(9)
        mujoco.mju_quat2Mat(R, np.array([w, x, y, z]))
        R = R.reshape(3, 3)
        return float(model.body_mass[body]), R @ np.diag(model.body_inertia[body]) @ R.T

    @pytest.mark.parametrize("kwargs", [{}, {"mass": 0.75}])
    def test_analytical_default_is_the_mjcf_airframe(self, kwargs):
        mass, inertia = self._mjcf_airframe(QuadrotorFlexCableMujoco(render=False, **kwargs))
        base = QuadrotorFlexCableBase(**kwargs)
        assert base.mass == pytest.approx(mass, abs=1e-12)
        np.testing.assert_allclose(base.inertia, inertia, rtol=0.0, atol=1e-10)

    def test_controller_and_flat_map_defaults_are_the_model_airframe(self):
        from udaan.control.quadrotor_flexcable import QuadrotorFlexCableController
        from udaan.flatness import QuadrotorFlexCable

        base = QuadrotorFlexCableBase()
        ctrl = QuadrotorFlexCableController()
        flat = QuadrotorFlexCable()
        assert ctrl.mass == base.mass and flat.mass_q == base.mass
        np.testing.assert_array_equal(ctrl.inertia, base.inertia)
        np.testing.assert_array_equal(flat.inertia, base.inertia)


class TestClock:
    """The clock counts substeps, so ``k`` steps read ``k·dt`` on both backends."""

    @pytest.fixture(params=["base", "mujoco"])
    def model(self, request):
        if request.param == "base":
            return QuadrotorFlexCableBase()
        return QuadrotorFlexCableMujoco(render=False)

    def test_one_second_is_two_hundred_steps(self, model):
        steps = []
        step = model.step

        def counting_step(u, desired_att=None):
            steps.append(model.t)
            step(u, desired_att)

        model.step = counting_step
        model.simulate(tf=1.0)
        assert len(steps) == 200
        assert model.t == 1.0

    def test_clock_is_steps_times_dt_without_round_off(self, model):
        model.reset()
        u = model.hover_input()
        readings = {}
        for k in range(1, 2001):
            model.step(u)
            readings[k] = model.t
        # Exact at every whole second: an accumulated `t += h` drifts off them.
        assert [readings[k] for k in (200, 400, 1000, 2000)] == [1.0, 2.0, 5.0, 10.0]
        if isinstance(model, QuadrotorFlexCableMujoco):
            assert model._mjMdl.data.time == model.t
        model.reset()
        assert model.t == 0.0
        model.step(u)
        assert model.t == model.dt


class TestClockAnySubstepCount:
    """The clock is exact for substep counts that are not powers of two."""

    @pytest.mark.parametrize("substeps", [3, 5, 6])
    def test_simulate_takes_tf_over_dt_steps(self, substeps):
        model = QuadrotorFlexCableBase(inner_loop_steps=substeps)
        counted = []
        step = model.step

        def counting_step(u, desired_att=None):
            counted.append(model.t)
            step(u, desired_att)

        model.step = counting_step
        for k in (5, 7, 10, 20, 37):
            counted.clear()
            model.simulate(tf=k * 0.005)
            assert len(counted) == k
            assert model.t == pytest.approx(k * 0.005, rel=0, abs=1e-15)

    @pytest.mark.parametrize("bad", [0, -1, 4.0, True])
    def test_inner_loop_steps_must_be_a_positive_integer(self, bad):
        with pytest.raises(ValueError, match="inner_loop_steps"):
            QuadrotorFlexCableBase(inner_loop_steps=bad)


class TestInputPathsAgree:
    """The non-default input paths fly the machine they belong to, at one attitude rate."""

    def test_prop_forces_holds_the_airframe_at_its_setpoint(self):
        model = QuadrotorFlexCableBase(num_links=3, input="prop_forces")
        model.simulate(6.0, position=np.array([0.0, 0.0, 2.0]))
        target = np.asarray(model._prop_controller._pos_controller.setpoint(model.t)[0])
        assert np.linalg.norm(np.asarray(model.state.position) - target) < 0.02

    def test_prop_forces_controller_follows_the_mass_setter(self):
        model = QuadrotorFlexCableBase(num_links=3, input="prop_forces")
        model.mass = 1.3
        assert model._prop_controller._pos_controller.mass == pytest.approx(model.total_mass)
        model.inertia = np.array([0.01, 0.01, 0.02])
        np.testing.assert_allclose(model._prop_controller._att_controller.inertia, model.inertia)

    @pytest.mark.parametrize("backend", ["base", "mujoco"])
    def test_attitude_loop_runs_every_substep(self, backend):
        model = (
            QuadrotorFlexCableBase(input="acceleration")
            if backend == "base"
            else QuadrotorFlexCableMujoco(render=False, input="acceleration")
        )
        calls = []
        compute = model._att_controller.compute

        def counting(*args, **kwargs):
            calls.append(model.t)
            return compute(*args, **kwargs)

        model._att_controller.compute = counting
        model.reset()
        model.step(model.compute_station_keeping_input())
        assert len(calls) == 4

    def test_mujoco_refuses_an_inertia_it_would_overwrite(self):
        with pytest.raises(ValueError, match="inertia"):
            QuadrotorFlexCableMujoco(render=False, inertia=[1.0, 1.0, 1.0])
