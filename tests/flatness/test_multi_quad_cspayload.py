"""Unit tests for :class:`udaan.flatness.MultiQuadCsPayload` — the
``nQ``-quadrotor + cable-suspended-point-mass-payload flatness map, and for the
tension-split policies that resolve its ``3(nQ-1)``-dimensional free choice."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import TOL_ANALYTIC, TOL_EXACT, TOL_TIGHT
from udaan.core.defaults import (
    DEFAULT_CABLE_LENGTH,
    DEFAULT_FORMATION_RADIUS,
    DEFAULT_PAYLOAD_MASS,
    DEFAULT_QUAD_INERTIA,
    DEFAULT_QUAD_MASS,
    GRAVITY,
)
from udaan.flatness import (
    DEFAULT_MIN_TENSION,
    AffineTensionSplit,
    Jet,
    LegacyFixedTensionSplit,
    MultiQuadCsPayload,
    MultiQuadCsPayloadFlats,
    QuadrotorCsPayload,
    QuadrotorCsPayloadFlats,
    RingTensionSplit,
)
from udaan.models.multi_quad_cspayload.base import (
    MultiQuadCsPayloadBase,
    formation_cable_attitudes,
)

_E3 = np.array([0.0, 0.0, 1.0])


# ─── Helpers ──────────────────────────────────────────────────────────


def _hover_flats(num_quadrotors: int) -> MultiQuadCsPayloadFlats:
    """Hover flats: payload position and every yaw jet identically zero."""
    return MultiQuadCsPayloadFlats(
        x_L=Jet.zeros(order=6, dim=3),
        psi=[Jet.zeros(order=2, dim=1) for _ in range(num_quadrotors)],
    )


def _circle_flats(
    t: float,
    num_quadrotors: int,
    omega: float = 0.5,
    radius: float = 0.4,
    height: float = 1.0,
) -> MultiQuadCsPayloadFlats:
    """Analytic flats for a horizontal payload circle at constant altitude.

    Every quadrotor's yaw tracks the heading angle. The radius is small enough
    that all cables stay comfortably taut.
    """
    ct, st = np.cos(omega * t), np.sin(omega * t)
    rows = [
        np.array([radius * ct, radius * st, height]),
        np.array([-radius * omega * st, radius * omega * ct, 0.0]),
        np.array([-radius * omega**2 * ct, -radius * omega**2 * st, 0.0]),
        np.array([radius * omega**3 * st, -radius * omega**3 * ct, 0.0]),
        np.array([radius * omega**4 * ct, radius * omega**4 * st, 0.0]),
        np.array([-radius * omega**5 * st, radius * omega**5 * ct, 0.0]),
        np.array([-radius * omega**6 * ct, -radius * omega**6 * st, 0.0]),
    ]
    psi_jet = Jet(np.array([omega * t, omega, 0.0]))
    return MultiQuadCsPayloadFlats(
        x_L=Jet.from_list(rows), psi=[psi_jet for _ in range(num_quadrotors)]
    )


def _net_force_jet(x_L_jet: Jet, payload_mass: float, gravity: float = GRAVITY) -> list:
    """``[F_net, …, F_net^(4)]`` — the quantity a policy splits."""
    net = [-payload_mass * (x_L_jet[2] + gravity * _E3)]
    net.extend(-payload_mass * x_L_jet[k] for k in range(3, 7))
    return net


# ─── Construction ─────────────────────────────────────────────────────


class TestConstruction:
    def test_defaults_broadcast_scalars(self):
        f2s = MultiQuadCsPayload(num_quadrotors=4)
        assert f2s.num_quadrotors == 4
        np.testing.assert_allclose(f2s.masses, np.full(4, DEFAULT_QUAD_MASS))
        np.testing.assert_allclose(f2s.cable_lengths, np.full(4, DEFAULT_CABLE_LENGTH))
        assert f2s.inertias.shape == (4, 3, 3)
        np.testing.assert_allclose(f2s.inertias[2], DEFAULT_QUAD_INERTIA)

    def test_per_index_parameters_accepted(self):
        f2s = MultiQuadCsPayload(
            num_quadrotors=3, masses=[0.8, 0.9, 1.0], cable_lengths=[1.0, 1.2, 0.9]
        )
        np.testing.assert_allclose(f2s.masses, [0.8, 0.9, 1.0])
        np.testing.assert_allclose(f2s.cable_lengths, [1.0, 1.2, 0.9])

    def test_from_model_reads_model_params(self):
        """Every parameter comes off the model — at *non-default* values.

        The radius in particular has to be non-default: with the model built at
        ``DEFAULT_FORMATION_RADIUS`` the assertion compares the default to
        itself, and a ``from_model`` that hard-coded the default (discarding the
        model's radius) would pass. That would unpin the property the whole
        classmethod exists for — that the flat hover reference *is* the model's
        ``reset()`` configuration — at every radius but one.
        """
        radius = 0.37
        assert radius != pytest.approx(DEFAULT_FORMATION_RADIUS)
        model = MultiQuadCsPayloadBase(num_quadrotors=4, formation_radius=radius)
        model.masses = [0.8, 0.9, 1.0, 1.1]
        model.cable_lengths = [1.0, 1.1, 1.2, 1.3]
        f2s = MultiQuadCsPayload.from_model(model)
        assert f2s.num_quadrotors == 4
        np.testing.assert_allclose(f2s.masses, model.masses)
        np.testing.assert_allclose(f2s.cable_lengths, model.cable_lengths)
        np.testing.assert_allclose(f2s.inertias, model.inertias)
        assert f2s.payload_mass == pytest.approx(model.payload_mass)
        assert f2s.formation_radius == pytest.approx(radius)
        assert f2s.tension_policy.radius == pytest.approx(radius)

        # …and the radius actually reaches the geometry: the flat hover
        # reference must reproduce the model's reset formation at this radius.
        model.reset()
        ref, _ = f2s.recover(_hover_flats(4))
        for i in range(4):
            np.testing.assert_allclose(
                f2s.tension_policy.hover_directions[i],
                np.asarray(model.state.cable_attitudes[i]),
                atol=TOL_TIGHT,
            )
            offset = ref.positions[i] - ref.payload_position
            assert float(np.linalg.norm(offset[:2])) == pytest.approx(radius, abs=TOL_ANALYTIC)

    def test_min_tension_default_is_re_exported(self):
        # The knob is part of the public constructor signature, so the constant
        # naming its default has to be reachable from `udaan.flatness`.
        import udaan.flatness as flatness

        assert "DEFAULT_MIN_TENSION" in flatness.__all__
        assert MultiQuadCsPayload(num_quadrotors=2).min_tension == pytest.approx(
            DEFAULT_MIN_TENSION
        )

    def test_class_attribute_triple_declared(self):
        assert MultiQuadCsPayload.Flats is MultiQuadCsPayloadFlats
        assert MultiQuadCsPayload.RefState.__name__ == "MultiQuadCsPayloadRefState"
        assert MultiQuadCsPayload.Inputs.__name__ == "MultiQuadCsPayloadInputs"

    def test_num_quadrotors_validated(self):
        with pytest.raises(ValueError, match="num_quadrotors must be >= 1"):
            MultiQuadCsPayload(num_quadrotors=0)

    def test_parameter_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="masses must have 3 entries"):
            MultiQuadCsPayload(num_quadrotors=3, masses=[1.0, 1.0])

    def test_negative_parameters_rejected(self):
        with pytest.raises(ValueError, match="masses must be positive"):
            MultiQuadCsPayload(num_quadrotors=2, masses=[1.0, -1.0])
        with pytest.raises(ValueError, match="cable_lengths must be positive"):
            MultiQuadCsPayload(num_quadrotors=2, cable_lengths=[1.0, 0.0])
        with pytest.raises(ValueError, match="payload_mass must be positive"):
            MultiQuadCsPayload(num_quadrotors=2, payload_mass=0.0)

    def test_inertia_shape_validated(self):
        with pytest.raises(ValueError, match=r"inertias must be \(3,\)"):
            MultiQuadCsPayload(num_quadrotors=2, inertias=np.zeros((2, 2)))

    def test_policy_size_must_match(self):
        with pytest.raises(ValueError, match="tension_policy is built for 3"):
            MultiQuadCsPayload(num_quadrotors=4, tension_policy=LegacyFixedTensionSplit())


# ─── Flats validation (derivative-order accounting) ───────────────────


class TestFlatsValidation:
    def test_valid_order_flats_construct(self):
        flats = _hover_flats(3)
        assert flats.num_quadrotors == 3

    def test_higher_order_flats_also_accepted(self):
        """Orders above the minimum are kept as given, not truncated, and the
        extra derivatives are simply unused by :meth:`recover`."""
        flats = MultiQuadCsPayloadFlats(
            x_L=Jet.zeros(order=8, dim=3),
            psi=[Jet.zeros(order=4, dim=1) for _ in range(2)],
        )
        assert flats.x_L.order == 8
        assert flats.num_quadrotors == 2
        assert all(p.order == 4 for p in flats.psi)
        # An over-specified jet still recovers, and to the hover reference.
        _, inputs = MultiQuadCsPayload(num_quadrotors=2).recover(flats)
        assert len(inputs.thrusts) == 2

    def test_x_wrong_dim_rejected(self):
        with pytest.raises(ValueError, match="x_L must be a 3-D jet"):
            MultiQuadCsPayloadFlats(x_L=Jet.zeros(order=6, dim=2), psi=[Jet.zeros(2, 1)])

    def test_x_insufficient_order_rejected(self):
        # q^(4) — and therefore M — consumes x_L^(6); order 5 is not enough.
        with pytest.raises(ValueError, match=r"x_L.order must be >= 6"):
            MultiQuadCsPayloadFlats(x_L=Jet.zeros(order=5, dim=3), psi=[Jet.zeros(2, 1)])

    def test_psi_insufficient_order_rejected(self):
        with pytest.raises(ValueError, match=r"psi\[1\].order must be >= 2"):
            MultiQuadCsPayloadFlats(
                x_L=Jet.zeros(order=6, dim=3),
                psi=[Jet.zeros(2, 1), Jet.zeros(1, 1)],
            )

    def test_psi_not_scalar_rejected(self):
        with pytest.raises(ValueError, match=r"psi\[0\] must be a scalar jet"):
            MultiQuadCsPayloadFlats(x_L=Jet.zeros(order=6, dim=3), psi=[Jet.zeros(2, 3)])

    def test_bare_jet_psi_rejected(self):
        with pytest.raises(ValueError, match="psi must be a list of scalar jets"):
            MultiQuadCsPayloadFlats(x_L=Jet.zeros(order=6, dim=3), psi=Jet.zeros(2, 1))

    def test_empty_psi_rejected(self):
        with pytest.raises(ValueError, match="at least one yaw jet"):
            MultiQuadCsPayloadFlats(x_L=Jet.zeros(order=6, dim=3), psi=[])

    def test_yaw_count_must_match_map(self):
        f2s = MultiQuadCsPayload(num_quadrotors=3)
        with pytest.raises(ValueError, match="flats carry 2 yaw jets"):
            f2s.recover(_hover_flats(2))


# ─── Tension-split policies ───────────────────────────────────────────


class TestAffineTensionSplit:
    def test_split_sums_to_net_force_at_every_order(self):
        policy = RingTensionSplit(
            num_quadrotors=4,
            payload_mass=DEFAULT_PAYLOAD_MASS,
            masses=[0.8, 0.9, 1.0, 1.1],
            cable_lengths=[1.0, 1.1, 1.2, 1.3],
        )
        rng = np.random.default_rng(7)
        net = [rng.normal(size=3) for _ in range(5)]
        net[0] = np.array([0.3, -0.2, -2.0])
        forces = policy.split(net)
        for k in range(5):
            total = sum(f[k] for f in forces)
            np.testing.assert_allclose(total, net[k], atol=TOL_EXACT)

    def test_derivatives_are_the_share_times_the_net_derivative(self):
        # 𝐓̄_i and F̄_net are constants ⇒ 𝐓_i^(k) = α_i F_net^(k) exactly, k ≥ 1.
        policy = RingTensionSplit(3, DEFAULT_PAYLOAD_MASS, [1.0, 2.0, 3.0], [1.0, 1.0, 1.0])
        rng = np.random.default_rng(11)
        net = [rng.normal(size=3) for _ in range(5)]
        forces = policy.split(net)
        for i in range(3):
            for k in range(1, 5):
                np.testing.assert_allclose(forces[i][k], policy.shares[i] * net[k], atol=TOL_EXACT)

    def test_shares_must_sum_to_one(self):
        with pytest.raises(ValueError, match="shares must sum to 1"):
            AffineTensionSplit(anchors=np.tile(-_E3, (2, 1)), shares=[0.4, 0.4])

    def test_zero_anchor_rejected(self):
        with pytest.raises(ValueError, match="non-zero cable force"):
            AffineTensionSplit(anchors=np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 0.0]]), shares=[1, 0])

    def test_anchor_shape_validated(self):
        with pytest.raises(ValueError, match=r"anchors must have shape \(nQ, 3\)"):
            AffineTensionSplit(anchors=np.zeros(3), shares=[1.0])


class TestRingTensionSplit:
    @pytest.mark.parametrize("nQ", [1, 2, 3, 4, 6])
    def test_anchors_balance_payload_weight(self, nQ):
        policy = RingTensionSplit(
            nQ,
            DEFAULT_PAYLOAD_MASS,
            np.full(nQ, DEFAULT_QUAD_MASS),
            np.full(nQ, DEFAULT_CABLE_LENGTH),
        )
        expected = -DEFAULT_PAYLOAD_MASS * GRAVITY * _E3
        np.testing.assert_allclose(policy.reference_net_force, expected, atol=TOL_TIGHT)

    @pytest.mark.parametrize("nQ", [1, 2, 3, 5])
    def test_hover_directions_match_model_ring_formation(self, nQ):
        """The flat map's hover geometry must be the model's reset formation.

        ``RingTensionSplit`` re-derives the ring inside ``udaan.flatness`` (the
        layering forbids importing from ``udaan.models``); this pins the two
        derivations together.
        """
        lengths = np.linspace(1.0, 1.4, nQ)
        policy = RingTensionSplit(nQ, DEFAULT_PAYLOAD_MASS, np.full(nQ, DEFAULT_QUAD_MASS), lengths)
        model_q = formation_cable_attitudes(nQ, lengths, DEFAULT_FORMATION_RADIUS)
        for i in range(nQ):
            np.testing.assert_allclose(
                policy.hover_directions[i], np.asarray(model_q[i]), atol=TOL_TIGHT
            )

    def test_shares_are_mass_weighted(self):
        masses = np.array([0.8, 1.2, 2.0])
        policy = RingTensionSplit(3, DEFAULT_PAYLOAD_MASS, masses, np.ones(3))
        np.testing.assert_allclose(policy.shares, masses / masses.sum(), atol=TOL_EXACT)

    def test_equal_quadrotors_give_symmetric_split(self):
        policy = RingTensionSplit(5, DEFAULT_PAYLOAD_MASS, np.ones(5), np.ones(5))
        np.testing.assert_allclose(policy.shares, np.full(5, 0.2), atol=TOL_EXACT)
        norms = np.linalg.norm(policy.anchors, axis=1)
        np.testing.assert_allclose(norms, norms[0], atol=TOL_TIGHT)

    def test_tension_magnitudes_scale_with_cable_length(self):
        lengths = np.array([1.0, 1.5, 2.0])
        policy = RingTensionSplit(3, DEFAULT_PAYLOAD_MASS, np.ones(3), lengths)
        norms = np.linalg.norm(policy.anchors, axis=1)
        np.testing.assert_allclose(norms / lengths, norms[0] / lengths[0], atol=TOL_TIGHT)

    def test_degenerate_zero_radius_rejected(self):
        # radius 0 ⇒ every cable vertical ⇒ the quadrotors stack up.
        with pytest.raises(ValueError, match="strictly positive"):
            RingTensionSplit(3, DEFAULT_PAYLOAD_MASS, np.ones(3), np.ones(3), radius=0.0)

    def test_radius_must_fit_inside_every_cable(self):
        with pytest.raises(ValueError, match="must be strictly below every cable length"):
            RingTensionSplit(3, DEFAULT_PAYLOAD_MASS, np.ones(3), [1.0, 0.4, 1.0], radius=0.5)

    def test_single_quadrotor_ignores_radius(self):
        """There is no ring at ``nQ = 1``: the radius is silently dropped, and
        with it the ``0 < r < min l_i`` validation that applies for ``nQ ≥ 2``."""
        policy = RingTensionSplit(1, DEFAULT_PAYLOAD_MASS, [1.0], [1.0], radius=0.5)
        np.testing.assert_allclose(policy.hover_directions[0], -_E3, atol=TOL_EXACT)
        np.testing.assert_allclose(policy.shares, [1.0], atol=TOL_EXACT)
        assert policy.radius == 0.0
        # Radii that are rejected outright at nQ >= 2 pass unnoticed here.
        for ignored in (0.0, 2.0):
            other = RingTensionSplit(1, DEFAULT_PAYLOAD_MASS, [1.0], [1.0], radius=ignored)
            np.testing.assert_allclose(other.anchors, policy.anchors, atol=TOL_EXACT)


class TestLegacyFixedTensionSplit:
    """The published ``nQ = 3`` split, kept selectable for reproducibility."""

    def test_anchors_balance_payload_weight(self):
        policy = LegacyFixedTensionSplit(DEFAULT_PAYLOAD_MASS)
        expected = -DEFAULT_PAYLOAD_MASS * GRAVITY * _E3
        np.testing.assert_allclose(policy.reference_net_force, expected, atol=TOL_TIGHT)

    def test_all_dynamics_land_on_quadrotor_one(self):
        policy = LegacyFixedTensionSplit(DEFAULT_PAYLOAD_MASS)
        np.testing.assert_allclose(policy.shares, [1.0, 0.0, 0.0], atol=TOL_EXACT)
        net = [np.array([0.0, 0.0, -2.0])] + [np.full(3, 1.0) for _ in range(4)]
        forces = policy.split(net)
        for i in (1, 2):
            for k in range(1, 5):
                np.testing.assert_allclose(forces[i][k], np.zeros(3), atol=TOL_EXACT)

    def test_matches_matlab_reference_magnitudes(self):
        """Cables 2, 3 hold ``m_L g / 3`` each, tilted 30° about the x-axis.

        The tilt has to be asserted *as an angle*. Norms, the ±symmetry in
        ``y`` and the ``x``-planarity are all invariant to it — ``RotX``
        preserves the norm, the two signed ``y`` components cancel, and
        ``RotX e₃`` has no ``x`` at any angle — so a wrong tilt would slip
        through them all. The round-trip tests are structurally blind to it
        too: any anchor set summing to ``-m_L g e₃`` round-trips exactly.
        Reproducing the published ``π/6`` is the entire reason this policy
        exists.
        """
        policy = LegacyFixedTensionSplit(DEFAULT_PAYLOAD_MASS)
        norms = np.linalg.norm(policy.anchors, axis=1)
        assert norms[1] == pytest.approx(DEFAULT_PAYLOAD_MASS * GRAVITY / 3.0, abs=TOL_TIGHT)
        assert norms[2] == pytest.approx(DEFAULT_PAYLOAD_MASS * GRAVITY / 3.0, abs=TOL_TIGHT)
        np.testing.assert_allclose(policy.hover_directions[0], -_E3, atol=TOL_TIGHT)
        # Its formation is planar: no quadrotor has any x-extent.
        np.testing.assert_allclose(policy.anchors[:, 0], np.zeros(3), atol=TOL_EXACT)

        # The MATLAB tilt: RotX(∓π/6) e₃, negated into this module's
        # force-on-the-quadrotor convention, so q̄_{2,3} = -[0, ∓sin φ, cos φ].
        tilt = np.pi / 6.0
        np.testing.assert_allclose(
            policy.hover_directions[1], [0.0, -np.sin(tilt), -np.cos(tilt)], atol=TOL_TIGHT
        )
        np.testing.assert_allclose(
            policy.hover_directions[2], [0.0, np.sin(tilt), -np.cos(tilt)], atol=TOL_TIGHT
        )
        for i in (1, 2):
            angle = np.arccos(np.clip(float(np.dot(policy.hover_directions[i], -_E3)), -1.0, 1.0))
            assert angle == pytest.approx(tilt, abs=TOL_TIGHT)
        # …and cable 1 is the untilted one, so the three are not all equal.
        assert np.arccos(
            np.clip(float(np.dot(policy.hover_directions[0], -_E3)), -1.0, 1.0)
        ) == pytest.approx(0.0, abs=TOL_TIGHT)

    def test_margin_is_worse_than_the_ring_default(self):
        legacy = LegacyFixedTensionSplit(DEFAULT_PAYLOAD_MASS)
        ring = RingTensionSplit(3, DEFAULT_PAYLOAD_MASS, np.full(3, DEFAULT_QUAD_MASS), np.ones(3))
        assert legacy.net_force_margin() < ring.net_force_margin()


# ─── Hover recovery ───────────────────────────────────────────────────


class TestHover:
    @pytest.mark.parametrize("nQ", [1, 2, 3, 4])
    def test_thrusts_and_tensions_carry_the_whole_system(self, nQ):
        f2s = MultiQuadCsPayload(num_quadrotors=nQ)
        ref, inputs = f2s.recover(_hover_flats(nQ))

        # Total vertical thrust = total weight; horizontal components cancel.
        total = np.zeros(3)
        for i in range(nQ):
            total = total + inputs.thrusts[i] * (np.asarray(ref.orientations[i]) @ _E3)
        expected = (nQ * DEFAULT_QUAD_MASS + DEFAULT_PAYLOAD_MASS) * GRAVITY * _E3
        np.testing.assert_allclose(total, expected, atol=TOL_ANALYTIC)

    @pytest.mark.parametrize("nQ", [1, 2, 3, 4])
    def test_cable_forces_sum_to_payload_weight(self, nQ):
        f2s = MultiQuadCsPayload(num_quadrotors=nQ)
        ref, inputs = f2s.recover(_hover_flats(nQ))
        total = np.zeros(3)
        for i in range(nQ):
            total = total + inputs.tensions[i] * np.asarray(ref.cable_attitudes[i])
        np.testing.assert_allclose(total, -DEFAULT_PAYLOAD_MASS * GRAVITY * _E3, atol=TOL_ANALYTIC)

    @pytest.mark.parametrize("nQ", [1, 3, 4])
    def test_hover_moments_and_rates_vanish(self, nQ):
        f2s = MultiQuadCsPayload(num_quadrotors=nQ)
        ref, inputs = f2s.recover(_hover_flats(nQ))
        for i in range(nQ):
            np.testing.assert_allclose(inputs.moments[i], np.zeros(3), atol=TOL_ANALYTIC)
            np.testing.assert_allclose(
                ref.cable_angular_velocities[i], np.zeros(3), atol=TOL_ANALYTIC
            )
            np.testing.assert_allclose(
                np.asarray(ref.angular_velocities[i]), np.zeros(3), atol=TOL_ANALYTIC
            )

    def test_hover_positions_sit_on_the_formation_ring(self):
        nQ = 4
        f2s = MultiQuadCsPayload(num_quadrotors=nQ)
        ref, _ = f2s.recover(_hover_flats(nQ))
        for i in range(nQ):
            offset = ref.positions[i] - ref.payload_position
            assert float(np.linalg.norm(offset[:2])) == pytest.approx(
                DEFAULT_FORMATION_RADIUS, abs=TOL_ANALYTIC
            )
            assert offset[2] > 0.0

    def test_as_state_projects_onto_the_model_state(self):
        nQ = 3
        f2s = MultiQuadCsPayload(num_quadrotors=nQ)
        ref, _ = f2s.recover(_hover_flats(nQ))
        state = ref.as_state()
        assert state.num_quadrotors == nQ
        assert len(state.cable_attitudes) == nQ
        for i in range(nQ):
            np.testing.assert_allclose(
                np.asarray(state.cable_attitudes[i]), ref.cable_attitudes[i], atol=TOL_EXACT
            )


# ─── nQ = 1 reduction to the single-quadrotor map ─────────────────────


class TestSingleQuadrotorReduction:
    """At ``nQ = 1`` the map must agree with
    :class:`~udaan.flatness.QuadrotorCsPayload` to the last bit.

    What this establishes is *plumbing*, not an independent cross-check of the
    algebra. The two maps are not independent derivations: they share
    ``tension_direction_jet``, ``normalize_derivatives`` and
    ``_attitude_from_thrust_vector`` verbatim, and at ``nQ = 1`` the split
    degenerates to the identity ``𝐓̄ + 1.0·(F_net − 𝐓̄)`` — bit-exactness here
    is Sterbenz's lemma, not a numerical coincidence worth defending. In
    particular it exercises **none** of the ``nQ > 1`` split machinery: no
    non-trivial share, no internal force, no ring. ``RingTensionSplit`` also
    special-cases ``nQ = 1`` to a single vertical offset, so
    ``formation_radius`` is silently ignored in this configuration.

    It is still worth keeping as a regression on the wiring: the per-quadrotor
    loop, the jet plumbing and the sign conventions all have to line up with
    the single-quadrotor map for it to hold at all.
    """

    @pytest.fixture
    def flats_pair(self):
        rng = np.random.default_rng(3)
        x_data = rng.normal(size=(7, 3)) * 0.3
        psi_data = np.array([0.4, 0.25, -0.15])
        multi = MultiQuadCsPayloadFlats(x_L=Jet(x_data), psi=[Jet(psi_data)])
        single = QuadrotorCsPayloadFlats(x_L=Jet(x_data), psi=Jet(psi_data))
        return multi, single

    def test_recovery_matches_m2_bit_for_bit(self, flats_pair):
        multi_flats, single_flats = flats_pair
        multi = MultiQuadCsPayload(
            num_quadrotors=1,
            masses=DEFAULT_QUAD_MASS,
            payload_mass=DEFAULT_PAYLOAD_MASS,
            inertias=DEFAULT_QUAD_INERTIA,
            cable_lengths=DEFAULT_CABLE_LENGTH,
        )
        single = QuadrotorCsPayload(
            mass_q=DEFAULT_QUAD_MASS,
            mass_l=DEFAULT_PAYLOAD_MASS,
            inertia=DEFAULT_QUAD_INERTIA,
            cable_length=DEFAULT_CABLE_LENGTH,
        )
        ref_m, in_m = multi.recover(multi_flats)
        ref_s, in_s = single.recover(single_flats)

        assert in_m.thrusts[0] == pytest.approx(in_s.thrust, abs=TOL_EXACT)
        assert in_m.tensions[0] == pytest.approx(in_s.tension, abs=TOL_EXACT)
        np.testing.assert_allclose(in_m.moments[0], in_s.moment, atol=TOL_EXACT)
        np.testing.assert_allclose(ref_m.positions[0], ref_s.position, atol=TOL_EXACT)
        np.testing.assert_allclose(ref_m.velocities[0], ref_s.velocity, atol=TOL_EXACT)
        np.testing.assert_allclose(
            ref_m.cable_attitudes[0], np.asarray(ref_s.cable_attitude), atol=TOL_EXACT
        )
        np.testing.assert_allclose(
            ref_m.cable_angular_velocities[0], ref_s.cable_angular_velocity, atol=TOL_EXACT
        )
        np.testing.assert_allclose(
            np.asarray(ref_m.orientations[0]), np.asarray(ref_s.orientation), atol=TOL_EXACT
        )
        np.testing.assert_allclose(
            np.asarray(ref_m.angular_velocities[0]),
            np.asarray(ref_s.angular_velocity),
            atol=TOL_EXACT,
        )


# ─── Taut-cable enforcement ───────────────────────────────────────────


class TestTautCableEnforcement:
    def test_slackening_trajectory_raises(self):
        """Drive ``F_net`` to exactly cancel cable 0's anchor."""
        f2s = MultiQuadCsPayload(num_quadrotors=3)
        policy = f2s.tension_policy
        # 𝐓_0 = 𝐓̄_0 + α_0 (F_net − F̄_net) = 0  ⇔  F_net − F̄_net = −𝐓̄_0/α_0,
        # and F_net − F̄_net = −m_L a_L, so a_L = 𝐓̄_0/(α_0 m_L).
        a_L = policy.anchors[0] / (policy.shares[0] * f2s.payload_mass)
        x_data = np.zeros((7, 3))
        x_data[2] = a_L
        flats = MultiQuadCsPayloadFlats(x_L=Jet(x_data), psi=[Jet.zeros(2, 1) for _ in range(3)])
        with pytest.raises(ValueError, match="cable 0 would go slack"):
            f2s.recover(flats)

    def test_slack_message_reports_the_acceleration_bound(self):
        f2s = MultiQuadCsPayload(num_quadrotors=3)
        policy = f2s.tension_policy
        a_L = policy.anchors[1] / (policy.shares[1] * f2s.payload_mass)
        x_data = np.zeros((7, 3))
        x_data[2] = a_L
        flats = MultiQuadCsPayloadFlats(x_L=Jet(x_data), psi=[Jet.zeros(2, 1) for _ in range(3)])
        with pytest.raises(ValueError, match=r"m/s²"):
            f2s.recover(flats)

    @pytest.mark.parametrize("nQ", [2, 3, 5])
    def test_acceleration_bound_is_a_valid_certificate(self, nQ):
        """Any payload acceleration below the bound must recover cleanly.

        ``min(tensions) > 0`` alone is near-tautological — the tensions are
        norms and ``recover`` already refuses below ``min_tension`` — so the
        assertion is the *quantitative* certificate: each ``T_i`` must clear
        the triangle-inequality floor ``‖𝐓̄_i‖ - |α_i| ‖Δ‖``, which is exactly
        what the bound claims and is strictly positive inside it.
        """
        f2s = MultiQuadCsPayload(num_quadrotors=nQ)
        policy = f2s.tension_policy
        bound = f2s.payload_acceleration_bound()
        assert np.isfinite(bound) and bound > 0.0
        anchor_norms = np.linalg.norm(policy.anchors, axis=1)
        rng = np.random.default_rng(19)
        for _ in range(60):
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            a_L = 0.99 * bound * direction
            x_data = np.zeros((7, 3))
            x_data[2] = a_L
            flats = MultiQuadCsPayloadFlats(
                x_L=Jet(x_data), psi=[Jet.zeros(2, 1) for _ in range(nQ)]
            )
            ref, inputs = f2s.recover(flats)
            delta = float(np.linalg.norm(f2s.payload_mass * a_L))
            for i in range(nQ):
                floor = anchor_norms[i] - abs(policy.shares[i]) * delta
                assert floor > 0.0
                assert inputs.tensions[i] >= floor - TOL_TIGHT
                # Inside the certificate no cable can have inverted either.
                assert float(np.dot(ref.cable_attitudes[i], policy.hover_directions[i])) > 0.0

    def test_ring_bound_beats_the_legacy_split(self):
        ring = MultiQuadCsPayload(num_quadrotors=3)
        legacy = MultiQuadCsPayload(
            num_quadrotors=3, tension_policy=LegacyFixedTensionSplit(DEFAULT_PAYLOAD_MASS)
        )
        assert ring.payload_acceleration_bound() > 2.0 * legacy.payload_acceleration_bound()

    def test_single_quadrotor_bound_is_free_fall(self):
        # α = 1 and ‖𝐓̄‖ = m_L g, so the certificate is exactly ‖a_L‖ < g —
        # the free-fall / slack-cable singularity of the single-quadrotor map.
        f2s = MultiQuadCsPayload(num_quadrotors=1)
        assert f2s.payload_acceleration_bound(min_tension=0.0) == pytest.approx(
            GRAVITY, abs=TOL_ANALYTIC
        )

    def test_negative_shares_count_against_the_margin(self):
        """A negative ``α_i`` moves ``𝐓_i`` just as fast as a positive one.

        ``net_force_margin`` used to ``continue`` on ``α_i <= 0`` on the
        grounds that "this cable's force never moves" — true only for
        ``α_i == 0``. With ``α = (-1, 2)`` the certificate then reported
        4.155 m/s² while the true bound was 1.500 m/s², and every "certified"
        acceleration in between inverted cable 0.
        """
        mL = 0.2
        anchors = np.array([[0.0, 0.0, -0.3], [0.0, 0.0, -1.662]])
        shares = np.array([-1.0, 2.0])
        np.testing.assert_allclose(
            anchors.sum(axis=0), -mL * GRAVITY * _E3, atol=TOL_TIGHT
        )  # in equilibrium at rest, so the bound's offset term is zero
        policy = AffineTensionSplit(anchors=anchors, shares=shares)
        f2s = MultiQuadCsPayload(num_quadrotors=2, payload_mass=mL, tension_policy=policy)

        true_bound = float(np.min(np.linalg.norm(anchors, axis=1) / np.abs(shares))) / mL
        assert true_bound == pytest.approx(1.5, abs=TOL_ANALYTIC)
        assert f2s.payload_acceleration_bound(min_tension=0.0) == pytest.approx(
            true_bound, abs=TOL_ANALYTIC
        )

        # The accelerations the old bound certified all invert cable 0.
        for magnitude in (2.0, 3.0, 4.0):
            x_data = np.zeros((7, 3))
            x_data[2] = np.array([0.0, 0.0, magnitude])
            flats = MultiQuadCsPayloadFlats(
                x_L=Jet(x_data), psi=[Jet.zeros(2, 1) for _ in range(2)]
            )
            with pytest.raises(ValueError, match="cable 0"):
                f2s.recover(flats)

    def test_bound_accounts_for_anchors_out_of_equilibrium(self):
        """``Δ = F_net - F̄_net`` is ``-m_L a_L`` only when the anchors balance
        the payload weight, which :class:`AffineTensionSplit` never enforces.

        Here the anchors sum to ``-6 e₃`` against a ``-1.962 e₃`` weight, so
        ``‖d‖ = 4.038 N`` of the policy's 6.0 N margin is already spent at
        hover: the honest bound is ``(6 - 4.038)/m_L = g``, not ``6/m_L = 30``.
        """
        mL = 0.2
        policy = AffineTensionSplit(anchors=[[0, 0, -3.0], [0, 0, -3.0]], shares=[0.5, 0.5])
        f2s = MultiQuadCsPayload(num_quadrotors=2, payload_mass=mL, tension_policy=policy)

        assert policy.net_force_margin(0.0) == pytest.approx(6.0, abs=TOL_TIGHT)
        offset = float(np.linalg.norm(-mL * GRAVITY * _E3 - policy.reference_net_force))
        assert offset == pytest.approx(4.038, abs=1e-3)
        assert f2s.payload_acceleration_bound(min_tension=0.0) == pytest.approx(
            GRAVITY, abs=TOL_ANALYTIC
        )

        # Everything the old bound certified up to 30 m/s² is past the crossing.
        for magnitude in (15.0, 27.0, 29.7):
            x_data = np.zeros((7, 3))
            x_data[2] = np.array([0.0, 0.0, -magnitude])
            flats = MultiQuadCsPayloadFlats(
                x_L=Jet(x_data), psi=[Jet.zeros(2, 1) for _ in range(2)]
            )
            with pytest.raises(ValueError, match="cable 0"):
                f2s.recover(flats)

    @pytest.mark.parametrize("scale", [1.0, 1.001, 1.5, 3.0])
    def test_overshooting_the_slack_point_also_raises(self, scale):
        """The map must refuse to *cross* the boundary, not only to land on it.

        ``T_i = ‖𝐓_i‖`` is a norm, so once ``α_i Δ`` overshoots ``-𝐓̄_i`` the
        tension is comfortably positive again — and ``q_i`` has flipped, with
        quadrotor ``i`` hanging *below* the payload. ``scale = 1`` used to be
        the only value that raised; 1.001 and up recovered happily with
        ``q_0 · q̄_0 = -1``.
        """
        f2s = MultiQuadCsPayload(num_quadrotors=3)
        policy = f2s.tension_policy
        a_L = scale * policy.anchors[0] / (policy.shares[0] * f2s.payload_mass)
        x_data = np.zeros((7, 3))
        x_data[2] = a_L
        flats = MultiQuadCsPayloadFlats(x_L=Jet(x_data), psi=[Jet.zeros(2, 1) for _ in range(3)])
        with pytest.raises(ValueError, match="cable 0"):
            f2s.recover(flats)

    def test_crossing_is_the_only_thing_the_sign_check_rejects(self):
        """The sign check must not fire just short of the crossing."""
        f2s = MultiQuadCsPayload(num_quadrotors=3)
        policy = f2s.tension_policy
        a_L = 0.99 * policy.anchors[0] / (policy.shares[0] * f2s.payload_mass)
        x_data = np.zeros((7, 3))
        x_data[2] = a_L
        flats = MultiQuadCsPayloadFlats(x_L=Jet(x_data), psi=[Jet.zeros(2, 1) for _ in range(3)])
        ref, inputs = f2s.recover(flats)
        assert inputs.tensions[0] > 0.0
        assert float(np.dot(ref.cable_attitudes[0], policy.hover_directions[0])) > 0.0


class TestSplitContract:
    """``Σ_i 𝐓_i^(k) = F_net^(k)`` is the policy's contract, and ``recover``
    checks it rather than returning a silently wrong reference."""

    def test_violating_policy_is_rejected(self):
        class _BrokenSplit(RingTensionSplit):
            def split(self, net_force):
                forces = super().split(net_force)
                forces[0][1] = forces[0][1] + np.array([0.0, 0.0, 1.0])
                return forces

        policy = _BrokenSplit(3, DEFAULT_PAYLOAD_MASS, np.ones(3), np.ones(3))
        f2s = MultiQuadCsPayload(num_quadrotors=3, tension_policy=policy)
        with pytest.raises(ValueError, match="violates the split contract at derivative order 1"):
            f2s.recover(_circle_flats(0.3, 3))

    def test_shipped_policies_satisfy_it(self):
        for policy in (None, LegacyFixedTensionSplit(DEFAULT_PAYLOAD_MASS)):
            f2s = MultiQuadCsPayload(num_quadrotors=3, tension_policy=policy)
            f2s.recover(_circle_flats(0.3, 3))


# ─── Kinematic consistency of the recovered reference ─────────────────


class TestKinematicConsistency:
    @pytest.mark.parametrize("nQ", [1, 2, 3, 4])
    @pytest.mark.parametrize("t", [0.0, 0.37, 1.9])
    def test_cable_constraints_hold(self, nQ, t):
        f2s = MultiQuadCsPayload(num_quadrotors=nQ)
        ref, _ = f2s.recover(_circle_flats(t, nQ))
        for i in range(nQ):
            q = np.asarray(ref.cable_attitudes[i])
            omega = np.asarray(ref.cable_angular_velocities[i])
            assert float(np.linalg.norm(q)) == pytest.approx(1.0, abs=TOL_ANALYTIC)
            assert float(q.dot(omega)) == pytest.approx(0.0, abs=TOL_ANALYTIC)

    @pytest.mark.parametrize("t", [0.0, 0.61])
    def test_quadrotor_positions_follow_the_cable_geometry(self, t):
        nQ = 3
        f2s = MultiQuadCsPayload(num_quadrotors=nQ, cable_lengths=[1.0, 1.2, 1.4])
        ref, _ = f2s.recover(_circle_flats(t, nQ))
        for i in range(nQ):
            expected = ref.payload_position - f2s.cable_lengths[i] * np.asarray(
                ref.cable_attitudes[i]
            )
            np.testing.assert_allclose(ref.positions[i], expected, atol=TOL_TIGHT)

    @pytest.mark.parametrize("t", [0.0, 0.61, 1.42])
    def test_recovered_forces_close_the_payload_equation(self, t):
        """``Σ_i 𝐓_i = -m_L (ẍ_L + g e₃)`` for the recovered tensions."""
        nQ = 4
        f2s = MultiQuadCsPayload(num_quadrotors=nQ, masses=[0.8, 0.9, 1.0, 1.1])
        flats = _circle_flats(t, nQ)
        ref, inputs = f2s.recover(flats)
        total = np.zeros(3)
        for i in range(nQ):
            total = total + inputs.tensions[i] * np.asarray(ref.cable_attitudes[i])
        expected = -f2s.payload_mass * (flats.x_L[2] + GRAVITY * _E3)
        np.testing.assert_allclose(total, expected, atol=TOL_ANALYTIC)

    @pytest.mark.parametrize("t", [0.0, 0.61, 1.42])
    def test_thrust_vectors_close_the_quadrotor_equations(self, t):
        """``m_i (ẍ_{Q_i} + g e₃) = f_i R_i e₃ + 𝐓_i`` per quadrotor."""
        nQ = 3
        f2s = MultiQuadCsPayload(
            num_quadrotors=nQ, masses=[0.8, 1.0, 1.2], cable_lengths=[1.0, 1.2, 1.1]
        )
        flats = _circle_flats(t, nQ)
        ref, inputs = f2s.recover(flats)
        for i in range(nQ):
            thrust_vec = inputs.thrusts[i] * (np.asarray(ref.orientations[i]) @ _E3)
            cable_vec = inputs.tensions[i] * np.asarray(ref.cable_attitudes[i])
            lhs = f2s.masses[i] * (ref.accelerations[i] + GRAVITY * _E3)
            np.testing.assert_allclose(lhs, thrust_vec + cable_vec, atol=TOL_ANALYTIC)

    def test_heterogeneous_parameters_recover(self):
        nQ = 4
        f2s = MultiQuadCsPayload(
            num_quadrotors=nQ,
            masses=[0.7, 0.9, 1.1, 1.3],
            cable_lengths=[0.9, 1.1, 1.3, 1.5],
            payload_mass=0.35,
        )
        ref, inputs = f2s.recover(_circle_flats(0.4, nQ))
        assert len(inputs.thrusts) == nQ
        assert all(t > 0.0 for t in inputs.tensions)
        # Heavier quadrotors take a larger dynamic share.
        assert f2s.tension_policy.shares[3] > f2s.tension_policy.shares[0]
        assert ref.num_quadrotors == nQ

    def test_wrench_helper_shape(self):
        nQ = 3
        f2s = MultiQuadCsPayload(num_quadrotors=nQ)
        _, inputs = f2s.recover(_circle_flats(0.2, nQ))
        wrench = inputs.wrench
        assert wrench.shape == (nQ, 4)
        for i in range(nQ):
            assert wrench[i, 0] == pytest.approx(inputs.thrusts[i])
            np.testing.assert_allclose(wrench[i, 1:], inputs.moments[i])


# ─── Per-quadrotor yaw ────────────────────────────────────────────────


class TestPerQuadrotorYaw:
    def test_yaw_hint_is_honoured_per_quadrotor(self):
        """``b_2,i ⊥ [cos ψ_i, sin ψ_i, 0]`` — the defining property of the yaw
        flat output: the heading hint lies in each quadrotor's ``b_1``–``b_3``
        plane, independently per quadrotor."""
        nQ = 4
        yaws = [0.0, 0.9, -1.4, 2.2]
        f2s = MultiQuadCsPayload(num_quadrotors=nQ)
        flats = MultiQuadCsPayloadFlats(
            x_L=Jet.zeros(order=6, dim=3),
            psi=[Jet(np.array([y, 0.0, 0.0])) for y in yaws],
        )
        ref, _ = f2s.recover(flats)
        for i, y in enumerate(yaws):
            b2 = np.asarray(ref.orientations[i])[:, 1]
            hint = np.array([np.cos(y), np.sin(y), 0.0])
            assert float(b2.dot(hint)) == pytest.approx(0.0, abs=TOL_ANALYTIC)

    def test_each_quadrotor_spins_about_its_own_thrust_axis(self):
        """A static payload with per-quadrotor yaw rates: ``b_3,i`` is constant,
        so each body rate is a pure spin about the body ``e_3`` with the sign of
        that quadrotor's yaw rate — and a zero yaw rate gives a zero body rate."""
        nQ = 3
        rates = [0.0, 0.4, -0.7]
        f2s = MultiQuadCsPayload(num_quadrotors=nQ)
        flats = MultiQuadCsPayloadFlats(
            x_L=Jet.zeros(order=6, dim=3),
            psi=[Jet(np.array([0.0, r, 0.0])) for r in rates],
        )
        ref, _ = f2s.recover(flats)
        for i, r in enumerate(rates):
            Omega = np.asarray(ref.angular_velocities[i])
            np.testing.assert_allclose(Omega[:2], np.zeros(2), atol=TOL_ANALYTIC)
            if r == 0.0:
                assert Omega[2] == pytest.approx(0.0, abs=TOL_ANALYTIC)
            else:
                assert np.sign(Omega[2]) == np.sign(r)
                assert abs(Omega[2]) == pytest.approx(abs(r), rel=0.05)

    def test_single_quadrotor_body_rate_is_the_yaw_rate(self):
        """With one quadrotor the hover thrust axis is ``e_3``, so the body rate
        equals the yaw rate exactly (the single-quadrotor result)."""
        f2s = MultiQuadCsPayload(num_quadrotors=1)
        flats = MultiQuadCsPayloadFlats(
            x_L=Jet.zeros(order=6, dim=3), psi=[Jet(np.array([0.0, 0.5, 0.0]))]
        )
        ref, _ = f2s.recover(flats)
        np.testing.assert_allclose(
            np.asarray(ref.angular_velocities[0]), np.array([0.0, 0.0, 0.5]), atol=TOL_ANALYTIC
        )

    def test_yaw_acceleration_drives_the_moment(self):
        f2s = MultiQuadCsPayload(num_quadrotors=1)
        flats = MultiQuadCsPayloadFlats(
            x_L=Jet.zeros(order=6, dim=3), psi=[Jet(np.array([0.0, 0.5, 0.3]))]
        )
        _, inputs = f2s.recover(flats)
        np.testing.assert_allclose(inputs.moments[0][:2], np.zeros(2), atol=TOL_ANALYTIC)
        assert inputs.moments[0][2] == pytest.approx(
            DEFAULT_QUAD_INERTIA[2, 2] * 0.3, abs=TOL_ANALYTIC
        )
