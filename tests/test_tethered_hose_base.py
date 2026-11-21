"""M7 / M8 analytical model: the pinned reduced dynamics, the trim, the reset and the gate."""

import logging

import numpy as np
import pytest

from udaan.manif import SO3
from udaan.models.tethered_hose import TetheredHoseBase, TetheredQuadrotorBase
from udaan.utils.hose import tethered_trim, tethered_vehicle_positions

G = 9.81


def _random_state(model, rng, spread=0.6):
    """A random configuration around 'straight up', with random rates."""
    n, nQ = model.num_links, model.num_quadrotors
    qs = []
    for _ in range(n):
        v = np.array([0.0, 0.0, 1.0]) + spread * rng.normal(size=3)
        qs.append(v / np.linalg.norm(v))
    rates = [0.5 * rng.normal(size=3) for _ in range(n)]
    tilt = [np.array([0.0, 0.0, 1.0]) + 0.3 * rng.normal(size=3) for _ in range(nQ)]
    orientations = [
        np.asarray(SO3.from_two_vectors(b / np.linalg.norm(b), np.array([1.0, 0.0, 0.0])))
        for b in tilt
    ]
    body_rates = [0.5 * rng.normal(size=3) for _ in range(nQ)]
    model.reset(
        link_attitudes=qs,
        link_angular_velocities=rates,
        orientations=orientations,
        angular_velocities=body_rates,
    )


class TestDefaults:
    def test_m7_is_one_quadrotor_on_a_two_metre_four_link_tether(self):
        m = TetheredQuadrotorBase()
        assert m.attachments == (4,)
        assert m.hose_length == pytest.approx(2.0)
        assert m.vertex_masses.sum() == pytest.approx(0.5)
        np.testing.assert_array_equal(m.anchor, [0.0, 0.0, 0.0])

    def test_m8_is_three_quadrotors_two_links_apart_on_three_metres(self):
        m = TetheredHoseBase()
        assert m.attachments == (2, 4, 6)
        assert m.hose_length == pytest.approx(3.0)
        assert m.vertex_masses.sum() == pytest.approx(0.75)

    def test_num_quadrotors_spreads_the_vehicles_two_links_apart(self):
        assert TetheredHoseBase(num_quadrotors=5).attachments == (2, 4, 6, 8, 10)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"num_links": 4, "attachments": (0, 4)},
            {"num_links": 4, "attachments": (2,)},
            {"sag_ratio": 0.5},
            {"anchor_height": -0.1},
        ],
    )
    def test_patterns_and_keywords_a_pinned_hose_has_no_use_for_are_refused(self, kwargs):
        with pytest.raises(ValueError):
            TetheredHoseBase(**kwargs)

    def test_m7_refuses_a_second_vehicle(self):
        with pytest.raises(ValueError, match="one quadrotor"):
            TetheredQuadrotorBase(num_links=4, attachments=(2, 4))


class TestPinnedDynamics:
    @pytest.mark.parametrize("cls", [TetheredQuadrotorBase, TetheredHoseBase])
    def test_power_balance(self, cls):
        """d/dt (T + V) equals the power of the thrusts and moments, at random states.

        The pin does no work (the anchor does not move), so this identity of the
        reduced equations holds only if the deleted root row really is the
        constraint's and nothing else.
        """
        rng = np.random.default_rng(3)
        m = cls()
        e3 = np.array([0.0, 0.0, 1.0])
        for _ in range(5):
            _random_state(m, rng)
            f = rng.uniform(5.0, 15.0, size=m.num_quadrotors)
            M = 0.05 * rng.normal(size=(m.num_quadrotors, 3))
            acc = m.accelerations(f, M)
            mbar = m.vertex_total_masses
            v = m.vertex_velocities()
            rate = float(np.sum(mbar[1:, None] * v[1:] * (acc["vertex_accels"][1:] + G * e3)))
            power = 0.0
            for k, vertex in enumerate(m.attachments):
                R = np.asarray(m.state.orientations[k])
                Om = np.asarray(m.state.angular_velocities[k])
                rate += float(Om @ (m.inertias[k] @ acc["quad_ang_accels"][k]))
                power += float(f[k] * (R @ e3) @ v[vertex] + M[k] @ Om)
            assert rate == pytest.approx(power, rel=1e-9, abs=1e-9)

    def test_the_anchor_does_not_move(self):
        m = TetheredQuadrotorBase()
        _random_state(m, np.random.default_rng(1), spread=0.1)
        for _ in range(200):
            m.step(m.hover_input())
        np.testing.assert_array_equal(m.state.root_position, m.anchor)
        np.testing.assert_array_equal(m.state.root_velocity, np.zeros(3))

    @pytest.mark.parametrize("cls", [TetheredQuadrotorBase, TetheredHoseBase])
    def test_the_trim_holds_open_loop(self, cls):
        m = cls()
        start = np.array(m.state.positions)
        for _ in range(400):
            m.step(m.hover_input())
        assert np.abs(np.array(m.state.positions) - start).max() < 1e-12

    def test_the_anchor_force_is_the_trims(self):
        m = TetheredHoseBase()
        trim = m.hover_trim()
        acc = m.accelerations(trim["wrench"][:, 0], trim["wrench"][:, 1:])
        np.testing.assert_allclose(acc["anchor_force"], trim["anchor_force"], atol=1e-12)
        assert np.abs(acc["link_ang_accels"]).max() < 1e-12

    def test_reset_refuses_the_root_coordinates(self):
        m = TetheredQuadrotorBase()
        with pytest.raises(ValueError):
            m.reset(root_position=np.array([0.0, 0.0, 1.0]))
        with pytest.raises(ValueError):
            m.reset(root_velocity=np.zeros(3))


class TestTrim:
    def test_matches_the_static_balance_at_every_vertex(self):
        m = TetheredHoseBase()
        trim = tethered_trim(
            m.link_lengths,
            m.vertex_total_masses,
            m.attachments,
            tethered_vehicle_positions([1.5, 0.0, 2.4], 3, m.anchor),
            anchor=m.anchor,
        )
        n = m.num_links
        forces = np.zeros((n + 1, 3))
        for k, v in enumerate(m.attachments):
            forces[v] = trim.thrust_vectors[k]
        for j in range(1, n + 1):
            above = trim.tensions[j] if j < n else np.zeros(3)
            residual = (
                forces[j]
                - trim.tensions[j - 1]
                + above
                - m.vertex_total_masses[j] * G * np.eye(3)[2]
            )
            assert np.abs(residual).max() < 1e-9
        np.testing.assert_allclose(
            trim.vertex_positions[list(m.attachments)][-1], [1.5, 0, 2.4], atol=1e-12
        )

    def test_refuses_points_farther_apart_than_the_hose(self):
        m = TetheredQuadrotorBase()
        with pytest.raises(ValueError, match="cannot be joined"):
            tethered_trim(
                m.link_lengths, m.vertex_total_masses, (4,), [[2.0, 0.0, 1.0]], anchor=m.anchor
            )


class TestStabilityGate:
    def test_eight_links_on_two_metres_warns_and_four_does_not(self, caplog):
        with caplog.at_level(logging.WARNING):
            TetheredQuadrotorBase()
        assert "sampled closed loop" not in caplog.text
        with caplog.at_level(logging.WARNING):
            m = TetheredQuadrotorBase(num_links=8)
        assert "sampled closed loop grows" in caplog.text
        assert m.stability_growth_rate() == pytest.approx(0.247, abs=0.005)

    @pytest.mark.slow
    @pytest.mark.parametrize("n,grows", [(4, False), (8, True)])
    def test_the_gate_predicts_the_simulated_loop(self, n, grows):
        """The growth rate's sign is what the analytical loop does from a small kick."""
        m = TetheredQuadrotorBase(num_links=n)
        ref, _ = m.hose_controller.equilibrium_reference(0.0)
        rng = np.random.default_rng(0)
        kicked = []
        for q in ref.link_attitudes:
            v = np.asarray(q) + 0.02 * rng.normal(size=3)
            kicked.append(v / np.linalg.norm(v))
        m.reset(link_attitudes=kicked)
        log = []
        while m.t < 12.0:
            m.step(m.compute_hose_input())
            err = max(
                float(np.linalg.norm(np.asarray(q) - np.asarray(qd)))
                for q, qd in zip(m.state.link_attitudes, ref.link_attitudes, strict=True)
            )
            log.append((m.t, err))
        log = np.array(log)
        early = log[(log[:, 0] > 4.0) & (log[:, 0] < 6.0), 1].max()
        late = log[log[:, 0] > 10.0, 1].max()
        assert (late > early) == grows


class TestClosedLoop:
    def test_m7_recovers_from_a_kicked_tether(self):
        m = TetheredQuadrotorBase()
        ref, _ = m.hose_controller.equilibrium_reference(0.0)
        rng = np.random.default_rng(1)
        kicked = []
        for q in ref.link_attitudes:
            v = np.asarray(q) + 0.1 * rng.normal(size=3)
            kicked.append(v / np.linalg.norm(v))
        m.simulate(10.0, link_attitudes=kicked)
        assert np.linalg.norm(m.state.positions[-1] - np.array([1.0, 0.0, 1.6])) < 1e-3

    def test_a_reference_below_the_ground_is_warned_about(self, caplog):
        from udaan.flatness.tethered_hose import TetherHold

        m = TetheredQuadrotorBase()
        low = TetherHold(m.hose_controller.flat_map, [1.0, 0.0, 1.3])
        m.hose_controller.setpoint = low.get
        with caplog.at_level(logging.WARNING):
            m.simulate(0.05)
        assert "below the ground plane" in caplog.text
        assert m.reference_ground_clearance < 0.0

    def test_the_default_station_is_above_the_ground(self, caplog):
        m = TetheredQuadrotorBase()
        with caplog.at_level(logging.WARNING):
            m.simulate(0.05)
        assert "below the ground plane" not in caplog.text
        assert m.reference_ground_clearance > 0.0


class TestReferenceRefusals:
    def test_a_refusal_mid_flight_runs_against_the_last_reference(self):
        from udaan.flatness.tethered_hose import TetherHold

        class RefusedLater(TetherHold):
            def flats(self, t):
                if 0.1 <= t < 0.2:
                    raise ValueError("refused")
                return super().flats(t)

        m = TetheredQuadrotorBase()
        ref = RefusedLater(m.hose_controller.flat_map, [1.0, 0.0, 1.6])
        m.hose_controller.setpoint = ref.get
        m.simulate(0.3)
        assert m.feedforward_drops == 20
        assert np.linalg.norm(m.state.positions[-1] - np.array([1.0, 0.0, 1.6])) < 1e-9

    def test_a_reference_beyond_the_thrust_limit_is_warned_about(self, caplog):
        from udaan.flatness.tethered_hose import TetherCircle

        m = TetheredQuadrotorBase(hose_length=20.0, hose_mass=1.0)
        fast = TetherCircle(m.hose_controller.flat_map, [10.0, 0.0, 16.0], rate=0.7, ramp_time=1.0)
        m.hose_controller.setpoint = fast.get
        with caplog.at_level(logging.WARNING):
            m.simulate(1.0)
        assert "of thrust" in caplog.text

    def test_the_default_station_asks_nothing_beyond_the_limits(self, caplog):
        m = TetheredHoseBase()
        with caplog.at_level(logging.WARNING):
            m.simulate(0.05)
        assert "of thrust" not in caplog.text


class TestM8Gate:
    @pytest.mark.parametrize("nq", [2, 3, 5])
    def test_two_links_per_segment_decay(self, nq):
        assert TetheredHoseBase(num_quadrotors=nq).stability_growth_rate() < 0.0

    def test_six_links_on_two_metres_still_decay(self):
        assert TetheredQuadrotorBase(num_links=6).stability_growth_rate() == pytest.approx(
            -0.13, abs=0.01
        )


def test_the_generator_scales_the_hose_mass_with_the_length(tmp_path):
    import mujoco

    from udaan.utils.assets.xml_model_generator import tethered_hose

    path = tethered_hose(
        num_links=6,
        attachments=(2, 4, 6),
        hose_length=3.0,
        filename=str(tmp_path / "t.xml"),
        verbose=False,
    )
    model = mujoco.MjModel.from_xml_path(path)
    hose = sum(
        model.body_mass[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"hose_link{i}")]
        for i in range(6)
    )
    root = model.body_mass[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hose_vertex0")]
    assert hose + root == pytest.approx(0.75)
