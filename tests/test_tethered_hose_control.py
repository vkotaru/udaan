"""M7 / M8 hose LQR: the pinned linearisation and the gain carried round the anchor."""

import numpy as np
import pytest

from udaan.control.tethered_hose import TetheredHoseController
from udaan.models.tethered_hose import TetheredHoseBase, TetheredQuadrotorBase


def _at(top):
    top = np.asarray(top, dtype=float)
    return lambda t: (top, np.zeros(3), np.zeros(3))


class TestLinearisation:
    @pytest.mark.parametrize("cls", [TetheredQuadrotorBase, TetheredHoseBase])
    def test_matches_finite_differences_of_the_model(self, cls):
        """A = ∂ṡ/∂s by central differences of the model's own accelerations at the trim."""
        m = cls()
        ctrl = m.hose_controller
        ref, ff = ctrl.equilibrium_reference(0.0)
        A, B = ctrl.linearize(ref, ff)
        n, nQ = m.num_links, m.num_quadrotors
        qd = [np.asarray(q) for q in ref.link_attitudes]
        Rd = [np.asarray(R) for R in ref.orientations]
        wrench = ff.as_wrench()

        def rates(xi, dwrench):
            q = [qd[i] + np.cross(xi[3 * i : 3 * i + 3], qd[i]) for i in range(n)]
            m.reset(link_attitudes=[v / np.linalg.norm(v) for v in q], orientations=Rd)
            w = wrench + dwrench
            return m.accelerations(w[:, 0], w[:, 1:])["link_ang_accels"].reshape(-1)

        eps = 1e-6
        rate0 = 3 * n
        # Link-direction columns of the ω̇ rows: ξ_i = q_d × q, so q = q_d + ξ × q_d.
        for col in range(3 * n):
            d = np.zeros(3 * n)
            d[col] = eps
            numeric = (rates(d, 0.0) - rates(-d, 0.0)) / (2 * eps)
            # Compare on the tangent directions only (the gauge rows are free).
            for i in range(n):
                P = np.eye(3) - np.outer(qd[i], qd[i])
                np.testing.assert_allclose(
                    P @ A[rate0 + 3 * i : rate0 + 3 * i + 3, col],
                    P @ numeric[3 * i : 3 * i + 3],
                    atol=1e-5,
                )
        for k in range(m.num_quadrotors):
            d = np.zeros((nQ, 4))
            d[k, 0] = eps
            numeric = (rates(np.zeros(3 * n), d) - rates(np.zeros(3 * n), -d)) / (2 * eps)
            np.testing.assert_allclose(B[rate0 : rate0 + 3 * n, k], numeric, atol=1e-5)

    def test_m7_poles_at_the_default_trim(self):
        ctrl = TetheredHoseController()
        poles = ctrl.closed_loop_poles
        assert poles[0].real == pytest.approx(-0.6035, abs=1e-3)
        assert abs(poles[0].imag) == pytest.approx(12.685, abs=1e-2)
        assert ctrl.controllability_margin == pytest.approx(0.0975, abs=1e-3)

    def test_root_weights_are_refused(self):
        with pytest.raises(ValueError, match="q_root_position"):
            TetheredHoseController(q_root_position=1.0)

    def test_the_trim_is_read_only(self):
        with pytest.raises(AttributeError):
            TetheredHoseController().trim_tensions = np.zeros((1, 3))


class TestAzimuthTransport:
    """A gain solved at azimuth 0 regulates a trim elsewhere round the anchor."""

    @staticmethod
    def _fly_from_kick(model, ctrl, top, seconds=8.0):
        ref, _ = ctrl.equilibrium_reference(1.0)
        rng = np.random.default_rng(4)
        kicked = []
        for q in ref.link_attitudes:
            v = np.asarray(q) + 0.05 * rng.normal(size=3)
            kicked.append(v / np.linalg.norm(v))
        model.reset(link_attitudes=kicked)
        while model.t < seconds:
            u = model.compute_hose_input()
            if not np.all(np.isfinite(u)):
                return np.inf
            model.step(u)
        return float(np.linalg.norm(np.asarray(model.state.positions[-1]) - top))

    @pytest.mark.parametrize("degrees", [90.0, 180.0])
    def test_a_trim_elsewhere_converges(self, degrees):
        """The gain is solved at ``setpoint(0)`` (azimuth 0); every later instant asks for a station turned by ``degrees``."""
        m = TetheredQuadrotorBase()
        ctrl = m.hose_controller
        theta = np.radians(degrees)
        start = np.array([1.0, 0.0, 1.6])
        top = np.array([np.cos(theta), np.sin(theta), 1.6])
        ctrl.setpoint = lambda t: (start if t <= 0.0 else top, np.zeros(3), np.zeros(3))
        ctrl.gain  # noqa: B018  (solved at azimuth 0)
        assert ctrl.transport(ctrl.equilibrium_reference(1.0)[0])[0][0, 0] == pytest.approx(
            np.cos(theta)
        )
        assert self._fly_from_kick(m, ctrl, top) < 1e-3

    def test_the_transport_carries_the_gain_exactly_to_180_degrees(self):
        """At 180° the transported gain is the gain a fresh solve gives there."""
        at_zero = TetheredHoseController()
        top = np.array([-1.0, 0.0, 1.6])
        there = TetheredHoseController(setpoint=_at(top))
        ref, _ = there.equilibrium_reference(0.0)
        E, Ys = at_zero.transport(ref)
        np.testing.assert_allclose(E, np.diag([-1.0, -1.0, 1.0]), atol=1e-12)
        A, B = there.linearize(*there.equilibrium_reference(0.0))
        W = there.tangent_projection(ref.link_attitudes)
        # The transported law as a matrix on the embedded error.
        n, nQ = at_zero.num_links, at_zero.num_quadrotors
        T = np.eye(at_zero.state_dim)
        for i in range(2 * n):
            T[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = E.T
        for base in (6 * n, 6 * n + 3 * nQ):
            T[base : base + 3, base : base + 3] = Ys[0].T
        S = np.eye(4)
        S[1:, 1:] = Ys[0]
        poles = np.linalg.eigvals(W.T @ (A + B @ S @ at_zero.gain @ T) @ W)
        assert poles.real.max() == pytest.approx(there.closed_loop_poles[0].real, abs=1e-6)


class TestSetpointRebinding:
    def test_a_new_station_gets_its_own_gain(self):
        """Rebinding the setpoint to another radius and height re-solves the gain there."""
        moved = np.array([0.7, 0.0, 1.35])
        ctrl = TetheredHoseController()
        ctrl.gain  # noqa: B018  (solved at the default station)
        ctrl.setpoint = _at(moved)
        fresh = TetheredHoseController(setpoint=_at(moved))
        np.testing.assert_allclose(ctrl.gain, fresh.gain, atol=1e-9)
        np.testing.assert_allclose(ctrl.trim_tensions, fresh.trim_tensions, atol=1e-12)

    def test_the_feedforward_drop_count_is_flight_only(self):
        """A sample the pre-flight ground check cannot recover is not a dropped flight step."""
        from udaan.flatness.tethered_hose import TetherHold

        class RefusedAtTheEnd(TetherHold):
            def flats(self, t):
                if t >= 0.05:  # the pre-flight grid reaches tf; the last flown step is 0.045 s
                    raise ValueError("refused")
                return super().flats(t)

        m = TetheredQuadrotorBase()
        ref = RefusedAtTheEnd(m.hose_controller.flat_map, [1.0, 0.0, 1.6])
        m.hose_controller.setpoint = ref.get
        m.simulate(0.05)
        assert m.feedforward_drops == 0
