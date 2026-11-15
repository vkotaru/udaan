"""``udaan run quad-payload``: the trajectory it flies and the cable length it flies on.

The ``recorded`` fixture replaces ``simulate`` with a recorder, so those tests
read the reference and the model the CLI built and never fly them; the
end-to-end tests run the real ``simulate()`` on the analytical model, headless.
"""

import numpy as np
import pytest
from typer.testing import CliRunner

from udaan.cli import app
from udaan.models.quadrotor_cspayload import QuadrotorCsPayloadBase


@pytest.fixture
def recorded(monkeypatch):
    seen = {}

    def record(self, tf, **kwargs):
        seen["model"] = self
        seen["setpoint"] = self.payload_controller.setpoint
        seen["tf"] = tf
        seen["start"] = np.asarray(kwargs["payload_position"], dtype=float)

    monkeypatch.setattr(QuadrotorCsPayloadBase, "simulate", record)
    return seen


def _invoke(*args):
    return CliRunner().invoke(app, ["run", "quad-payload", "--no-render", *args])


def _run(*args):
    result = _invoke(*args)
    assert result.exit_code == 0, result.output
    return result


class TestTrajectory:
    def test_hover_is_the_default_and_keeps_the_model_setpoint(self, recorded):
        _run("-m", "base", "-t", "3")
        default = QuadrotorCsPayloadBase().payload_controller.setpoint
        np.testing.assert_allclose(recorded["setpoint"](1.0)[0], default(1.0)[0])
        np.testing.assert_allclose(recorded["start"], [-1.0, 2.0, 0.5])
        assert recorded["tf"] == 3.0

    @pytest.mark.parametrize(
        ("traj", "cls"), [("circle", "CircularTraj"), ("spiral", "HelixTrajectory")]
    )
    def test_binds_a_trajectory_with_a_jet_so_the_feedforward_engages(self, recorded, traj, cls):
        _run("-m", "base", "--traj", traj, "-t", "6")
        owner = recorded["setpoint"].__self__
        assert type(owner).__name__ == cls
        assert callable(getattr(owner, "get_jet", None))
        assert len(recorded["setpoint"](1.5)) == 5  # jerk and snap reach the cable loop

    @pytest.mark.parametrize("traj", ["circle", "spiral"])
    def test_payload_starts_on_its_reference_clear_of_the_ground(self, recorded, traj):
        _run("-m", "base", "--traj", traj, "-t", "6")
        np.testing.assert_allclose(recorded["start"], recorded["setpoint"](0.0)[0], atol=1e-12)
        heights = [recorded["setpoint"](t)[0][2] for t in np.linspace(0, recorded["tf"], 400)]
        assert min(heights) >= 1.5 - 1e-9

    def test_spiral_is_the_horizontal_helix_and_sets_its_own_duration(self, recorded):
        _run("-m", "base", "--traj", "spiral", "-t", "5", "-p", "1,2,3")
        helix = recorded["setpoint"].__self__
        assert recorded["tf"] == pytest.approx(helix.duration)
        np.testing.assert_allclose(recorded["start"], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(helix.end_position, [1.0, 2.0, 4.0])

    def test_circle_is_centred_on_the_position(self, recorded):
        _run("-m", "base", "--traj", "circle", "-t", "6", "-p", "1,2,3")
        pts = np.array([recorded["setpoint"](t)[0] for t in np.linspace(0, 6, 50)])
        np.testing.assert_allclose(np.linalg.norm(pts[:, :2] - [1.0, 2.0], axis=1), 1.0)
        np.testing.assert_allclose(pts[:, 2], 3.0)

    def test_circle_starts_from_rest_and_reaches_one_metre_per_second(self, recorded):
        _run("-m", "base", "--traj", "circle", "-t", "6")
        np.testing.assert_allclose(recorded["setpoint"](0.0)[1], 0.0, atol=1e-12)
        assert np.linalg.norm(recorded["setpoint"](3.0)[1]) == pytest.approx(1.0)

    def test_unknown_trajectory_lists_the_choices(self):
        result = _invoke("-m", "base", "--traj", "nope")
        assert result.exit_code == 1
        for name in ("hover", "circle", "spiral"):
            assert name in result.output

    @pytest.mark.parametrize("traj", ["hover", "circle", "spiral"])
    def test_flies_end_to_end_on_the_analytical_model(self, traj):
        result = _run("-m", "base", "--traj", traj, "-t", "2")
        assert f"base, {traj}" in result.output

    @staticmethod
    def _payload_errors(monkeypatch):
        errors = []
        step = QuadrotorCsPayloadBase.step

        def logged(self, u, **kwargs):
            ref = self.payload_controller.setpoint(self.t)[0]
            errors.append(np.linalg.norm(self.state.payload_position - ref))
            return step(self, u, **kwargs)

        monkeypatch.setattr(QuadrotorCsPayloadBase, "step", logged)
        return errors

    def test_circle_has_no_start_up_transient(self, monkeypatch):
        """The real closed loop on the circle, judged over the whole run from t = 0.

        Measured max 0.055 cm on the analytical model; the bound is 0.25 cm.
        """
        errors = self._payload_errors(monkeypatch)
        _run("-m", "base", "--traj", "circle", "-t", "10")
        assert len(errors) == 2000
        assert max(errors) < 0.0025

    def test_helix_tracks_the_payload_to_a_centimetre(self, monkeypatch):
        """The real closed loop on the default helix (2 m/s, 0.8 m, 1 m climb)."""
        errors = self._payload_errors(monkeypatch)
        _run("-m", "base", "--traj", "spiral")
        assert len(errors) > 1500  # the whole 9 s manoeuvre
        assert np.sqrt(np.mean(np.square(errors))) < 0.01
        assert max(errors) < 0.02


class TestCableLength:
    def test_default_is_the_model_default(self, recorded):
        _run("-m", "base", "-t", "1")
        assert recorded["model"].cable_length == QuadrotorCsPayloadBase().cable_length

    def test_honoured_on_the_analytical_model_and_its_controller(self, recorded):
        _run("-m", "base", "--cable-length", "2.5", "-t", "1")
        mdl = recorded["model"]
        assert mdl.cable_length == 2.5
        assert mdl.payload_controller.cable_length == 2.5

    def test_the_payload_hangs_a_cable_length_below_the_airframe(self, monkeypatch):
        seen = {}
        simulate = QuadrotorCsPayloadBase.simulate

        def run(self, tf, **kwargs):
            simulate(self, tf, **kwargs)
            seen["model"] = self

        monkeypatch.setattr(QuadrotorCsPayloadBase, "simulate", run)
        _run("-m", "base", "--cable-length", "2.5", "-t", "3")
        s = seen["model"].state
        assert np.linalg.norm(s.position - s.payload_position) == pytest.approx(2.5, abs=1e-6)
        # The controller carries the new length: the payload settles on its setpoint.
        target = seen["model"].payload_controller.setpoint(3.0)[0]
        assert abs(s.payload_position[2] - target[2]) < 0.05

    @pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
    def test_rejects_a_non_positive_or_non_finite_length(self, value):
        result = _invoke("-m", "base", f"--cable-length={value}", "-t", "1")
        assert result.exit_code == 1
        assert "--cable-length" in result.output

    @pytest.mark.mujoco
    def test_honoured_on_the_tendon(self, recorded):
        _run("-m", "mujoco", "-c", "tendon", "--cable-length", "2.5", "-t", "1")
        mdl = recorded["model"]
        assert mdl.cable_length == 2.5
        assert mdl.payload_controller.cable_length == 2.5
        assert mdl._mjMdl.model.tendon_range[mdl._mj_cable_index][1] == 2.5

    @pytest.mark.mujoco
    def test_a_different_length_is_refused_on_the_link_chain(self):
        result = _invoke("-m", "mujoco", "-c", "links", "--cable-length", "2", "-t", "1")
        assert result.exit_code == 1
        assert "link chain" in result.output

    @pytest.mark.mujoco
    def test_the_link_chains_own_length_is_accepted(self, recorded):
        _run("-m", "mujoco", "-c", "links", "--cable-length", "1", "-t", "1")
        assert recorded["model"].cable_length == pytest.approx(1.0)
