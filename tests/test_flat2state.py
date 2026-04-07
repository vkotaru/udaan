import numpy as np

from udaan.utils.flat2state import Flat2State
from udaan.utils.trajectory import circleXY, setpoint


class TestFlat2StateQuadrotor:
    def test_hover_setpoint(self):
        traj = setpoint(0, sp=np.array([0.0, 0.0, 1.0]))
        s = Flat2State.quadrotor(traj)

        # At hover, R should be identity
        np.testing.assert_allclose(s["R"], np.eye(3), atol=1e-10)
        # Omega should be zero
        np.testing.assert_allclose(s["Omega"], np.zeros(3), atol=1e-10)
        # thrust should equal mg
        assert s["f"] > 0

    def test_returns_expected_keys(self):
        traj = setpoint(0)
        s = Flat2State.quadrotor(traj)
        for key in ["R", "Omega", "dOmega", "M", "f", "fb3", "xQ", "vQ", "aQ"]:
            assert key in s

    def test_circle_trajectory(self):
        traj = circleXY(1.0, r=1.0)
        s = Flat2State.quadrotor(traj)
        # R should be a valid rotation matrix
        R = s["R"]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-8)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-8)


class TestFlat2StatePayload:
    def test_hover_setpoint(self):
        traj = setpoint(0, sp=np.array([0.0, 0.0, 1.0]))
        s = Flat2State.quadrotor_payload(traj, mQ=0.85, mL=0.1, cable_len=1.0)

        for key in ["R", "xL", "xQ", "q", "omega"]:
            assert key in s

        # cable direction should be unit vector
        np.testing.assert_allclose(np.linalg.norm(s["q"]), 1.0, atol=1e-8)

    def test_circle_trajectory(self):
        traj = circleXY(1.0, r=0.5, w=0.1)
        s = Flat2State.quadrotor_payload(traj, mQ=0.85, mL=0.1, cable_len=0.5)

        R = s["R"]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(s["q"]), 1.0, atol=1e-8)


class TestComputeQVectors:
    def test_hover(self):
        aL = np.zeros(3)
        daL = np.zeros(3)
        d2aL = np.zeros(3)
        d3aL = np.zeros(3)
        d4aL = np.zeros(3)

        q, dq, d2q, d3q, d4q, Tp, dTp, d2Tp = Flat2State.compute_q_vectors(
            aL, daL, d2aL, d3aL, d4aL
        )

        # At hover, cable should point down (negative z)
        np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-8)
        assert q[2] < 0  # pointing down
        np.testing.assert_allclose(dq, np.zeros(3), atol=1e-8)
