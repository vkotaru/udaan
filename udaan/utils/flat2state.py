import numpy as np

from ..core.defaults import GRAVITY
from ..manif import vee


class Flat2State:
    """Differential flatness maps for quadrotor and quadrotor-payload systems."""

    @staticmethod
    def compute_moment(
        axQ,
        daxQ,
        d2axQ,
        mQ=0.85,
        J=None,
        Tp=np.zeros(3),
        dTp=np.zeros(3),
        d2Tp=np.zeros(3),
    ):
        if J is None:
            J = np.array(
                [
                    [0.005315307431627, 0.000005567447099, 0.000005445855427],
                    [0.000005567447099, 0.004949258422243, 0.000020951458431],
                    [0.000005445855427, 0.000020951458431, 0.009806225007686],
                ]
            )
        g = GRAVITY
        e1 = np.array([1.0, 0.0, 0.0])
        e3 = np.array([0.0, 0.0, 1.0])

        b1d = e1
        db1d = np.zeros(3)
        d2b1d = np.zeros(3)

        fb3 = np.dot(mQ, axQ + np.dot(g, e3)) - Tp
        norm_fb3 = np.linalg.norm(fb3)
        f = norm_fb3
        b3 = np.divide(fb3, norm_fb3)
        b3_b1d = np.cross(b3, b1d)
        norm_b3_b1d = np.linalg.norm(b3_b1d)
        b1 = np.divide(-np.cross(b3, b3_b1d), norm_b3_b1d)
        b2 = np.cross(b3, b1)
        R = np.array(np.vstack((b1, b2, b3)))

        dfb3 = np.dot(mQ, daxQ) - dTp
        dnorm_fb3 = np.divide(fb3.dot(dfb3), norm_fb3)
        db3 = np.divide(np.multiply(dfb3, norm_fb3) - np.multiply(fb3, dnorm_fb3), norm_fb3**2.0)
        db3_b1d = np.cross(db3, b1d) + np.cross(b3, db1d)
        dnorm_b3_b1d = np.divide(b3_b1d.dot(db3_b1d), norm_b3_b1d)
        db1 = np.divide(
            -np.cross(db3, b3_b1d) - np.cross(b3, db3_b1d) - np.multiply(b1, dnorm_b3_b1d),
            norm_b3_b1d,
        )
        db2 = np.cross(db3, b1) + np.cross(b3, db1)
        dR = np.array(np.vstack((db1, db2, db3)))

        Omega = vee(np.dot(dR, R.conj().T))

        d2fb3 = np.dot(mQ, d2axQ) - d2Tp
        d2norm_fb3 = np.divide(
            dfb3.dot(dfb3) + fb3.dot(d2fb3) - np.dot(dnorm_fb3, dnorm_fb3), norm_fb3
        )
        d2b3 = np.divide(
            np.dot(
                np.dot(d2fb3, norm_fb3)
                + np.dot(dfb3, dnorm_fb3)
                - np.dot(dfb3, dnorm_fb3)
                - np.dot(fb3, d2norm_fb3),
                norm_fb3**2.0,
            )
            - np.dot(np.dot(np.dot(db3, norm_fb3**2.0) * 2.0, norm_fb3), dnorm_fb3),
            norm_fb3**4.0,
        )
        d2b3_b1d = (
            np.cross(d2b3, b1d) + np.cross(db3, db1d) + np.cross(db3, db1d) + np.cross(b3, d2b1d)
        )
        d2norm_b3_b1d = np.divide(
            np.dot(db3_b1d.dot(db3_b1d) + b3_b1d.dot(d2b3_b1d), norm_b3_b1d)
            - np.dot(b3_b1d.dot(db3_b1d), dnorm_b3_b1d),
            norm_b3_b1d**2.0,
        )
        d2b1 = np.divide(
            np.dot(
                -np.cross(d2b3, b3_b1d)
                - np.cross(db3, db3_b1d)
                - np.cross(db3, db3_b1d)
                - np.cross(b3, d2b3_b1d)
                - np.dot(db1, dnorm_b3_b1d)
                - np.dot(b1, d2norm_b3_b1d),
                norm_b3_b1d,
            )
            - np.dot(np.dot(db1, norm_b3_b1d), dnorm_b3_b1d),
            norm_b3_b1d**2.0,
        )
        d2b2 = np.cross(d2b3, b1) + np.cross(db3, db1) + np.cross(db3, db1) + np.cross(b3, d2b1)
        d2R = np.array(np.vstack((d2b1, d2b2, d2b3)))
        dOmega = vee(np.dot(dR, dR.conj().T) + np.dot(d2R, R.conj().T))
        M = np.dot(J, dOmega) + np.cross(Omega, np.dot(J, Omega))

        s = {}
        s["R"] = R
        s["Omega"] = Omega
        s["dOmega"] = dOmega
        s["M"] = M
        s["f"] = f
        s["fb3"] = fb3
        return s

    @staticmethod
    def quadrotor(traj, mQ=0.85, J=None):
        if J is None:
            J = np.array(
                [
                    [0.005315307431627, 0.000005567447099, 0.000005445855427],
                    [0.000005567447099, 0.004949258422243, 0.000020951458431],
                    [0.000005445855427, 0.000020951458431, 0.009806225007686],
                ]
            )

        xQ = traj["x"]
        vxQ = traj["dx"]
        axQ = traj["d2x"]
        daxQ = traj["d3x"]
        d2axQ = traj["d4x"]

        s = Flat2State.compute_moment(axQ, daxQ, d2axQ, mQ, J)
        s["xQ"] = xQ
        s["vQ"] = vxQ
        s["aQ"] = axQ
        return s

    @staticmethod
    def compute_q_vectors(aL, daL, d2aL, d3aL, d4aL, mQ=0.85, mL=0.05):
        g = GRAVITY
        e3 = np.array([0.0, 0.0, 1.0])

        Tp = -np.dot(mL, (aL + np.dot(g, e3)))
        norm_Tp = np.linalg.norm(Tp)
        q = np.divide(Tp, norm_Tp)

        dTp = -np.dot(mL, daL)
        dnorm_Tp = np.divide(np.dot(Tp, dTp), norm_Tp)
        dq = np.divide((dTp - np.dot(q, dnorm_Tp)), norm_Tp)

        d2Tp = -np.dot(mL, d2aL)
        d2norm_Tp = np.divide(
            np.dot(dTp, dTp) + np.dot(Tp, d2Tp) - np.dot(dnorm_Tp, dnorm_Tp), norm_Tp
        )
        d2q = np.divide(
            d2Tp - np.dot(dq, dnorm_Tp) - np.dot(q, d2norm_Tp) - np.dot(dq, dnorm_Tp),
            norm_Tp,
        )

        d3Tp = -np.dot(mL, d3aL)
        d3norm_Tp = np.dot(
            2 * np.dot(d2Tp, dTp)
            + np.dot(dTp, d2Tp)
            + np.dot(Tp, d3Tp)
            - 3 * np.dot(dnorm_Tp, d2norm_Tp),
            norm_Tp,
        )
        d3q = np.divide(
            d3Tp
            - np.dot(d2q, dnorm_Tp)
            - np.dot(dq, d2norm_Tp)
            - np.dot(dq, d2norm_Tp)
            - np.dot(q, d3norm_Tp)
            - np.dot(d2q, dnorm_Tp)
            - np.dot(dq, d2norm_Tp)
            - np.dot(d2q, dnorm_Tp),
            norm_Tp,
        )

        d4Tp = np.dot(-mL, d4aL)
        d4norm_Tp = np.divide(
            2 * np.dot(d3Tp, dTp)
            + 2 * np.dot(d2Tp, d2Tp)
            + np.dot(d2Tp, d2Tp)
            + np.dot(dTp, d3Tp)
            + np.dot(dTp, d3Tp)
            + np.dot(Tp, d4Tp)
            - 3 * d2norm_Tp**2
            - 3 * np.dot(dnorm_Tp, d3norm_Tp)
            - np.dot(d3norm_Tp, dnorm_Tp),
            norm_Tp,
        )
        d4q = np.divide(
            d4Tp
            - np.dot(d3q, dnorm_Tp)
            - np.dot(d2q, d2norm_Tp)
            - np.dot(d2q, d2norm_Tp)
            - np.dot(dq, d3norm_Tp)
            - np.dot(d2q, d2norm_Tp)
            - np.dot(dq, d3norm_Tp)
            - np.dot(dq, d3norm_Tp)
            - np.dot(q, d4norm_Tp)
            - np.dot(d3q, dnorm_Tp)
            - np.dot(d2q, d2norm_Tp)
            - np.dot(d2q, d2norm_Tp)
            - np.dot(dq, d3norm_Tp)
            - np.dot(d3q, dnorm_Tp)
            - np.dot(d2q, d2norm_Tp)
            - np.dot(d3q, dnorm_Tp),
            norm_Tp,
        )

        return q, dq, d2q, d3q, d4q, Tp, dTp, d2Tp

    @staticmethod
    def quadrotor_payload(traj, mQ=0.85, mL=0.1, cable_len=1, J=None):
        if J is None:
            J = np.array(
                [
                    [0.005315307431627, 0.000005567447099, 0.000005445855427],
                    [0.000005567447099, 0.004949258422243, 0.000020951458431],
                    [0.000005445855427, 0.000020951458431, 0.009806225007686],
                ]
            )

        xL = traj["x"]
        vL = traj["dx"]
        aL = traj["d2x"]
        daL = traj["d3x"]
        d2aL = traj["d4x"]
        d3aL = traj["d5x"]
        d4aL = traj["d6x"]

        q, dq, d2q, d3q, d4q, Tp, dTp, d2Tp = Flat2State.compute_q_vectors(
            aL, daL, d2aL, d3aL, d4aL, mQ, mL
        )

        xQ = xL - np.dot(cable_len, q)
        vQ = vL - np.dot(cable_len, dq)
        aQ = aL - np.dot(cable_len, d2q)
        daQ = daL - np.dot(cable_len, d3q)
        d2aQ = d2aL - np.dot(cable_len, d4q)

        s = Flat2State.compute_moment(aQ, daQ, d2aQ, mQ, J, Tp, dTp, d2Tp)
        s["xL"] = xL
        s["vL"] = vL
        s["aL"] = aL
        s["xQ"] = xQ
        s["vQ"] = vQ
        s["aQ"] = aQ
        s["q"] = q
        s["dq"] = dq
        s["d2q"] = d2q
        s["omega"] = np.cross(q, dq)
        s["domega"] = np.cross(q, d2q)
        return s
