import numpy as np

from ..core.defaults import GRAVITY


class Gains:
    def __init__(self, kp=np.zeros(3), kd=np.zeros(3), ki=np.zeros(3)):
        self.kp = kp
        self.kd = kd
        self.ki = ki
        return


class Controller:
    def __init__(self):
        self.freq = 500.0
        self._g = GRAVITY
        self._ge3 = np.array([0.0, 0.0, self._g])
        self._e1 = np.array([1.0, 0.0, 0.0])
        self._e2 = np.array([0.0, 1.0, 0.0])
        self._e3 = np.array([0.0, 0.0, 1.0])
        self._gravity = np.array([0.0, 0.0, -self._g])
        return

    def compute(self, *args):
        raise NotImplementedError


class PDController(Controller):
    def __init__(self, **kwargs):
        super().__init__()
        self._gains = Gains()

        if "kp" in kwargs:
            self._gains.kp = kwargs["kp"]
        if "kd" in kwargs:
            self._gains.kd = kwargs["kd"]

        if "setpoint" in kwargs:
            self.setpoint = kwargs["setpoint"]
        else:
            self.setpoint = lambda t: (
                np.array([0.0, 0.0, 1.0]),
                np.zeros(3),
                np.zeros(3),
            )
        return

    def compute(self, *args):
        t = args[0]
        s, ds = args[1][0], args[1][1]
        sd, dsd, d2sd = self.setpoint(t)
        e = s - sd
        de = ds - dsd
        u = -self._gains.kp * e - self._gains.kd * de + d2sd
        return u


#: Riccati residual above which :func:`solve_lqr_gain` refuses the solution, as
#: a fraction of ``\|Q\|``.  A residual this size means ``P`` solves the
#: equation for a ``Q`` that differs from the one posed by a percent or more,
#: so the gain is optimal for a different problem.  Measured legitimate cases
#: sit far below: ``1e-12`` across every shipped M6 configuration, and M5's
#: chain — whose residual grows with ``n`` — reaches only ``1e-5`` at ``n =
#: 30``, well past its zero-order-hold envelope.
ARE_RESIDUAL_TOLERANCE = 1e-2

#: Riccati residual above which :func:`solve_lqr_gain` warns.  Between this and
#: :data:`ARE_RESIDUAL_TOLERANCE` the solution is still *accurate* — the
#: backward error stays at ``1e-16`` throughout — but the problem is badly
#: scaled, and a badly scaled one is where ``\|P\|`` runs to ``1e13`` and the
#: closed loop crawls up to the imaginary axis.
ARE_RESIDUAL_WARNING = 1e-6


def solve_lqr_gain(A, B, Q, R, *, describe: str = "the system"):
    r"""Solve the continuous algebraic Riccati equation and check the answer.

    Returns ``(K, poles)``: the infinite-horizon LQR gain ``K = -R^{-1}B^\top P``
    in the sign convention ``\delta u = K s``, and the eigenvalues of the
    closed loop ``A + BK``, sorted by real part with the most positive first.

    Both LQR controllers in this package (M5's
    :class:`~udaan.control.quadrotor_flexcable.QuadrotorFlexCableController`
    and M6's
    :class:`~udaan.control.multi_quad_hose.MultiQuadHoseController`) accept
    caller-supplied masses, lengths and inertias spanning orders of magnitude,
    and :func:`scipy.linalg.solve_continuous_are` does **not** refuse every
    pair it cannot usefully solve.  On the badly scaled ones it returns,
    without complaint, a ``P`` of norm ``1e13`` whose residual is ``1e6`` times
    ``\|Q\|`` and whose closed loop sits within ``1e-10`` of the imaginary
    axis — a gain of norm ``1e10`` that the calling controller would then
    report on through ``closed_loop_poles`` and ``controllability_margin`` as
    though it were sound.  So the solve is checked rather than trusted.

    Two things are checked, and it is worth being precise about what each one
    means, because the obvious reading of the first is wrong:

    * the **residual** ``A^\top P + PA - PBR^{-1}B^\top P + Q``, against
      ``\|Q\|``.  This is *not* a numerical-accuracy test.  Those ``1e13``
      solutions are accurate: their backward error — the residual normalised
      by ``\|Q\| + 2\|A\|\|P\| + \|PBR^{-1}B^\top P\|`` — is ``1e-16``, the
      same as the healthy cases'.  What the residual measures is **scaling**:
      it is the perturbation to ``Q`` that ``P`` actually solves for, so a
      residual comparable with ``Q`` says the returned gain is optimal for a
      materially different cost than the one posed.  Normalising by ``\|P\|``
      instead would hide exactly the cases this exists to catch, since they
      are bad *because* ``\|P\|`` is enormous.

      Because it is a conditioning signal and not an error, it is graded:
      warned about at :data:`ARE_RESIDUAL_WARNING`, refused only at
      :data:`ARE_RESIDUAL_TOLERANCE`.  A long but perfectly legitimate chain
      drifts into the first band on its own.
    * the **closed loop** being strictly stable, which is the whole property
      the solution is wanted for, and is unambiguous when it fails.

    Args:
        A, B, Q, R: the linearised pair and the two cost matrices.
        describe: what to name in the diagnostics — the caller's plant, its
            parameters, whatever makes the failure actionable.

    Raises:
        ValueError: if no stabilising solution exists, if the returned one
            misses the equation by :data:`ARE_RESIDUAL_TOLERANCE` of ``\|Q\|``,
            or if the closed loop it produces is not strictly stable.
    """
    from scipy.linalg import solve_continuous_are

    from ..utils.logging import get_logger

    try:
        P = solve_continuous_are(A, B, Q, R)
    except (np.linalg.LinAlgError, ValueError) as err:
        # Both failure modes land here: an unstabilisable pair, and a pair
        # whose generalised Schur reordering the solver refuses as too
        # ill-conditioned.
        raise ValueError(
            f"no stabilising solution to the algebraic Riccati equation for {describe}: {err}"
        ) from err

    gain = -np.linalg.solve(R, B.T @ P)
    residual = A.T @ P + P @ A + P @ B @ gain + Q
    relative = float(np.linalg.norm(residual)) / float(np.linalg.norm(Q))
    if relative >= ARE_RESIDUAL_TOLERANCE:
        raise ValueError(
            f"the algebraic Riccati solver returned a solution it did not refuse but "
            f"which does not solve the posed problem for {describe}: residual "
            f"{relative:.3g} of ||Q||, against a tolerance of "
            f"{ARE_RESIDUAL_TOLERANCE:g} (||P|| = {float(np.linalg.norm(P)):.3g}, "
            f"||K|| = {float(np.linalg.norm(gain)):.3g}). The pair is too badly scaled "
            f"to regulate; rescale the masses, the lengths or the cost weights."
        )
    if relative >= ARE_RESIDUAL_WARNING:
        get_logger(__name__).warning(
            "algebraic Riccati residual is %.3g of ||Q|| for %s (||P|| = %.3g, "
            "||K|| = %.3g): the pair is badly scaled and the gain, while it is the "
            "solution of the equation as posed, may be far larger and the closed loop "
            "far slower than the weights suggest.",
            relative,
            describe,
            float(np.linalg.norm(P)),
            float(np.linalg.norm(gain)),
        )

    poles = np.linalg.eigvals(A + B @ gain)
    poles = poles[np.argsort(-poles.real)]
    if not poles[0].real < 0.0:
        raise ValueError(
            f"the algebraic Riccati solution for {describe} does not stabilise: the "
            f"closed loop keeps an eigenvalue at {poles[0]:.6g}. The pair is not "
            f"stabilisable, whatever the solver returned."
        )
    return gain, poles


from . import multi_quad_cspayload as multi_quad_cspayload  # noqa: E402
from . import multi_quad_hose as multi_quad_hose  # noqa: E402
from . import multi_quad_rigidbody as multi_quad_rigidbody  # noqa: E402
from . import quadrotor as quadrotor  # noqa: E402
from . import quadrotor_cspayload as quadrotor_cspayload  # noqa: E402
from . import quadrotor_flexcable as quadrotor_flexcable  # noqa: E402

__all__ = [
    "ARE_RESIDUAL_TOLERANCE",
    "ARE_RESIDUAL_WARNING",
    "Gains",
    "Controller",
    "PDController",
    "solve_lqr_gain",
    "multi_quad_cspayload",
    "multi_quad_hose",
    "multi_quad_rigidbody",
    "quadrotor",
    "quadrotor_cspayload",
    "quadrotor_flexcable",
]
