"""Pin the ``data.ctrl`` layout of every generated multi-agent MJCF.

Each of these models resolves *bodies* by name but writes ``data.ctrl``
**positionally** — ``data.ctrl[:] = wrench.reshape(-1)`` for the M3 model,
``u[4*i : 4*i+4] = [f, *M]`` for the two fleets. Nothing in the generator
guaranteed that slot ``4*i + k`` drives quadrotor ``i``: the emitted ``<motor>``
elements carried no ``name``, so the mapping was unassertable and a permuted
actuator block was invisible to the XML-shape tests (``test_actuators_live_in
_one_block`` counts blocks and motors, and counts survive any permutation).

Actuators are named now, so these tests check both halves of the mapping:

1. the actuator *names*, in MuJoCo id order, are exactly the order the model
   writes ``data.ctrl`` in; and
2. each actuator's transmission really lands on the body of the agent its slot
   belongs to, and carries the gear column (thrust / Mx / My / Mz) that slot
   means.

(2) is what makes the test independent of the naming convention itself.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

pytestmark = pytest.mark.mujoco

#: Per-agent actuator suffixes, in the order each model writes them into
#: ``data.ctrl``: ``[thrust, Mx, My, Mz]``.
_SUFFIXES = ("thrust", "Mx", "My", "Mz")

#: Gear column each suffix must drive, in the site frame: force along +z, then
#: torque about x, y, z.
_GEAR = {
    "thrust": np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    "Mx": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
    "My": np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    "Mz": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
}


def assert_ctrl_layout(model, quad_names):
    """Assert ``data.ctrl`` slot ``4*i + k`` drives ``quad_names[i]``'s ``k``-th axis."""
    assert model.nu == 4 * len(quad_names)

    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, k) for k in range(model.nu)]
    expected = [f"{q}_{s}" for q in quad_names for s in _SUFFIXES]
    assert names == expected

    for i, quad in enumerate(quad_names):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, quad)
        assert body_id >= 0, f"no body named {quad!r}"
        for k, suffix in enumerate(_SUFFIXES):
            slot = 4 * i + k
            # The transmission is a site; the site must sit on this quad's body.
            assert model.actuator_trntype[slot] == mujoco.mjtTrn.mjTRN_SITE
            site_id = int(model.actuator_trnid[slot, 0])
            assert model.site_bodyid[site_id] == body_id, (
                f"ctrl slot {slot} drives body "
                f"{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.site_bodyid[site_id])}"
                f", expected {quad}"
            )
            np.testing.assert_allclose(model.actuator_gear[slot], _GEAR[suffix], atol=1e-12)


@pytest.mark.parametrize("nQ", [1, 2, 3, 4])
def test_multi_quad_cspayload_ctrl_layout(nQ):
    """``MultiQuadCsPayloadMujoco.step`` writes ``wrench.reshape(-1)`` — pin it."""
    from udaan.models.multi_quad_cspayload import MultiQuadCsPayloadMujoco

    mdl = MultiQuadCsPayloadMujoco(render=False, num_quadrotors=nQ)
    assert_ctrl_layout(mdl._mjMdl.model, [f"quad{i}" for i in range(nQ)])
    # ...and that the body ids the model resolved by name agree with it.
    model = mdl._mjMdl.model
    for i in range(nQ):
        site_id = int(model.actuator_trnid[4 * i, 0])
        assert model.site_bodyid[site_id] == mdl._quad_body_id[i]


@pytest.mark.parametrize("nQ", [2, 3])
def test_quadrotor_fleet_ctrl_layout(nQ):
    """``QuadrotorFleet.simulate`` fills ``u[4*i : 4*i+4]`` — pin it."""
    from udaan.models.mujoco import QuadrotorFleet

    fleet = QuadrotorFleet(num_quadrotors=nQ, render=False)
    assert_ctrl_layout(fleet._mjMdl.model, [f"quad{i}" for i in range(nQ)])


@pytest.mark.parametrize("nQ", [2, 3])
def test_quadrotor_cspayload_fleet_ctrl_layout(nQ):
    """``QuadrotorCsPayloadFleet.simulate`` fills ``u[4*i : 4*i+4]`` — pin it."""
    from udaan.models.mujoco import QuadrotorCsPayloadFleet

    fleet = QuadrotorCsPayloadFleet(num_agents=nQ, render=False, cable_links=3)
    assert_ctrl_layout(fleet._mjMdl.model, [f"agent{i}_quad" for i in range(nQ)])
