r"""Tests for the M6 MuJoCo backend: ``nQ`` quadrotors carrying one flexible hose.

What these check that the analytical tests cannot: the generated MJCF is the
machine the analytical model describes.  Three things make that a stronger
claim here than for M5, and each has a test:

* **The mass is at the vertices in both.**  M5's MJCF spreads each link's mass
  along a cylinder and its backend lumps it onto the distal joint — first order
  in the link length.  Here the cylinders are massless and each link body
  carries a sphere at its distal end, so ``inertiafromgeom`` puts the derived
  centre of mass exactly where the analytical model lumps it and
  ``_read_params_from_mjcf`` has no approximation to make.
* **Both backends spawn the same shape**, because both read
  :func:`udaan.utils.hose.hose_spawn_directions`.
* **The gauge degrees of freedom MuJoCo carries and the analytical model does
  not — each link's spin about its own axis, and the rotation of each massive
  vertex — are inert**, because every force in this system acts through a
  body's own centre of mass.  ``test_the_gauge_degrees_of_freedom_stay_inert``
  is what turns that argument into a measurement.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mujoco
import numpy as np
import pytest

from tests.conftest import TOL_ANALYTIC, TOL_EXACT, TOL_INTEGRATION, TOL_TIGHT
from udaan.core.defaults import DEFAULT_MJCF_TIMESTEP, GRAVITY
from udaan.manif import S2, SO3, TS2, TSO3, rodrigues_expm
from udaan.models.multi_quad_hose import MultiQuadHoseBase, MultiQuadHoseMujoco
from udaan.utils.assets.cache import generated_mjcf_dir
from udaan.utils.hose import hose_spawn_directions

pytestmark = pytest.mark.mujoco

#: Generated assets live in the user cache, not inside the package.
_GENERATED_DIR = generated_mjcf_dir()

#: ``(num_links, attachments)`` pairs. Each generates its own cached MJCF, so
#: the list is deliberately shorter than the analytical one.
PATTERNS = [(2, (0, 2)), (4, (0, 4)), (4, (0, 2, 4)), (4, (1, 3))]

LIGHT = {"masses": 0.4, "hose_mass": 0.4}


def build(num_links=4, attachments=(0, 4), **kwargs):
    return MultiQuadHoseMujoco(num_links=num_links, attachments=attachments, **{**LIGHT, **kwargs})


def random_state(model, rng, rate_scale=1.0):
    directions, rates = [], []
    for _ in range(model.num_links):
        q = rng.normal(size=3)
        q /= np.linalg.norm(q)
        omega = rng.normal(size=3)
        omega -= float(omega @ q) * q
        directions.append(q)
        rates.append(rate_scale * omega)
    orientations = [
        np.asarray(rodrigues_expm(rng.normal(size=3) * 0.4)) for _ in range(model.num_quadrotors)
    ]
    body_rates = [rate_scale * rng.normal(size=3) for _ in range(model.num_quadrotors)]
    return {
        "link_attitudes": directions,
        "link_angular_velocities": rates,
        "orientations": orientations,
        "angular_velocities": body_rates,
    }


@pytest.fixture(scope="module")
def mdl():
    return build()


# ── Setup ─────────────────────────────────────────────────────────────


class TestMujocoSetup:
    """The generated asset, and what the backend reads back off it."""

    @pytest.mark.parametrize("n,attachments", PATTERNS)
    def test_instantiate(self, n, attachments):
        model = build(n, attachments)
        assert model.num_links == n
        assert model.attachments == attachments
        assert model.num_quadrotors == len(attachments)
        assert model._mjMdl.model.nu == 4 * len(attachments)

    @pytest.mark.parametrize("n,attachments", PATTERNS)
    def test_link_lengths_come_from_the_file(self, n, attachments):
        model = build(n, attachments, hose_length=1.6)
        assert np.allclose(model.link_lengths, 1.6 / n, atol=TOL_TIGHT)
        assert model.hose_length == pytest.approx(1.6, abs=TOL_TIGHT)

    def test_masses_come_from_the_file(self, mdl):
        mj = mdl._mjMdl.model
        # Every hose vertex mass is a sphere on its own body — no lumping.
        assert np.allclose(mdl.vertex_masses, 0.4 / (mdl.num_links + 1))
        assert mdl.vertex_masses.sum() == pytest.approx(0.4)
        assert np.allclose(mdl.masses, [mj.body_mass[i] for i in mdl._quad_body_id])
        assert mdl.total_mass == pytest.approx(sum(mj.body_mass))

    def test_the_link_cylinders_are_massless(self, mdl):
        """The hose's mass is at the vertices, so the tubes carry none of it.

        Each link body holds two geoms — a cylinder and the vertex sphere at its
        distal end — and its total mass being *exactly* the sphere's is what says
        the cylinder contributed nothing.
        """
        mj = mdl._mjMdl.model
        for i in range(mdl.num_links):
            body_id = int(mdl._link_body_id[i])
            assert int(mj.body_geomnum[body_id]) == 2
            assert float(mj.body_mass[body_id]) == pytest.approx(
                float(mdl.vertex_masses[i + 1]), abs=TOL_EXACT
            )

    def test_the_vertex_mass_sits_at_the_links_distal_end(self, mdl):
        """``inertiafromgeom`` therefore puts each body's COM exactly there."""
        mj = mdl._mjMdl.model
        for i in range(mdl.num_links):
            body_id = int(mdl._link_body_id[i])
            half = 0.5 * float(mdl.link_lengths[i])
            assert np.allclose(mj.body_ipos[body_id], [0.0, 0.0, -half], atol=TOL_ANALYTIC)

    def test_inertias_come_from_the_file(self, mdl):
        """Each airframe's tensor in its own body frame, read from the MJCF file.

        MuJoCo stores principal moments in the principal frame ``body_iquat``,
        and for this airframe that frame is rotated (the moments come out in
        the order ``J_z, J_y, J_x``), so the diagonal of ``body_inertia`` is
        not the body-frame tensor. The reference is computed from a fresh load
        of the file, not from the live model the setter writes into.
        """
        fresh = mujoco.MjModel.from_xml_path(mdl._mjMdl.full_path)
        for k, body_id in enumerate(mdl._quad_body_id):
            flat = np.zeros(9)
            mujoco.mju_quat2Mat(flat, fresh.body_iquat[body_id])
            rotation = flat.reshape(3, 3)
            expected = rotation @ np.diag(fresh.body_inertia[body_id]) @ rotation.T
            np.testing.assert_allclose(mdl.inertias[k], expected, atol=TOL_TIGHT)
            # A flat airframe yaws hardest: J_z is its largest moment.
            assert mdl.inertias[k][2, 2] > mdl.inertias[k][0, 0]
            assert mdl.inertias[k][2, 2] > mdl.inertias[k][1, 1]

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_a_body_moment_spins_each_airframe_up_at_the_model_inertia(self, axis):
        """From rest, one step of a body moment gives ``Ω ≈ J⁻¹ M dt``.

        Each airframe hangs on a ball joint at its own centre of mass, so the
        hose pulls through that point and adds no moment. ``J`` is the model's
        own :attr:`inertias`, so a model that reads a different tensor from the
        one MuJoCo integrates predicts the wrong rate.
        """
        model = build(4, (0, 2, 4), input="wrench")
        for k in range(model.num_quadrotors):
            model.reset(root_position=np.array([0.0, 0.0, 2.0]))
            moment = np.zeros(3)
            moment[axis] = 2e-3
            wrench = np.zeros((model.num_quadrotors, 4))
            wrench[k, 1:] = moment
            model.step(wrench)
            predicted = np.linalg.solve(model.inertias[k], moment) * model.dt
            measured = np.asarray(model.state.angular_velocities[k])
            assert measured[axis] == pytest.approx(predicted[axis], rel=1e-2)

    def test_timestep_divides_dt(self, mdl):
        assert mdl._mjDt == pytest.approx(DEFAULT_MJCF_TIMESTEP)
        assert mdl._step_iter * mdl._mjDt == pytest.approx(mdl.dt)
        assert mdl.dt == pytest.approx(1 / 200.0)

    def test_no_damping_anywhere(self, mdl):
        """The analytical model has no drag term, so neither does the asset."""
        assert np.allclose(mdl._mjMdl.model.dof_damping, 0.0)
        assert np.allclose(mdl._mjMdl.model.jnt_stiffness, 0.0)

    def test_contacts_are_disabled(self, mdl):
        """Deliberate: the geometry allows overlap and the model has no contact term."""
        disabled = int(mdl._mjMdl.model.opt.disableflags)
        assert disabled & int(mujoco.mjtDisableBit.mjDSBL_CONTACT)

    def test_every_quadrotor_hangs_on_a_ball_joint(self, mdl):
        """A ball joint at the airframe's own COM transmits force but no torque."""
        mj = mdl._mjMdl.model
        for k, body_id in enumerate(mdl._quad_body_id):
            jnt = int(mj.body_jntadr[body_id])
            assert int(mj.jnt_type[jnt]) == int(mujoco.mjtJoint.mjJNT_BALL)
            assert np.allclose(mj.jnt_pos[jnt], 0.0)
            assert np.allclose(mj.body_ipos[body_id], 0.0, atol=1e-12)
            # ... and it hangs off the body owning its vertex.
            vertex = mdl.attachments[k]
            parent = mdl._root_body_id if vertex == 0 else int(mdl._link_body_id[vertex - 1])
            assert int(mj.body_parentid[body_id]) == parent

    @pytest.mark.parametrize("n,attachments", PATTERNS)
    def test_the_spawn_matches_the_analytical_backend(self, n, attachments):
        model = build(n, attachments)
        expected = hose_spawn_directions(model.link_lengths, attachments, model.sag_ratio)
        measured = np.array([np.asarray(q) for q in model.state.link_attitudes])
        assert np.allclose(measured, expected, atol=TOL_ANALYTIC)
        # ... and it is taut: every link at exactly its own length.
        vertices = model.vertex_positions()
        spans = np.linalg.norm(np.diff(vertices, axis=0), axis=1)
        assert np.allclose(spans, model.link_lengths, atol=TOL_ANALYTIC)

    def test_state_types(self, mdl):
        assert all(isinstance(q, S2) for q in mdl.state.link_attitudes)
        assert all(isinstance(w, TS2) for w in mdl.state.link_angular_velocities)
        assert all(isinstance(R, SO3) for R in mdl.state.orientations)
        assert all(isinstance(O, TSO3) for O in mdl.state.angular_velocities)

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"vertex_masses": [0.1] * 5}, "vertex masses and link lengths"),
            ({"link_lengths": [0.5] * 4}, "vertex masses and link lengths"),
            ({"inertias": np.eye(3) * 0.01}, "derives inertias"),
        ],
    )
    def test_overwritten_kwargs_are_rejected(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            MultiQuadHoseMujoco(num_links=4, attachments=(0, 4), **kwargs)

    def test_non_uniform_masses_are_rejected_at_construction(self):
        with pytest.raises(ValueError, match="uniform masses"):
            MultiQuadHoseMujoco(num_links=4, attachments=(0, 4), masses=[0.4, 0.6])

    def test_the_asset_name_carries_every_shaping_parameter(self):
        """A changed parameter must never reuse a cached file."""
        base = MultiQuadHoseMujoco._ensure_xml(4, (0, 4), 2.0, 0.4, 0.4, 0.8, 2.0)
        variants = [
            MultiQuadHoseMujoco._ensure_xml(5, (0, 5), 2.0, 0.4, 0.4, 0.8, 2.0),
            MultiQuadHoseMujoco._ensure_xml(4, (0, 2, 4), 2.0, 0.4, 0.4, 0.8, 2.0),
            MultiQuadHoseMujoco._ensure_xml(4, (0, 4), 2.5, 0.4, 0.4, 0.8, 2.0),
            MultiQuadHoseMujoco._ensure_xml(4, (0, 4), 2.0, 0.6, 0.4, 0.8, 2.0),
            MultiQuadHoseMujoco._ensure_xml(4, (0, 4), 2.0, 0.4, 0.5, 0.8, 2.0),
            MultiQuadHoseMujoco._ensure_xml(4, (0, 4), 2.0, 0.4, 0.4, 0.7, 2.0),
            MultiQuadHoseMujoco._ensure_xml(4, (0, 4), 2.0, 0.4, 0.4, 0.8, 2.5),
        ]
        assert len(set(variants)) == len(variants)
        assert base not in variants
        for name in [base, *variants]:
            assert Path(name).exists()
            assert Path(name).parent == _GENERATED_DIR
        assert list(_GENERATED_DIR.glob("multi_quad_hose_*.xml"))

    def test_a_changed_parameter_reaches_the_model(self):
        light = build(hose_mass=0.3)
        heavy = build(hose_mass=0.9)
        assert light.vertex_masses.sum() == pytest.approx(0.3)
        assert heavy.vertex_masses.sum() == pytest.approx(0.9)
        assert light._mjMdl.full_path != heavy._mjMdl.full_path

    def test_actuator_order_is_vehicle_order(self, mdl):
        """``data.ctrl`` is written positionally, so the layout has to be pinned."""
        mj = mdl._mjMdl.model
        names = [mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(mj.nu)]
        expected = [
            f"quad{k}_{suffix}"
            for k in range(mdl.num_quadrotors)
            for suffix in ("thrust", "Mx", "My", "Mz")
        ]
        assert names == expected


# ── Property setters ──────────────────────────────────────────────────


class TestPropertySetters:
    """Setters must write into MuJoCo, not just into the analytical parameters."""

    def test_masses_setter_writes_into_mujoco(self):
        model = build()
        model.masses = [0.55, 0.65]
        mj = model._mjMdl.model
        assert [mj.body_mass[i] for i in model._quad_body_id] == pytest.approx([0.55, 0.65])

    def test_vertex_masses_setter_writes_into_mujoco(self):
        model = build()
        model.vertex_masses = 0.07
        mj = model._mjMdl.model
        assert mj.body_mass[model._root_body_id] == pytest.approx(0.07)
        assert [mj.body_mass[i] for i in model._link_body_id] == pytest.approx([0.07] * 4)

    def test_vertex_masses_setter_also_rewrites_the_inertia(self):
        """A vertex is one sphere: its inertia is not independent of its mass.

        MuJoCo derives ``body_inertia`` once at compile time, so writing
        ``body_mass`` alone would leave every vertex spinning with the inertia
        of its previous mass.
        """
        model = build()
        mj = model._mjMdl.model
        bodies = [model._root_body_id, *(int(b) for b in model._link_body_id)]
        before = np.array([mj.body_inertia[b].copy() for b in bodies])
        original = float(model.vertex_masses[0])

        model.vertex_masses = 0.07
        after = np.array([mj.body_inertia[b].copy() for b in bodies])
        # Same geometry, so the inertia must track the mass exactly.
        assert np.allclose(after, before * (0.07 / original), rtol=0.0, atol=TOL_EXACT)
        # And it must equal 2/5 m r² for the radius the file actually declares.
        for v, body_id in enumerate(bodies):
            radius = float(model._vertex_radius[v])
            assert np.allclose(mj.body_inertia[body_id], 0.4 * 0.07 * radius**2, atol=TOL_EXACT)

        # Round-tripping the original masses restores the compiled values.
        model.vertex_masses = original
        restored = np.array([mj.body_inertia[b].copy() for b in bodies])
        assert np.allclose(restored, before, rtol=0.0, atol=TOL_EXACT)

    def test_inertias_setter_writes_into_mujoco(self):
        model = build()
        model.inertias = np.array([0.011, 0.012, 0.013])
        mj = model._mjMdl.model
        for body_id in model._quad_body_id:
            assert np.allclose(mj.body_inertia[body_id], [0.011, 0.012, 0.013])

    def test_inertias_setter_writes_the_body_frame_tensor(self):
        """An assigned tensor is the one MuJoCo integrates, off-diagonals included."""
        model = build()
        tensor = np.array([[6e-3, 4e-4, -2e-4], [4e-4, 7e-3, 3e-4], [-2e-4, 3e-4, 1.1e-2]])
        model.inertias = tensor
        mj = model._mjMdl.model
        for body_id in model._quad_body_id:
            flat = np.zeros(9)
            mujoco.mju_quat2Mat(flat, mj.body_iquat[body_id])
            rotation = flat.reshape(3, 3)
            np.testing.assert_allclose(
                rotation @ np.diag(mj.body_inertia[body_id]) @ rotation.T, tensor, atol=TOL_TIGHT
            )


# ── State measurement ─────────────────────────────────────────────────


class TestPerLinkStateIsMeasured:
    """Nothing is finite-differenced: every link and vehicle is read."""

    @pytest.mark.parametrize("n,attachments", PATTERNS)
    def test_reset_round_trips_the_whole_chain(self, n, attachments):
        model = build(n, attachments)
        rng = np.random.default_rng(200 + n)
        state = random_state(model, rng, 0.5)
        root = np.array([0.3, -0.4, 3.0])
        velocity = np.array([0.2, -0.1, 0.05])
        model.reset(root_position=root, root_velocity=velocity, **state)

        assert np.allclose(model.state.root_position, root, atol=TOL_ANALYTIC)
        assert np.allclose(model.state.root_velocity, velocity, atol=TOL_ANALYTIC)
        for i in range(n):
            assert np.allclose(
                np.asarray(model.state.link_attitudes[i]),
                state["link_attitudes"][i],
                atol=TOL_ANALYTIC,
            )
            assert np.allclose(
                np.asarray(model.state.link_angular_velocities[i]),
                state["link_angular_velocities"][i],
                atol=TOL_ANALYTIC,
            )
        for k in range(model.num_quadrotors):
            assert np.allclose(
                np.asarray(model.state.orientations[k]),
                state["orientations"][k],
                atol=TOL_ANALYTIC,
            )
            assert np.allclose(
                np.asarray(model.state.angular_velocities[k]),
                state["angular_velocities"][k],
                atol=TOL_ANALYTIC,
            )

    @pytest.mark.parametrize("n,attachments", PATTERNS)
    def test_measured_state_agrees_with_the_chain_geometry(self, n, attachments):
        """Vehicle positions must be the vertex positions the chain implies."""
        model = build(n, attachments)
        rng = np.random.default_rng(210 + n)
        model.reset(root_position=np.array([0.0, 0.0, 3.0]), **random_state(model, rng, 0.5))
        vertices = model.vertex_positions()
        velocities = model.vertex_velocities()
        for k, vertex in enumerate(attachments):
            assert np.allclose(model.state.positions[k], vertices[vertex], atol=TOL_ANALYTIC)
            assert np.allclose(model.state.velocities[k], velocities[vertex], atol=TOL_ANALYTIC)

    @pytest.mark.parametrize("n,attachments", PATTERNS)
    def test_state_after_a_step_is_the_current_configuration(self, n, attachments):
        """After ``step()`` every body is read at the configuration MuJoCo holds now.

        ``mj_step`` leaves ``xpos``/``xmat``/``cvel`` at the configuration the
        step started from, while the root is read from ``qpos``/``qvel``. The
        reference is a forward pass on a copy of the current data.
        """
        model = build(n, attachments, input="wrench")
        rng = np.random.default_rng(300 + n)
        model.reset(root_position=np.array([0.0, 0.0, 3.0]), **random_state(model, rng, 0.5))
        wrench = np.zeros((model.num_quadrotors, 4))
        wrench[:, 0] = model.total_mass * GRAVITY / model.num_quadrotors
        for _ in range(20):
            model.step(wrench)
        mj = model._mjMdl.model
        fresh = mujoco.MjData(mj)
        fresh.qpos[:] = model._mjMdl.data.qpos
        fresh.qvel[:] = model._mjMdl.data.qvel
        mujoco.mj_forward(mj, fresh)
        for i in range(n):
            q_i = -np.asarray(fresh.xmat[model._link_body_id[i]]).reshape(3, 3)[:, 2]
            np.testing.assert_allclose(
                np.asarray(model.state.link_attitudes[i]), q_i, atol=TOL_TIGHT
            )
        for k, body_id in enumerate(model._quad_body_id):
            np.testing.assert_allclose(
                np.asarray(model.state.positions[k]), fresh.xpos[body_id], atol=TOL_TIGHT
            )
            np.testing.assert_allclose(
                np.asarray(model.state.orientations[k]),
                np.asarray(fresh.xmat[body_id]).reshape(3, 3),
                atol=TOL_TIGHT,
            )
        # The chain closes on the measured state after stepping, not only at reset.
        vertices = model.vertex_positions()
        for k, vertex in enumerate(model.attachments):
            np.testing.assert_allclose(
                np.asarray(model.state.positions[k]), vertices[vertex], atol=TOL_ANALYTIC
            )

    def test_velocities_are_not_finite_differenced(self, mdl):
        """A reset state has an exact velocity even before a single step."""
        mdl.reset(root_position=np.array([0.0, 0.0, 3.0]), root_velocity=np.array([1.0, 0.0, 0.0]))
        assert np.allclose(mdl.state.root_velocity, [1.0, 0.0, 0.0], atol=TOL_ANALYTIC)
        for velocity in mdl.state.velocities:
            assert np.allclose(velocity, [1.0, 0.0, 0.0], atol=TOL_ANALYTIC)

    def test_link_directions_stay_unit_norm(self, mdl):
        mdl.reset(root_position=np.array([0.0, 0.0, 3.0]))
        for _ in range(200):
            mdl.step(mdl.compute_default_input())
        for q in mdl.state.link_attitudes:
            assert abs(np.linalg.norm(np.asarray(q)) - 1.0) < TOL_TIGHT

    def test_link_rates_stay_exactly_tangent(self, mdl):
        """Projected on read, so the residual is round-off rather than ``O(h)``."""
        mdl.reset(root_position=np.array([0.0, 0.0, 3.0]))
        for _ in range(200):
            mdl.step(mdl.compute_default_input())
        for i in range(mdl.num_links):
            q = np.asarray(mdl.state.link_attitudes[i])
            omega = np.asarray(mdl.state.link_angular_velocities[i])
            assert abs(float(omega @ q)) < TOL_TIGHT

    def test_the_gauge_degrees_of_freedom_stay_inert(self):
        """MuJoCo carries DOFs the analytical model factors out; they must not drive it.

        Every force in this system acts through a body's own centre of mass —
        gravity, the thrust at a vehicle's own origin, and each ball joint's
        constraint force at an anchor the generator places on the parent's own
        symmetry axis. So no torque ever reaches vertex 0's rotation or a link's
        spin about its own axis.

        Vertex 0's rotation is pinned to round-off, and so is link 0's spin,
        whose parent is that vertex.

        Further down the chain ``ω_i · q_i`` is *not* zero and should not be:
        link ``i``'s body angular velocity inherits its parent's, and the
        parent's tilt rate has a component along ``q_i`` whenever the two links
        are not parallel. That component is real, and it is precisely the gauge
        direction the analytical model discards — ``q_i = -R_i e_3`` is
        invariant under rotation about ``q_i``, and
        :meth:`_query_latest_state` projects it out. So what this asserts for
        ``i > 0`` is that the *stored* ``ω_i`` is exactly tangent whatever the
        body frame is doing.
        """
        model = build()
        rng = np.random.default_rng(220)
        model.reset(root_position=np.array([0.0, 0.0, 3.0]), **random_state(model, rng, 0.5))
        data = model._mjMdl.data
        for _ in range(400):
            model.step(model.compute_default_input())
        # The root is a point mass: its rotational DOFs are pinned exactly.
        adr = model._root_qveladr
        assert np.max(np.abs(data.qvel[adr + 3 : adr + 6])) < TOL_INTEGRATION
        assert np.allclose(
            data.qpos[model._root_qposadr + 3 : model._root_qposadr + 7],
            [1.0, 0.0, 0.0, 0.0],
            atol=TOL_INTEGRATION,
        )
        # Link 0 hangs off the pinned root, so its own spin has nothing to
        # inherit and stays at zero.
        q0 = np.asarray(model.state.link_attitudes[0])
        omega_world, _ = model._body_world_velocity(int(model._link_body_id[0]))
        assert abs(float(omega_world @ q0)) < TOL_INTEGRATION
        # Everywhere: the state the model actually integrates is exactly tangent.
        for i in range(model.num_links):
            q = np.asarray(model.state.link_attitudes[i])
            assert abs(float(np.asarray(model.state.link_angular_velocities[i]) @ q)) < TOL_TIGHT


# ── Reset ─────────────────────────────────────────────────────────────


class TestReset:
    def test_root_position_moves_the_whole_model(self, mdl):
        mdl.reset(root_position=np.array([0.0, 0.0, 3.0]))
        before = mdl.vertex_positions()
        mdl.reset(root_position=np.array([1.0, -2.0, 4.0]))
        after = mdl.vertex_positions()
        assert np.allclose(after - before, [1.0, -2.0, 1.0], atol=TOL_ANALYTIC)

    def test_unknown_kwarg_is_rejected(self, mdl):
        with pytest.raises(ValueError, match="cannot honour"):
            mdl.reset(payload_position=np.zeros(3))

    def test_reset_clears_time(self, mdl):
        mdl.reset(root_position=np.array([0.0, 0.0, 3.0]))
        for _ in range(10):
            mdl.step(np.zeros((mdl.num_quadrotors, 3)))
        assert mdl.t > 0.0
        mdl.reset(root_position=np.array([0.0, 0.0, 3.0]))
        assert mdl.t == 0.0

    def test_reset_repairs_off_manifold_input_like_the_base(self):
        """Both backends land on the same state from the same bad input."""
        rng = np.random.default_rng(230)
        directions = [rng.normal(size=3) * rng.uniform(0.2, 4.0) for _ in range(4)]
        rates = [rng.normal(size=3) for _ in range(4)]
        kwargs = {
            "root_position": np.array([0.0, 0.0, 3.0]),
            "link_attitudes": directions,
            "link_angular_velocities": rates,
        }
        mujoco_model = build()
        base = MultiQuadHoseBase(num_links=4, attachments=(0, 4), **LIGHT)
        mujoco_model.reset(**kwargs)
        base.reset(**kwargs)
        for i in range(4):
            assert np.allclose(
                np.asarray(mujoco_model.state.link_attitudes[i]),
                np.asarray(base.state.link_attitudes[i]),
                atol=TOL_ANALYTIC,
            )
            assert np.allclose(
                np.asarray(mujoco_model.state.link_angular_velocities[i]),
                np.asarray(base.state.link_angular_velocities[i]),
                atol=TOL_ANALYTIC,
            )

    def test_a_zero_link_direction_is_rejected(self, mdl):
        directions = [np.array([0.0, 0.0, -1.0])] * mdl.num_links
        directions[1] = np.zeros(3)
        with pytest.raises(ValueError, match="does not define a direction"):
            mdl.reset(link_attitudes=directions)


# ── Dynamics ──────────────────────────────────────────────────────────


class TestDynamics:
    @pytest.mark.parametrize("n,attachments", PATTERNS)
    def test_hover_trim_holds(self, n, attachments):
        """The analytically derived equilibrium holds in MuJoCo too."""
        model = build(n, attachments)
        model._max_thrust = 1e3
        trim = model.hover_trim()
        model.reset(
            root_position=np.array([0.0, 0.0, 3.0]),
            link_attitudes=list(trim["link_attitudes"]),
            orientations=list(trim["orientations"]),
        )
        model._input_type = model.INPUT_TYPE.WRENCH
        root = model.state.root_position.copy()
        for _ in range(200):
            model.step(trim["wrench"])
        assert np.max(np.abs(model.state.root_position - root)) < TOL_INTEGRATION

    def test_zero_input_is_free_fall(self, mdl):
        """MuJoCo's semi-implicit step, so ``-½g(t² + h t)`` rather than ``-½g t²``."""
        mdl.reset(root_position=np.array([0.0, 0.0, 5.0]))
        mdl._input_type = mdl.INPUT_TYPE.WRENCH
        directions = np.array([np.asarray(q) for q in mdl.state.link_attitudes])
        for _ in range(100):
            mdl.step(np.zeros((mdl.num_quadrotors, 4)))
        expected = 5.0 - 0.5 * GRAVITY * (mdl.t**2 + mdl._mjDt * mdl.t)
        assert mdl.state.root_position[2] == pytest.approx(expected, abs=TOL_INTEGRATION)
        # Free fall is rigid: nothing in the shape changes.
        after = np.array([np.asarray(q) for q in mdl.state.link_attitudes])
        assert np.allclose(after, directions, atol=TOL_INTEGRATION)

    def test_station_keeping_settles(self):
        model = build()
        model.simulate(tf=5.0, root_position=np.array([0.0, 0.0, 2.0]))
        for k in range(model.num_quadrotors):
            assert np.max(np.abs(model.state.positions[k] - model._station_targets[k])) < 0.3
            assert np.max(np.abs(model.state.velocities[k])) < 0.1


class TestBackendAgreement:
    """The analytical model and MuJoCo are the same machine."""

    @staticmethod
    def _pair(n, attachments):
        return (
            build(n, attachments),
            MultiQuadHoseBase(num_links=n, attachments=attachments, **LIGHT),
        )

    @pytest.mark.parametrize("n,attachments", PATTERNS)
    def test_initial_conditions_match_exactly(self, n, attachments):
        mujoco_model, base = self._pair(n, attachments)
        rng = np.random.default_rng(240 + n)
        kwargs = {
            "root_position": np.array([0.0, 0.0, 3.0]),
            "root_velocity": np.array([0.1, 0.2, -0.1]),
            **random_state(mujoco_model, rng, 0.5),
        }
        mujoco_model.reset(**kwargs)
        base.reset(**kwargs)
        assert np.allclose(
            mujoco_model.state.root_position, base.state.root_position, atol=TOL_ANALYTIC
        )
        assert np.allclose(
            mujoco_model.state.root_velocity, base.state.root_velocity, atol=TOL_ANALYTIC
        )
        for i in range(n):
            assert np.allclose(
                np.asarray(mujoco_model.state.link_attitudes[i]),
                np.asarray(base.state.link_attitudes[i]),
                atol=TOL_ANALYTIC,
            )
            assert np.allclose(
                np.asarray(mujoco_model.state.link_angular_velocities[i]),
                np.asarray(base.state.link_angular_velocities[i]),
                atol=TOL_ANALYTIC,
            )
        for k in range(len(attachments)):
            assert np.allclose(
                mujoco_model.state.positions[k], base.state.positions[k], atol=TOL_ANALYTIC
            )
            assert np.allclose(
                mujoco_model.state.velocities[k], base.state.velocities[k], atol=TOL_ANALYTIC
            )

    def test_the_masses_agree(self):
        mujoco_model, base = self._pair(4, (0, 4))
        assert np.allclose(mujoco_model.vertex_total_masses, base.vertex_total_masses)
        assert mujoco_model.total_mass == pytest.approx(base.total_mass)

    @pytest.mark.slow
    @pytest.mark.parametrize("n,attachments", [(2, (0, 2)), (4, (0, 4))])
    def test_free_swing_agrees(self, n, attachments):
        """Two integrators on the same equations, so an ``O(h)`` gap and no more.

        Both are held up by the same open-loop trim wrench and the hose is left
        to swing from the taut spawn, which is *not* an equilibrium — so this
        exercises the coupled dynamics rather than a rigid translation.
        """
        mujoco_model, base = self._pair(n, attachments)
        kwargs = {"root_position": np.array([0.0, 0.0, 3.0])}
        mujoco_model.reset(**kwargs)
        base.reset(**kwargs)
        mujoco_model._input_type = mujoco_model.INPUT_TYPE.WRENCH
        base._input_type = base.INPUT_TYPE.WRENCH
        nQ = len(attachments)
        wrench = np.zeros((nQ, 4))
        wrench[:, 0] = base.total_mass * GRAVITY / nQ

        peak_position = 0.0
        peak_direction = 0.0
        while base.t < 2.0:
            mujoco_model.step(wrench)
            base.step(wrench)
            peak_position = max(
                peak_position,
                float(np.max(np.abs(mujoco_model.state.root_position - base.state.root_position))),
            )
            peak_direction = max(
                peak_direction,
                max(
                    float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
                    for a, b in zip(
                        mujoco_model.state.link_attitudes, base.state.link_attitudes, strict=True
                    )
                ),
            )
        assert peak_position < 5e-2
        assert peak_direction < 5e-2

    @pytest.mark.slow
    def test_the_gap_shrinks_when_the_analytical_step_does(self):
        """Refining the analytical substep alone must not make the gap worse.

        There is no convergence study to run against MuJoCo: the MJCF pins its
        integrator at :data:`~udaan.core.defaults.DEFAULT_MJCF_TIMESTEP`, so
        shrinking ``dt`` shrinks the analytical substep while leaving MuJoCo's
        fixed. What that *does* isolate is the analytical model's own error at a
        fixed reference — and it stays bounded rather than growing, which is
        what a shared right-hand side implies.
        """
        gaps = []
        for divisor in (1, 2, 4):
            dt = 1.0 / (200.0 * divisor)
            mujoco_model = build(dt=dt)
            base = MultiQuadHoseBase(num_links=4, attachments=(0, 4), dt=dt, **LIGHT)
            kwargs = {"root_position": np.array([0.0, 0.0, 3.0])}
            mujoco_model.reset(**kwargs)
            base.reset(**kwargs)
            mujoco_model._input_type = mujoco_model.INPUT_TYPE.WRENCH
            base._input_type = base.INPUT_TYPE.WRENCH
            wrench = np.zeros((2, 4))
            wrench[:, 0] = base.total_mass * GRAVITY / 2
            while base.t < 1.0:
                mujoco_model.step(wrench)
                base.step(wrench)
            gaps.append(
                float(np.max(np.abs(mujoco_model.state.root_position - base.state.root_position)))
            )
        assert max(gaps) < 1e-3


# ── Divergence and the stiffness warning ─────────────────────────────


class TestDivergence:
    """A diverged MuJoCo integration raises instead of flying on from MuJoCo's reset."""

    def test_step_raises_when_mujoco_diverges(self):
        model = build(input="wrench")
        model.reset(root_position=np.array([0.0, 0.0, 2.0]))
        wrench = np.zeros((model.num_quadrotors, 4))
        model.step(wrench)
        model._mjMdl.data.qvel[0] = np.nan
        with pytest.raises(RuntimeError, match="diverged"):
            model.step(wrench)

    def test_simulate_stops_at_the_divergence(self):
        """MuJoCo resets its clock on divergence; ``simulate()`` must not run on.

        The NaN is injected once: a test that re-injected it every time the
        clock passed 0.1 s would fly forever on a backend without the guard.
        """
        model = build(input="wrench")
        target = np.array([0.0, 0.0, 2.0])
        data = model._mjMdl.data
        injected = []

        def setpoint(t):
            if t >= 0.1 and not injected:
                injected.append(t)
                data.qvel[0] = np.nan
            return target, np.zeros(3), np.zeros(3)

        model.hose_controller.setpoint = setpoint
        with pytest.raises(RuntimeError, match="diverged"):
            model.simulate(tf=1.0, root_position=target)
        assert model.t < 0.2


class TestStiffnessWarningScope:
    """The ZOH-Euler stiffness warning belongs to the analytical integrator only."""

    def test_the_mujoco_backend_does_not_warn(self, caplog):
        with caplog.at_level(logging.WARNING):
            model = MultiQuadHoseMujoco(num_links=8, attachments=(0, 8))
        assert model.stability_growth_rate() > 1.0
        assert "stiff for the ZOH-Euler integrator" not in caplog.text

    def test_the_analytical_model_still_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            model = MultiQuadHoseBase(num_links=8, attachments=(0, 8))
        assert model.stability_growth_rate() > 1.0
        assert "stiff for the ZOH-Euler integrator" in caplog.text


class TestClock:
    """The clock counts substeps, so ``k`` steps read ``k·dt`` on both backends."""

    @pytest.fixture(params=["base", "mujoco"])
    def model(self, request):
        if request.param == "base":
            return MultiQuadHoseBase(input="wrench", **LIGHT)
        return build(input="wrench")

    def test_one_second_is_two_hundred_steps(self, model):
        steps = []
        step = model.step

        def counting_step(u, desired_att=None):
            steps.append(model.t)
            step(u, desired_att)

        model.step = counting_step
        target = np.array([0.0, 0.0, 2.0])
        model.hose_controller.setpoint = lambda t: (target, np.zeros(3), np.zeros(3))
        model.simulate(tf=1.0, root_position=target)
        assert len(steps) == 200
        assert model.t == 1.0

    def test_clock_is_steps_times_dt_without_round_off(self, model):
        model.reset(root_position=np.array([0.0, 0.0, 2.0]))
        u = np.zeros((model.num_quadrotors, 4))
        u[:, 0] = model.total_mass * GRAVITY / model.num_quadrotors
        readings = {}
        for k in range(1, 2001):
            model.step(u)
            readings[k] = model.t
        # Exact at every whole second: an accumulated `t += h` drifts off them.
        assert [readings[k] for k in (200, 400, 1000, 2000)] == [1.0, 2.0, 5.0, 10.0]
        if isinstance(model, MultiQuadHoseMujoco):
            assert model._mjMdl.data.time == model.t
        model.reset(root_position=np.array([0.0, 0.0, 2.0]))
        assert model.t == 0.0
        model.step(u)
        assert model.t == model.dt


class TestDefaultAirframe:
    """The analytical model, its controller and its flat map default to the generated MJCF's airframe."""

    def test_the_analytical_default_is_the_mujoco_backend_airframe(self):
        base = MultiQuadHoseBase(input="wrench")
        mj = MultiQuadHoseMujoco(input="wrench")
        np.testing.assert_allclose(base.masses, mj.masses, rtol=0, atol=1e-9)
        np.testing.assert_allclose(base.inertias, mj.inertias, rtol=0, atol=1e-9)
        np.testing.assert_allclose(base.hose_controller.inertias, mj.inertias, rtol=0, atol=1e-9)
        np.testing.assert_allclose(
            base.hose_controller.flat_map.inertias, mj.inertias, rtol=0, atol=1e-9
        )

    def test_the_default_inertia_scales_with_the_mass(self):
        base = MultiQuadHoseBase(masses=0.45)
        mj = build(masses=0.45, hose_mass=0.5)
        np.testing.assert_allclose(base.inertias, mj.inertias, rtol=1e-6, atol=0)


class TestReferenceSmoothness:
    """``simulate()`` warns when the root path is not smooth enough for the flat map."""

    def _circle(self, ramp):
        from udaan.utils.trajectory import CircularTraj

        return CircularTraj(
            center=np.array([0.0, 0.0, 2.0]), radius=1.0, speed=1.0, th0=-np.pi / 2, ramp_time=ramp
        )

    def test_a_two_second_ramp_warns_and_records_its_steps(self, caplog):
        model = MultiQuadHoseBase(input="wrench")
        circle = self._circle(2.0)
        model.hose_controller.setpoint = circle.get
        with caplog.at_level(logging.WARNING):
            model.simulate(tf=2.5, root_position=circle.get(0.0)[0])
        assert "not smooth enough" in caplog.text
        steps = model.reference_steps
        assert [round(s.t, 3) for s in steps] == [0.0, 2.0]
        assert all(s.order == 7 for s in steps)
        assert max(s.acceleration for s in steps) > 0.3 * GRAVITY

    def test_a_six_second_ramp_stays_under_the_threshold(self):
        from udaan.core.defaults import MAX_HOSE_REFERENCE_ACCEL_STEP

        ctrl = MultiQuadHoseBase(input="wrench").hose_controller
        steps = ctrl.flat_map.reference_steps(ctrl.build_flats, self._circle(6.0).get_jet, 6.5)
        assert [round(s.t, 3) for s in steps] == [0.0, 6.0]
        assert max(s.acceleration for s in steps) < 0.01 * MAX_HOSE_REFERENCE_ACCEL_STEP

    @pytest.mark.parametrize("ramp", [2.0, 6.0])
    def test_a_circle_built_with_the_maps_order_has_no_joints(self, ramp):
        from udaan.utils.trajectory import CircularTraj

        ctrl = MultiQuadHoseBase(input="wrench").hose_controller
        circle = CircularTraj(
            center=np.array([0.0, 0.0, 2.0]),
            radius=1.0,
            speed=1.0,
            th0=-np.pi / 2,
            ramp_time=ramp,
            smoothness=ctrl.flat_map.jet_order,
        )
        assert ctrl.flat_map.reference_steps(ctrl.build_flats, circle.get_jet, ramp + 0.5) == []
