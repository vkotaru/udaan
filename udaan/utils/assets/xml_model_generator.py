#!/usr/bin/python

from ...core.defaults import (
    DEFAULT_ATTACHMENT_HEIGHT,
    DEFAULT_ATTACHMENT_RADIUS,
    DEFAULT_CABLE_LENGTH,
    DEFAULT_CABLE_MASS,
    DEFAULT_FORMATION_RADIUS,
    DEFAULT_GENERATED_QUAD_MASS,
    DEFAULT_HOSE_LENGTH,
    DEFAULT_HOSE_LINK_RADIUS,
    DEFAULT_HOSE_MASS,
    DEFAULT_HOSE_SAG_RATIO,
    DEFAULT_HOSE_VERTEX_RADIUS,
    DEFAULT_MAX_THRUST,
    DEFAULT_MAX_TORQUE,
    DEFAULT_MIN_THRUST,
    DEFAULT_MIN_TORQUE,
    DEFAULT_MJCF_TIMESTEP,
    DEFAULT_NUM_CABLE_LINKS,
    DEFAULT_NUM_HOSE_LINKS,
    DEFAULT_NUM_QUADROTORS,
    DEFAULT_PAYLOAD_MASS,
    DEFAULT_PAYLOAD_RADIUS,
    DEFAULT_PHYSICS_RATE_HZ,
    DEFAULT_RIGID_PAYLOAD_HALF_EXTENTS,
    DEFAULT_RIGID_PAYLOAD_MASS,
    DEFAULT_SPAWN_HEIGHT,
    DEFAULT_TENDON_LIMIT_TIMESTEPS,
    DEFAULT_TETHER_ANCHOR_HEIGHT,
    DEFAULT_TETHER_LENGTH,
    DEFAULT_TETHER_LINEAR_DENSITY,
    DEFAULT_TETHER_NUM_LINKS,
)
from ..formation import check_formation, default_formation_radius, default_spawn_height
from ..grasp import (
    check_rigid_formation,
    rigid_formation_radius,
    rigid_spawn_height,
    ring_attachment_points,
)
from .mujoco_asset_creator import *


def multi_quad_pointmass(nQ=2, filename="./assets/multi_quad_pointmass.xml", verbose=True):
    cable_length = 1.0
    z = 2.0
    r = 0.5
    dzL = np.sqrt(cable_length**2 - r**2)
    zL = z - dzL
    zL = 1.34
    dth = 2 * np.pi / nQ
    TH = np.arange(0, 2 * np.pi, dth)

    mjcWriter = MujocoAssetCreator("MultiQuadPointMass")
    for i in range(nQ):
        q1 = mjcWriter.create_quadrotor0(
            mjcWriter.worldbody,
            "quad" + str(i),
            np.array([r * np.cos(TH[i]), r * np.sin(TH[i]), z]),
        )

    pyld = mjcWriter.body(mjcWriter.worldbody, "pyld", pos=np.array([0.0, 0.0, zL]))
    mjcWriter.sphere(pyld, "pyld", radius=0.05, rgb=[0.2, 0.8, 0.4], mass=0.15)
    mjcWriter.site(
        pyld, "end2", pos=np.array([0.0, 0.0, 0.0]), type="sphere", size=[0.01], rgba=[0, 0, 0, 0]
    )
    mjcWriter.joint(pyld, "pyld_joint", type="free")

    tendon = mjcWriter.tendon(mjcWriter.root)
    for i in range(nQ):
        mjcWriter.spatial(tendon, "quad" + str(i) + "_end1", "end2", range=[0.0, cable_length])

    mjcWriter.save_to(filename, verbose=verbose)
    return


def payload_height(
    cable_length: float = DEFAULT_CABLE_LENGTH,
    formation_radius: float = DEFAULT_FORMATION_RADIUS,
    spawn_height: float = DEFAULT_SPAWN_HEIGHT,
) -> float:
    """Height of the payload that makes every cable of a ring formation taut.

    The quadrotors sit on a horizontal circle of ``formation_radius`` at
    ``spawn_height``; the payload hangs on the circle's axis. A taut cable of
    length ``l`` closes the right triangle ``r`` × ``(z - z_L)``::

        r² + (z - z_L)² = l²   =>   z_L = z - sqrt(l² - r²)

    Raises:
        ValueError: if ``cable_length <= formation_radius``, i.e. the cable is
            too short to reach the axis at all and the triangle has no
            solution.
    """
    if formation_radius < 0.0:
        raise ValueError(f"formation_radius must be >= 0, got {formation_radius}")
    if cable_length <= formation_radius:
        raise ValueError(
            f"cable_length ({cable_length}) must be strictly greater than the formation "
            f"radius ({formation_radius}); a taut cable from the ring to the payload axis "
            "has no solution otherwise"
        )
    return float(spawn_height - np.sqrt(cable_length**2 - formation_radius**2))


def multi_quad_cspayload(
    num_quadrotors: int = DEFAULT_NUM_QUADROTORS,
    filename: str = "./assets/multi_quad_cspayload.xml",
    cable_length: float = DEFAULT_CABLE_LENGTH,
    payload_mass: float = DEFAULT_PAYLOAD_MASS,
    quad_mass: float = DEFAULT_GENERATED_QUAD_MASS,
    formation_radius: float | None = None,
    spawn_height: float | None = None,
    verbose: bool = True,
) -> str:
    """Write the MJCF for ``num_quadrotors`` quadrotors carrying one point mass.

    The quadrotors are spread evenly around a circle of ``formation_radius`` at
    ``spawn_height``; each is tied to the payload by a spatial tendon limited to
    ``cable_length``. The payload height comes from :func:`payload_height`, so
    **the model starts taut** — the cables are exactly at their limit and there
    is no initial sag.

    The ring is checked by :func:`udaan.utils.formation.check_formation`:
    neighbouring airframes must be at least
    :data:`~udaan.core.defaults.DEFAULT_ROTOR_SPAN` plus
    :data:`~udaan.core.defaults.DEFAULT_FORMATION_CLEARANCE` apart, and the
    radius must stay below the cable length.

    The asset is the machine the analytical model describes at its defaults:
    the timestep is ``1/DEFAULT_PHYSICS_RATE_HZ``, each tendon's limit has a
    time constant of :data:`~udaan.core.defaults.DEFAULT_TENDON_LIMIT_TIMESTEPS`
    timesteps, and each airframe's ``ctrlrange`` is the actuator envelope of
    :mod:`udaan.core.defaults` (``DEFAULT_MIN/MAX_THRUST``,
    ``DEFAULT_MIN/MAX_TORQUE``).

    A single quadrotor (``num_quadrotors == 1``) hovers directly above the
    payload and the formation radius is unused, matching
    :func:`udaan.utils.grasp.formation_cable_attitudes`.

    Args:
        num_quadrotors: number of quadrotors, >= 1.
        filename: path to write the MJCF to.
        cable_length: taut cable length, in metres, shared by every cable.
        payload_mass: point-mass payload mass, in kg.
        quad_mass: total mass of each quadrotor, in kg.
        formation_radius: horizontal radius of the quadrotor ring, in metres;
            ``None`` is :func:`udaan.utils.formation.default_formation_radius`.
        spawn_height: height of the quadrotor ring, in metres; ``None`` is
            :func:`udaan.utils.formation.default_spawn_height`, which puts the
            payload at :data:`~udaan.core.defaults.DEFAULT_PAYLOAD_SPAWN_HEIGHT`.
        verbose: log the generated XML.

    Returns:
        The ``filename`` written.

    Raises:
        ValueError: for a ring :func:`~udaan.utils.formation.check_formation`
            refuses, or a payload that would spawn at or below the ground.
    """
    if num_quadrotors < 1:
        raise ValueError(f"num_quadrotors must be >= 1, got {num_quadrotors}")
    if num_quadrotors == 1:
        radius = 0.0
    else:
        radius = (
            default_formation_radius(num_quadrotors)
            if formation_radius is None
            else float(formation_radius)
        )
        check_formation(num_quadrotors, radius, cable_length)
    if spawn_height is None:
        spawn_height = default_spawn_height(cable_length, radius)
    z_payload = payload_height(cable_length, radius, spawn_height)
    if z_payload <= DEFAULT_PAYLOAD_RADIUS:
        # As fatal as `cable_length <= formation_radius` above: a payload
        # spawned inside the ground plane is not a model anyone asked for, and
        # a log warning is invisible by default.
        raise ValueError(
            f"payload would spawn at z={z_payload:.3f} m, at or below the ground plane "
            f"(payload radius {DEFAULT_PAYLOAD_RADIUS} m); raise spawn_height "
            f"({spawn_height}) or shorten the cable ({cable_length})"
        )

    angles = np.linspace(0.0, 2 * np.pi, num_quadrotors, endpoint=False)
    timestep = 1.0 / DEFAULT_PHYSICS_RATE_HZ
    torque_range = np.stack([DEFAULT_MIN_TORQUE, DEFAULT_MAX_TORQUE], axis=1)

    writer = MujocoAssetCreator("MultiQuadPointMass")
    writer.option(timestep=timestep)
    for i in range(num_quadrotors):
        writer.create_quadrotor0(
            writer.worldbody,
            f"quad{i}",
            np.array([radius * np.cos(angles[i]), radius * np.sin(angles[i]), spawn_height]),
            mass=quad_mass,
            # Undamped: the analytical M3 model has no drag term, and
            # MujocoAssetCreator's default 0.1 would make the backends
            # different machines.
            root_damping=0.0,
            thrust_range=(DEFAULT_MIN_THRUST, DEFAULT_MAX_THRUST),
            torque_range=torque_range,
        )

    pyld = writer.body(writer.worldbody, "pyld", pos=np.array([0.0, 0.0, z_payload]))
    writer.sphere(
        pyld, "pyld", radius=DEFAULT_PAYLOAD_RADIUS, rgb=[0.2, 0.8, 0.4], mass=payload_mass
    )
    writer.site(
        pyld, "end2", pos=np.array([0.0, 0.0, 0.0]), type="sphere", size=[0.01], rgba=[0, 0, 0, 0]
    )
    writer.joint(pyld, "pyld_joint", type="free", damping=0.0)

    tendon = writer.tendon(writer.root)
    for i in range(num_quadrotors):
        writer.spatial(
            tendon,
            f"quad{i}_end1",
            "end2",
            range=[0.0, cable_length],
            solreflimit=(DEFAULT_TENDON_LIMIT_TIMESTEPS * timestep, 1.0),
        )

    writer.save_to(filename, verbose=verbose)
    return filename


def rigid_payload_height(
    cable_length: float = DEFAULT_CABLE_LENGTH,
    formation_radius: float = DEFAULT_FORMATION_RADIUS,
    spawn_height: float = DEFAULT_SPAWN_HEIGHT,
    attachment_height: float = DEFAULT_ATTACHMENT_HEIGHT,
) -> float:
    r"""Payload-centre height that makes every cable of an M4 ring taut.

    The rigid-body sibling of :func:`payload_height`, and the difference is the
    whole point of M4: the cable does **not** end at the payload's centre of
    mass, it ends at attachment point ``r_i``.  Quadrotor ``i`` sits radially
    *outboard of its own attachment* — the pairing
    :func:`~udaan.utils.grasp.ring_attachment_points` /
    :func:`~udaan.utils.grasp.formation_cable_attitudes` builds, and the one
    that makes the hover trim moment-free — so the right triangle the cable
    closes has horizontal leg ``formation_radius`` alone, not
    ``attachment_radius + formation_radius``::

        r_f² + (z_Q - z_L - h_a)² = l²
        =>  z_L = z_Q - h_a - sqrt(l² - r_f²)

    with ``z_Q = spawn_height`` the height of the quadrotor ring, ``h_a =
    attachment_height`` the attachment ring's height above the payload centre,
    and ``r_f = formation_radius`` the *horizontal offset of each quadrotor
    from its own attachment point*.

    Any other payload height spawns the cables off their tendon limit; below
    it they start slack, carry zero tension at ``t = 0``, and (the tendons
    having no stiffness) the payload free-falls until the limit catches it — a
    mode the analytical model does not have.

    A single quadrotor hovers on the payload's own vertical axis, where
    ``r_f = 0`` and this reduces to ``z_L = z_Q - h_a - l``.

    Raises:
        ValueError: if ``formation_radius`` is negative, or if
            ``cable_length <= formation_radius`` — the cable is then too short
            to reach its attachment's axis and the triangle has no solution.
    """
    if formation_radius < 0.0:
        raise ValueError(f"formation_radius must be >= 0, got {formation_radius}")
    if cable_length <= formation_radius:
        raise ValueError(
            f"cable_length ({cable_length}) must be strictly greater than the formation "
            f"radius ({formation_radius}); a taut cable from the quadrotor ring down to "
            "its own attachment point has no solution otherwise"
        )
    return float(spawn_height - attachment_height - np.sqrt(cable_length**2 - formation_radius**2))


def multi_quad_rigidbody(
    num_quadrotors: int = DEFAULT_NUM_QUADROTORS,
    filename: str = "./assets/multi_quad_rigidbody.xml",
    cable_length: float = DEFAULT_CABLE_LENGTH,
    payload_mass: float = DEFAULT_RIGID_PAYLOAD_MASS,
    payload_half_extents=DEFAULT_RIGID_PAYLOAD_HALF_EXTENTS,
    quad_mass: float = DEFAULT_GENERATED_QUAD_MASS,
    attachment_radius: float = DEFAULT_ATTACHMENT_RADIUS,
    attachment_height: float = DEFAULT_ATTACHMENT_HEIGHT,
    formation_radius: float | None = None,
    spawn_height: float | None = None,
    verbose: bool = True,
) -> str:
    """Write the MJCF for ``num_quadrotors`` quadrotors carrying one rigid slab.

    The rigid-body sibling of :func:`multi_quad_pointmass`.  The payload is a
    box of ``payload_half_extents``; ``num_quadrotors`` sites ``pyld_poc{i}``
    sit on the ring :func:`~udaan.utils.grasp.ring_attachment_points` defines,
    and quadrotor ``i`` is placed *radially outboard of site* ``i`` on the same
    azimuth, tied to it by a spatial tendon limited to ``cable_length``.

    Four invariants hold by construction:

    * **The formation starts taut.**  The payload height comes from
      :func:`rigid_payload_height`, so every cable is exactly at its tendon
      limit at ``t = 0``.
    * **The azimuths agree with the model's.**  Both rings run at ``2πi/nQ``,
      the azimuths :func:`~udaan.utils.grasp.ring_attachment_points` and
      :func:`~udaan.utils.grasp.formation_cable_attitudes` share, so the grasp
      geometry this file encodes is the one
      :class:`~udaan.control.multi_quad_rigidbody.MultiQuadRigidbodyController`
      allocates through and its hover trim is exact. What matters is the
      **pairing** of sites with cable directions: sites on any other azimuth,
      read through the model's ``2πi/nQ`` cable directions as the controller
      reads them, leave a non-zero ``hover_residual``.
    * **The free joints are undamped.**  ``MujocoAssetCreator.joint`` defaults
      to ``damping=0.1``, and applied to the quadrotors' and the payload's
      *free* joints that is a viscous drag on every translational and
      rotational degree of freedom.
      :class:`~udaan.models.multi_quad_rigidbody.MultiQuadRigidbodyBase` has no
      such term, so damping would be a physics difference between the
      backends, not an integration one. It is therefore switched off here
      rather than modelled; the analytical model stays the definition of the
      machine.
    * **The integrator timestep is the default physics rate's**,
      ``1/DEFAULT_PHYSICS_RATE_HZ``. The MuJoCo backend overwrites it with
      ``dt/N`` from its own ``physics_rate_hz``, and writes the tendons' limit
      stiffness from that timestep.

    The ring is checked by :func:`udaan.utils.grasp.check_rigid_formation`:
    neighbouring airframes must be at least
    :data:`~udaan.core.defaults.DEFAULT_ROTOR_SPAN` plus
    :data:`~udaan.core.defaults.DEFAULT_FORMATION_CLEARANCE` apart, and the
    formation radius must stay below the cable length.

    Args:
        num_quadrotors: number of quadrotors, >= 1.
        filename: path to write the MJCF to.
        cable_length: taut cable length, in metres, shared by every cable.
        payload_mass: rigid-body payload mass, in kg.
        payload_half_extents: ``(a, b, c)`` half-extents of the slab, in metres.
        quad_mass: total mass of each quadrotor, in kg.
        attachment_radius: radius of the attachment ring in the payload body
            frame, in metres.
        attachment_height: height of the attachment ring above the payload
            centre of mass, in metres.
        formation_radius: horizontal offset of each quadrotor from *its own*
            attachment point, in metres; ``None`` is
            :func:`udaan.utils.grasp.rigid_formation_radius`.
        spawn_height: height of the quadrotor ring, in metres; ``None`` is
            :func:`udaan.utils.grasp.rigid_spawn_height`, which puts the
            payload's centre at
            :data:`~udaan.core.defaults.DEFAULT_PAYLOAD_SPAWN_HEIGHT`.
        verbose: log the generated XML.

    Returns:
        The ``filename`` written.

    Raises:
        ValueError: for a non-positive quadrotor count, a cable too short for
            the formation radius, neighbouring airframes closer than the rotor
            span plus the clearance, an attachment ring that leaves the
            payload, or a payload that would spawn at or below the ground
            plane.
    """
    if num_quadrotors < 1:
        raise ValueError(f"num_quadrotors must be >= 1, got {num_quadrotors}")

    extents = np.asarray(payload_half_extents, dtype=float).reshape(-1)
    if extents.shape[0] != 3:
        raise ValueError(f"payload_half_extents must be a 3-vector; got {extents.tolist()}")
    if np.any(extents <= 0.0):
        raise ValueError(f"payload_half_extents must be positive; got {extents.tolist()}")

    radius_a = 0.0 if num_quadrotors == 1 else float(attachment_radius)
    if radius_a < 0.0:
        raise ValueError(f"attachment_radius must be >= 0, got {attachment_radius}")
    # The sites are rigidly attached to the slab, so they have to be *on* it:
    # an attachment ring wider than the footprint describes a cable anchored to
    # thin air, and the moment arm r_i the whole grasp is built from would then
    # be a number with no body behind it.
    if radius_a > float(min(extents[0], extents[1])):
        raise ValueError(
            f"attachment_radius ({radius_a}) leaves the payload footprint "
            f"(half-extents {extents[0]} x {extents[1]}); the attachment sites would "
            "hang off the slab"
        )
    if abs(float(attachment_height)) > float(extents[2]):
        raise ValueError(
            f"attachment_height ({attachment_height}) is outside the slab's half-thickness "
            f"({extents[2]})"
        )

    if formation_radius is None:
        formation_radius = rigid_formation_radius(num_quadrotors, radius_a)
    radius_f = 0.0 if num_quadrotors == 1 else float(formation_radius)
    points = ring_attachment_points(num_quadrotors, radius_a, float(attachment_height))
    check_rigid_formation(points, radius_f, cable_length)
    if spawn_height is None:
        spawn_height = rigid_spawn_height(cable_length, radius_f, float(attachment_height))
    z_payload = rigid_payload_height(cable_length, radius_f, spawn_height, attachment_height)
    if z_payload - extents[2] <= 0.0:
        # As fatal as an infeasible triangle: a payload spawned inside the
        # ground plane is not a model anyone asked for, and a log warning is
        # invisible by default.
        raise ValueError(
            f"payload would spawn with its underside at z={z_payload - extents[2]:.3f} m, at "
            f"or below the ground plane; raise spawn_height ({spawn_height}) or shorten "
            f"the cable ({cable_length})"
        )

    angles = np.linspace(0.0, 2 * np.pi, num_quadrotors, endpoint=False)

    writer = MujocoAssetCreator("MultiQuadRigidbody")
    writer.option(timestep=1.0 / DEFAULT_PHYSICS_RATE_HZ)
    for i in range(num_quadrotors):
        # Radially outboard of attachment ``i``, on its own azimuth, lifted by
        # the vertical leg of the taut triangle. At nQ = 1 both radii are zero
        # and the quadrotor sits straight above the payload's axis.
        radius_q = radius_a + radius_f
        writer.create_quadrotor0(
            writer.worldbody,
            f"quad{i}",
            np.array([radius_q * np.cos(angles[i]), radius_q * np.sin(angles[i]), spawn_height]),
            mass=quad_mass,
            root_damping=0.0,
        )

    pyld = writer.body(writer.worldbody, "pyld", pos=np.array([0.0, 0.0, z_payload]))
    writer.box(pyld, "pyld", size=extents, rgb=[0.2, 0.8, 0.4], mass=payload_mass)
    for i in range(num_quadrotors):
        writer.site(
            pyld,
            f"pyld_poc{i}",
            pos=points[i],
            type="sphere",
            size=[0.01],
            rgba=[0, 0, 0, 0],
        )
    writer.joint(pyld, "pyld_joint", type="free", damping=0.0)

    tendon = writer.tendon(writer.root)
    for i in range(num_quadrotors):
        writer.spatial(tendon, f"quad{i}_end1", f"pyld_poc{i}", range=[0.0, cable_length])

    writer.save_to(filename, verbose=verbose)
    return filename


def quad_flexcable(
    num_links: int = DEFAULT_NUM_CABLE_LINKS,
    filename: str = "./assets/quad_flexcable.xml",
    cable_length: float = DEFAULT_CABLE_LENGTH,
    cable_mass: float = DEFAULT_CABLE_MASS,
    payload_mass: float = DEFAULT_PAYLOAD_MASS,
    quad_mass: float = DEFAULT_GENERATED_QUAD_MASS,
    spawn_height: float = DEFAULT_SPAWN_HEIGHT,
    verbose: bool = True,
) -> str:
    """Write the MJCF for one quadrotor carrying a payload on an ``n``-link cable.

    The M5 asset: :meth:`~udaan.utils.assets.mujoco_asset_creator.MujocoAssetCreator.create_quadrotor0`
    plus :meth:`~udaan.utils.assets.mujoco_asset_creator.MujocoAssetCreator.create_flexible_cable_payload`,
    hung from the airframe origin.  Unlike M3's and M4's assets there is no
    tendon anywhere: the cable is a **kinematic tree** of ball joints, so every
    link configuration is feasible and there is no taut/slack spawn condition
    to get wrong.  The chain spawns hanging straight down, and the payload ends
    up at ``spawn_height - cable_length``.

    Four choices make this asset and
    :class:`~udaan.models.quadrotor_flexcable.QuadrotorFlexCableBase` the same
    machine rather than merely similar ones:

    * **Each link's mass at its distal vertex** — the link cylinders are
      massless and each link's ``cable_mass / num_links`` sits in a small
      sphere at its lower end, the last one concentric with the payload, which
      is where the analytical model lumps it.

    * **No damping anywhere** — not on the airframe's free joint, not on the
      cable's ball joints, not on the payload's.  ``MujocoAssetCreator.joint``
      defaults to ``0.1`` and ``create_flexible_cable_payload`` to ``1e-3``
      (plus a near-welded ``1000/1000`` payload joint); the analytical model
      has no drag term at all, so damping would be a *physics* difference
      between the backends rather than an integration one.
    * **A 1.25 ms integrator timestep**, the analytical model's own inner
      substep, so it divides the 5 ms outer ``dt`` exactly.
    * **A free-spinning payload joint**, which is what makes the payload sphere
      a point mass: its ball joint is anchored at its own centre, so with zero
      stiffness and damping no torque ever reaches it and only its
      translational inertia participates.

    Args:
        num_links: number of rigid links the cable is discretised into, >= 1.
        filename: path to write the MJCF to.
        cable_length: total cable length, in metres.
        cable_mass: total mass of the cable itself, in kg.
        payload_mass: payload mass, in kg.
        quad_mass: total mass of the quadrotor, in kg.
        spawn_height: height of the airframe at spawn, in metres.
        verbose: log the generated XML.

    Returns:
        The ``filename`` written.

    Raises:
        ValueError: for a non-positive link count or cable length, or a payload
            that would spawn at or below the ground plane.
    """
    if num_links < 1:
        raise ValueError(f"num_links must be >= 1, got {num_links}")
    if cable_length <= 0.0:
        raise ValueError(f"cable_length must be positive, got {cable_length}")
    z_payload = float(spawn_height) - float(cable_length)
    if z_payload <= DEFAULT_PAYLOAD_RADIUS:
        raise ValueError(
            f"payload would spawn at z={z_payload:.3f} m, at or below the ground plane "
            f"(payload radius {DEFAULT_PAYLOAD_RADIUS} m); raise spawn_height "
            f"({spawn_height}) or shorten the cable ({cable_length})"
        )

    writer = MujocoAssetCreator("QuadFlexCable")
    writer.option(timestep=DEFAULT_MJCF_TIMESTEP)
    chassis, _ = writer.create_quadrotor0(
        writer.worldbody,
        "quad",
        np.array([0.0, 0.0, float(spawn_height)]),
        mass=quad_mass,
        root_damping=0.0,
    )
    writer.create_flexible_cable_payload(
        chassis,
        "cable",
        N=int(num_links),
        length=float(cable_length),
        mass=float(payload_mass),
        cable_mass=float(cable_mass),
        joint_damping=0.0,
        payload_joint_damping=0.0,
        payload_joint_stiffness=0.0,
        payload_radius=DEFAULT_PAYLOAD_RADIUS,
        vertex_masses=True,
    )
    writer.save_to(filename, verbose=verbose)
    return filename


def multi_quad_hose(
    num_links: int = DEFAULT_NUM_HOSE_LINKS,
    attachments=None,
    filename: str = "./assets/multi_quad_hose.xml",
    hose_length: float = DEFAULT_HOSE_LENGTH,
    hose_mass: float = DEFAULT_HOSE_MASS,
    quad_mass: float = DEFAULT_GENERATED_QUAD_MASS,
    sag_ratio: float = DEFAULT_HOSE_SAG_RATIO,
    spawn_height: float = DEFAULT_SPAWN_HEIGHT,
    link_radius: float = DEFAULT_HOSE_LINK_RADIUS,
    vertex_radius: float = DEFAULT_HOSE_VERTEX_RADIUS,
    verbose: bool = True,
) -> str:
    """Write the MJCF for ``nQ`` quadrotors carrying one ``n``-link flexible hose.

    The M6 asset.  It is a **kinematic tree**, not a
    tendon model — the hose is a chain of ball joints, so every link
    configuration is feasible and no link can stretch.

    Body layout
    -----------
    ``hose_vertex0`` is the root and carries the free joint.  Link ``i`` is a
    body ``hose_link{i}`` hinged to its parent by ``hose_joint{i}`` at its
    **proximal** end, whose ``z`` axis points from its distal end back to that
    joint — the convention
    :meth:`~udaan.utils.assets.mujoco_asset_creator.MujocoAssetCreator.create_flexible_cable_payload`
    and :class:`~udaan.models.quadrotor_flexcable.QuadrotorFlexCableMujoco`
    already use, so ``q_i = -R_i e_3`` reads a link direction exactly rather
    than by finite difference.

    **The hose's mass lives at the vertices, not in the links.**  Each link body
    carries a massless cylinder for looks and a sphere of the vertex mass at its
    distal end, so MuJoCo's ``inertiafromgeom`` puts that body's centre of mass
    exactly where the analytical model lumps it.  Vertex 0's mass is on the root
    body.  A quadrotor is a child of the body owning its vertex, attached by a
    **ball joint at its own centre of mass**: free to rotate, carried
    translationally by the hose, transmitting force but no torque — which is
    what makes the airframe's rotational dynamics decouple from the chain, as
    :class:`~udaan.models.multi_quad_hose.MultiQuadHoseBase` assumes.

    Spawn geometry
    --------------
    Taut, from :func:`udaan.utils.hose.hose_spawn_directions`: every segment
    between consecutive vehicles folds into a V whose horizontal span is
    ``sag_ratio`` times its own length, and the infeasible case — a span the
    hose cannot reach, or a fold it cannot close — is refused there rather than
    clamped.  The analytical model's :meth:`reset` reads the same function, so
    both backends spawn the same shape.

    Contacts are **disabled model-wide**.  The analytical model has no contact
    term, and this geometry deliberately allows overlap: a quadrotor's 0.4 m
    rotor span straddles the hose running through its own vertex, and a free
    hose end dangles back across the vehicle it hangs from.  Leaving contacts on
    would make the two backends different machines in a way that is not subtle.
    Damping is off everywhere for the same reason, as in
    :func:`multi_quad_rigidbody` and :func:`quad_flexcable`.

    Args:
        num_links: number of rigid links, >= 1.
        attachments: strictly increasing vertex indices carrying a quadrotor;
            ``None`` puts one at each end, ``(0, num_links)``.
        filename: path to write the MJCF to.
        hose_length: total hose length, in metres, split evenly over the links.
        hose_mass: total hose mass, in kg, lumped evenly onto the ``n + 1``
            vertices. Must be strictly positive — MuJoCo refuses a body with a
            joint and no mass, so unlike the analytical model this generator
            cannot emit a massless vertex.
        quad_mass: total mass of each quadrotor, in kg.
        sag_ratio: horizontal span of each interior segment as a fraction of
            that segment's own length.
        spawn_height: height of vertex 0 (the root), in metres.
        link_radius: radius of the (massless) link cylinders, in metres.
        vertex_radius: radius of the vertex mass spheres, in metres.
        verbose: log the generated XML.

    Returns:
        The ``filename`` written.

    Raises:
        ValueError: for a non-positive link count, length or mass, an invalid
            attachment list, or a spawn geometry with no solution.
    """
    from ..hose import hose_spawn_directions, validate_attachments

    if num_links < 1:
        raise ValueError(f"num_links must be >= 1, got {num_links}")
    if hose_length <= 0.0:
        raise ValueError(f"hose_length must be positive, got {hose_length}")
    if hose_mass <= 0.0:
        raise ValueError(
            f"hose_mass must be strictly positive, got {hose_mass}; MuJoCo refuses a body "
            "with a joint and no mass, so a massless hose vertex cannot be emitted"
        )

    n = int(num_links)
    indices = validate_attachments((0, n) if attachments is None else attachments, n)
    lengths = np.full(n, float(hose_length) / n)
    directions = hose_spawn_directions(lengths, indices, sag_ratio)
    return _write_hose(
        "MultiQuadHose",
        filename,
        lengths=lengths,
        indices=indices,
        vertex_mass=float(hose_mass) / (n + 1),
        directions=directions,
        quad_mass=quad_mass,
        root_position=np.array([0.0, 0.0, float(spawn_height)]),
        root_joint=True,
        link_radius=link_radius,
        vertex_radius=vertex_radius,
        verbose=verbose,
    )


def tethered_hose(
    num_links: int = DEFAULT_TETHER_NUM_LINKS,
    attachments=None,
    filename: str = "./assets/tethered_hose.xml",
    hose_length: float = DEFAULT_TETHER_LENGTH,
    hose_mass: float | None = None,
    quad_mass: float = DEFAULT_GENERATED_QUAD_MASS,
    anchor_height: float = DEFAULT_TETHER_ANCHOR_HEIGHT,
    link_radius: float = DEFAULT_HOSE_LINK_RADIUS,
    vertex_radius: float = DEFAULT_HOSE_VERTEX_RADIUS,
    verbose: bool = True,
) -> str:
    """Write the MJCF for quadrotors on an ``n``-link hose pinned to a ground anchor (M7 / M8).

    :func:`multi_quad_hose`'s body tree with **no root joint**: ``hose_vertex0``
    is welded to the world at ``(0, 0, anchor_height)``, so vertex 0 is the
    anchor exactly (no soft equality constraint, no drift) and ``hose_joint0``
    is the first ball joint.  The tree spawns standing straight up from the
    anchor, every ``q_i = +e_3``; the backend's ``reset()`` writes the trim it
    flies into the ball joints.  A thin massless plate under the anchor marks
    it.  Contacts and damping are off, as in :func:`multi_quad_hose`.

    Args:
        num_links: number of rigid links, >= 1.
        attachments: strictly increasing vehicle vertices, all >= 1, the last
            ``num_links``; ``None`` puts one vehicle at the top.
        filename: path to write the MJCF to.
        hose_length: total length, in metres, split evenly over the links.
        hose_mass: total hose mass, in kg, lumped evenly onto the ``n + 1``
            vertices (vertex 0's share sits on the welded anchor body);
            default 0.25 kg per metre of hose.
        quad_mass: mass of each quadrotor, in kg.
        anchor_height: height of the anchor, in metres.
        link_radius, vertex_radius: geometry, as in :func:`multi_quad_hose`.
        verbose: log the generated XML.

    Raises:
        ValueError: for a non-positive count, length or mass, or an attachment
            pattern with a vehicle at vertex 0 or none at the top.
    """
    from ..hose import validate_attachments

    if num_links < 1:
        raise ValueError(f"num_links must be >= 1, got {num_links}")
    if hose_length <= 0.0:
        raise ValueError(f"hose_length must be positive, got {hose_length}")
    if hose_mass is None:
        hose_mass = DEFAULT_TETHER_LINEAR_DENSITY * float(hose_length)
    if hose_mass <= 0.0:
        raise ValueError(f"hose_mass must be strictly positive, got {hose_mass}")
    n = int(num_links)
    indices = validate_attachments((n,) if attachments is None else attachments, n)
    if indices[0] < 1 or indices[-1] != n:
        raise ValueError(
            f"a tethered hose needs every vehicle at a vertex in [1, {n}] and one at vertex {n}; "
            f"got {indices}"
        )
    lengths = np.full(n, float(hose_length) / n)
    return _write_hose(
        "TetheredHose",
        filename,
        lengths=lengths,
        indices=indices,
        vertex_mass=float(hose_mass) / (n + 1),
        directions=np.tile(np.array([0.0, 0.0, 1.0]), (n, 1)),
        quad_mass=quad_mass,
        root_position=np.array([0.0, 0.0, float(anchor_height)]),
        root_joint=False,
        link_radius=link_radius,
        vertex_radius=vertex_radius,
        verbose=verbose,
    )


def _write_hose(
    model_name: str,
    filename: str,
    *,
    lengths: np.ndarray,
    indices: tuple[int, ...],
    vertex_mass: float,
    directions: np.ndarray,
    quad_mass: float,
    root_position: np.ndarray,
    root_joint: bool,
    link_radius: float,
    vertex_radius: float,
    verbose: bool,
) -> str:
    """Emit the hose's body tree: the shared writer of :func:`multi_quad_hose` and
    :func:`tethered_hose`.  ``root_joint`` gives ``hose_vertex0`` a free joint;
    without it the root body is welded to the world."""
    from scipy.spatial.transform import Rotation as sp_rot

    from ...manif import SO3

    n = int(lengths.shape[0])

    def _quat(rotation: np.ndarray) -> np.ndarray:
        """3x3 rotation -> ``(x, y, z, w)``, the order ``body()`` expects."""
        return np.asarray(sp_rot.from_matrix(np.asarray(rotation, dtype=float)).as_quat())

    def _exact_quat(element, quat: np.ndarray) -> None:
        """Rewrite ``element``'s ``quat`` at full precision.

        A body's ``quat`` is a structural rotation: MuJoCo composes it into the
        world orientation and into a child ball joint's anchor, so the six
        significant digits ``body()`` writes would put each anchor ~1e-7 m out
        of place, accumulating down the chain.
        """
        q = np.asarray(quat, dtype=float)
        element.attrib["quat"] = "%.17g %.17g %.17g %.17g" % (q[3], q[0], q[1], q[2])

    #: Link ``i``'s body frame: third column is ``-q_i``.
    frames = [
        np.asarray(SO3.from_two_vectors(-directions[i], np.array([1.0, 0.0, 0.0])))
        for i in range(n)
    ]
    identity = np.eye(3)

    writer = MujocoAssetCreator(model_name)
    writer.option(timestep=DEFAULT_MJCF_TIMESTEP, contact=False)

    def _attach_quadrotor(parent, vertex: int, pos, parent_frame):
        """Bolt vehicle ``k`` onto ``parent`` at ``pos``, world-aligned at spawn."""
        if vertex not in indices:
            return
        quat = _quat(np.asarray(parent_frame).T)
        chassis, actuator = writer.create_quadrotor0(
            parent,
            f"quad{indices.index(vertex)}",
            np.asarray(pos, dtype=float),
            quat=quat,
            mass=quad_mass,
            root_joint_type="ball",
            root_damping=0.0,
        )
        _exact_quat(chassis, quat)
        # Name each motor after its site, so the data.ctrl layout (written
        # positionally, vehicle by vehicle) can be checked against the bodies.
        for motor in actuator.iter("motor"):
            motor.attrib.setdefault("name", motor.attrib["site"])

    root = writer.body(writer.worldbody, "hose_vertex0", pos=np.asarray(root_position, dtype=float))
    writer.sphere(
        root, "hose_vertex_geom0", radius=vertex_radius, mass=vertex_mass, rgb=[0.15, 0.35, 0.7]
    )
    if root_joint:
        writer.joint(root, "hose_root_joint", "free", damping=0.0)
    else:
        writer.cylinder(
            root,
            "hose_anchor_plate",
            radius=4.0 * vertex_radius,
            length=0.01,
            mass=0.0,
            density=0.0,
            rgb=[0.3, 0.3, 0.3],
        )
    _attach_quadrotor(root, 0, np.zeros(3), identity)

    parent = root
    parent_frame = identity
    for i in range(n):
        half = 0.5 * float(lengths[i])
        # Midpoint of link i relative to the parent's origin: half of the
        # previous link plus half of this one, rotated into the parent's frame.
        offset_world = half * directions[i]
        if i > 0:
            offset_world = offset_world + 0.5 * float(lengths[i - 1]) * directions[i - 1]
        link_quat = _quat(parent_frame.T @ frames[i])
        link = writer.body(
            parent,
            f"hose_link{i}",
            pos=parent_frame.T @ offset_world,
            quat=link_quat,
        )
        _exact_quat(link, link_quat)
        writer.joint(
            link,
            f"hose_joint{i}",
            "ball",
            pos=np.array([0.0, 0.0, half]),
            damping=0.0,
            stiffness=0.0,
        )
        # Massless: the hose's mass is at the vertices, so MuJoCo's derived
        # body centre of mass lands on the distal sphere, where the analytical
        # model lumps it.
        writer.cylinder(
            link,
            f"hose_geom{i}",
            radius=link_radius,
            length=float(lengths[i]),
            mass=0.0,
            density=0.0,
            rgb=[0.2, 0.2, 0.2],
        )
        writer.sphere(
            link,
            f"hose_vertex_geom{i + 1}",
            pos=np.array([0.0, 0.0, -half]),
            radius=vertex_radius,
            mass=vertex_mass,
            rgb=[0.15, 0.35, 0.7],
        )
        _attach_quadrotor(link, i + 1, np.array([0.0, 0.0, -half]), frames[i])
        parent = link
        parent_frame = frames[i]

    writer.save_to(filename, verbose=verbose)
    return filename


def quadrotor_comparison(**kwargs):
    model_name = kwargs["model_name"] if "model_name" in kwargs else "QuadrotorComparison"
    if "filename" in kwargs:
        filename = kwargs["filename"]
    else:
        raise ValueError("filename not provided")
    verbosity = kwargs["verbose"] if "verbose" in kwargs else False

    unmodeled_dynamics = False
    if "unmodeled_mass" in kwargs:
        unmodeled_dynamics = True
        unmodeled_mass = kwargs["unmodeled_mass"]
    if "unmodeled_mass_loc" in kwargs:
        unmodeled_mass_loc = kwargs["unmodeled_mass_loc"]
    else:
        unmodeled_mass_loc = np.array([0.0, 0.0, 0.0])

    mjcWriter = MujocoAssetCreator(model_name)
    if unmodeled_dynamics:
        mjcWriter.create_quadrotor0(
            mjcWriter.worldbody,
            "plant",
            np.array([0.0, 0, 0.4]),
            rgb=[0.2, 0.4, 0.9],
            unmodeled_mass=unmodeled_mass,
            unmodeled_mass_loc=unmodeled_mass_loc,
        )
        mjcWriter.create_quadrotor0(
            mjcWriter.worldbody,
            "reference",
            np.array([0.0, 0, 0.4]),
            rgb=[0.1, 0.8, 0.8],
            alpha=0.25,
            unmodeled_mass=unmodeled_mass,
            unmodeled_mass_loc=unmodeled_mass_loc,
        )

    else:
        mjcWriter.create_quadrotor0(
            mjcWriter.worldbody,
            "plant",
            np.array([0.0, 0, 0.4]),
            rgb=[0.2, 0.4, 0.9],
        )
        mjcWriter.create_quadrotor0(
            mjcWriter.worldbody,
            "reference",
            np.array([0.0, 0, 0.4]),
            rgb=[0.1, 0.8, 0.8],
            alpha=0.25,
        )
    mjcWriter.exclude_contact("plant", "reference")
    mjcWriter.save_to(filename=filename, verbose=verbosity)
    return
