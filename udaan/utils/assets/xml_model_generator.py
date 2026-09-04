#!/usr/bin/python

import numpy as np

from ...core.defaults import (
    DEFAULT_ATTACHMENT_HEIGHT,
    DEFAULT_ATTACHMENT_RADIUS,
    DEFAULT_CABLE_LENGTH,
    DEFAULT_CABLE_MASS,
    DEFAULT_FORMATION_RADIUS,
    DEFAULT_MJCF_TIMESTEP,
    DEFAULT_NUM_CABLE_LINKS,
    DEFAULT_NUM_QUADROTORS,
    DEFAULT_PAYLOAD_MASS,
    DEFAULT_PAYLOAD_RADIUS,
    DEFAULT_QUAD_MASS,
    DEFAULT_RIGID_PAYLOAD_HALF_EXTENTS,
    DEFAULT_RIGID_PAYLOAD_MASS,
    DEFAULT_SPAWN_HEIGHT,
)
from ..grasp import ring_attachment_points
from .mujoco_asset_creator import MujocoAssetCreator


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
            solution. (This is the geometry the retired ``zL = 1.34`` literal
            papered over: at the default parameters it put the payload 0.206 m
            *above* the derived height of 1.134 m, which left each cable
            spanning only 0.828 m of its 1.0 m limit — 0.172 m of slack — so
            every simulation opened with a sag transient.)
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


def multi_quad_pointmass(
    num_quadrotors: int = DEFAULT_NUM_QUADROTORS,
    filename: str = "./assets/multi_quad_pointmass.xml",
    cable_length: float = DEFAULT_CABLE_LENGTH,
    payload_mass: float = DEFAULT_PAYLOAD_MASS,
    quad_mass: float = DEFAULT_QUAD_MASS,
    formation_radius: float = DEFAULT_FORMATION_RADIUS,
    spawn_height: float = DEFAULT_SPAWN_HEIGHT,
    verbose: bool = True,
) -> str:
    """Write the MJCF for ``num_quadrotors`` quadrotors carrying one point mass.

    The quadrotors are spread evenly around a circle of ``formation_radius`` at
    ``spawn_height``; each is tied to the payload by a spatial tendon limited to
    ``cable_length``. The payload height comes from :func:`payload_height`, so
    **the model starts taut** — the cables are exactly at their limit and there
    is no initial sag.

    A single quadrotor (``num_quadrotors == 1``) hovers directly above the
    payload and the formation radius is unused, matching
    :func:`udaan.utils.grasp.formation_cable_attitudes`.

    Args:
        num_quadrotors: number of quadrotors, >= 1.
        filename: path to write the MJCF to.
        cable_length: taut cable length, in metres, shared by every cable.
        payload_mass: point-mass payload mass, in kg.
        quad_mass: total mass of each quadrotor, in kg.
        formation_radius: horizontal radius of the quadrotor ring, in metres.
        spawn_height: height of the quadrotor ring, in metres.
        verbose: log the generated XML.

    Returns:
        The ``filename`` written.
    """
    if num_quadrotors < 1:
        raise ValueError(f"num_quadrotors must be >= 1, got {num_quadrotors}")
    radius = 0.0 if num_quadrotors == 1 else float(formation_radius)
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

    writer = MujocoAssetCreator("MultiQuadPointMass")
    for i in range(num_quadrotors):
        writer.create_quadrotor0(
            writer.worldbody,
            f"quad{i}",
            np.array([radius * np.cos(angles[i]), radius * np.sin(angles[i]), spawn_height]),
            mass=quad_mass,
            # Undamped, for the reason `multi_quad_rigidbody` spells out: the
            # analytical M3 model has no drag term, and MujocoAssetCreator's
            # default 0.1 would make the backends different machines.
            root_damping=0.0,
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
        writer.spatial(tendon, f"quad{i}_end1", "end2", range=[0.0, cable_length])

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

    The retired hand-written asset already used this reading of the triangle —
    measured from that file, ``quad0`` sits at radius 1.20711 and its site
    ``pyld_poc0`` at 0.70711, both at 45°, so the horizontal leg between them
    is ``formation_radius`` (0.5) alone, on the attachment's own azimuth. What
    it got wrong was the *vertical* leg: it put the payload at ``z = 1.34`` —
    a literal, chosen rather than derived — where this formula gives 1.134,
    leaving 0.789 m of cable against a 1.0 m tendon limit: 21% slack, zero
    tension at ``t = 0``, and (the tendons carrying no stiffness) a payload
    that free-falls until the limit constraint catches it. The analytical
    model has no slack mode at all, so the two backends were not simulating
    the same machine. Its second defect was independent of this function: its
    site ring was rotated 45° off the model's, so the pairing the controller
    allocates through was not the one the file encoded.

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
    quad_mass: float = DEFAULT_QUAD_MASS,
    attachment_radius: float = DEFAULT_ATTACHMENT_RADIUS,
    attachment_height: float = DEFAULT_ATTACHMENT_HEIGHT,
    formation_radius: float = DEFAULT_FORMATION_RADIUS,
    spawn_height: float = DEFAULT_SPAWN_HEIGHT,
    verbose: bool = True,
) -> str:
    """Write the MJCF for ``num_quadrotors`` quadrotors carrying one rigid slab.

    The rigid-body sibling of :func:`multi_quad_pointmass`.  The payload is a
    box of ``payload_half_extents``; ``num_quadrotors`` sites ``pyld_poc{i}``
    sit on the ring :func:`~udaan.utils.grasp.ring_attachment_points` defines,
    and quadrotor ``i`` is placed *radially outboard of site* ``i`` on the same
    azimuth, tied to it by a spatial tendon limited to ``cable_length``.

    Three invariants hold by construction, and the shipped hand-written asset
    broke the first two:

    * **The formation starts taut.**  The payload height comes from
      :func:`rigid_payload_height`, so every cable is exactly at its tendon
      limit at ``t = 0``. The old asset spawned 21% slack.
    * **The azimuths agree with the model's.**  Both rings run at ``2πi/nQ``,
      the azimuths :func:`~udaan.utils.grasp.ring_attachment_points` and
      :func:`~udaan.utils.grasp.formation_cable_attitudes` share, so the grasp
      geometry this file encodes is the one
      :class:`~udaan.control.multi_quad_rigidbody.MultiQuadRigidbodyController`
      allocates through and its hover trim is exact. The old asset put the
      sites on the slab's corners (45°, 135°, …) while the model's ring runs at
      0°, 90°, …. Corner attachments are not themselves un-hoverable — paired
      with the 45° cable directions the old asset's *own* quadrotor ring
      implies, the same corners trim to a 7e-15 N residual. What cannot hover
      is the **pairing**: corner sites read through the model's 0°/90° cable
      directions, which is what the controller does, and the controller's own
      ``hover_residual`` reads 2.7208 N for that combination against 3.9e-15 N
      for the consistent ring.
    * **The free joints are undamped.**  ``MujocoAssetCreator.joint`` defaults
      to ``damping=0.1``, and applied to the quadrotors' and the payload's
      *free* joints that is a viscous drag on every translational and
      rotational degree of freedom.
      :class:`~udaan.models.multi_quad_rigidbody.MultiQuadRigidbodyBase` has no
      such term, so it is a physics difference between the backends and not an
      integration one: over the 2 s step manoeuvre of
      ``TestBackendAgreement`` it accounted for two thirds of the discrepancy
      the tolerance was absorbing — at otherwise identical integration the
      peak payload-position gap fell from 2.12e-2 m to 7.29e-3 m and the peak
      attitude gap from 2.34° to 0.85° when it was removed. Damping is
      therefore switched off here rather than modelled; the analytical model
      stays the definition of the machine.
    * **The integrator timestep divides the model's** ``dt``. The file sets
      :data:`~udaan.core.defaults.DEFAULT_MJCF_TIMESTEP`, the analytical
      model's own inner substep. MuJoCo's default of 2 ms does not divide the
      5 ms outer loop, so the backend took ``round(2.5) = 2`` substeps and
      advanced 4 ms of physics per ``step()`` while reporting ``dt = 5 ms``.

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
            attachment point, in metres.
        spawn_height: height of the quadrotor ring, in metres.
        verbose: log the generated XML.

    Returns:
        The ``filename`` written.

    Raises:
        ValueError: for a non-positive quadrotor count, a cable too short for
            the formation radius, an attachment ring that leaves the payload,
            or a payload that would spawn at or below the ground plane.
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

    radius_f = 0.0 if num_quadrotors == 1 else float(formation_radius)
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

    points = ring_attachment_points(num_quadrotors, radius_a, float(attachment_height))
    angles = np.linspace(0.0, 2 * np.pi, num_quadrotors, endpoint=False)

    writer = MujocoAssetCreator("MultiQuadRigidbody")
    writer.option(timestep=DEFAULT_MJCF_TIMESTEP)
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
    quad_mass: float = DEFAULT_QUAD_MASS,
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

    Three departures from the historical hand-written ``quad_flxcbl_mj.xml``,
    all of them there to make this asset and
    :class:`~udaan.models.quadrotor_flexcable.QuadrotorFlexCableBase` the same
    machine rather than merely similar ones:

    * **No damping anywhere** — not on the airframe's free joint, not on the
      cable's ball joints, not on the payload's.  ``MujocoAssetCreator.joint``
      defaults to ``0.1`` and ``create_flexible_cable_payload`` to ``1e-3``
      (plus a near-welded ``1000/1000`` payload joint); the analytical model
      has no drag term at all, so damping would be a *physics* difference
      between the backends rather than an integration one.  The same reasoning
      :func:`multi_quad_rigidbody` records for its free joints.
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
    )
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
            rgb=[1.0, 0.0, 0.0],
            unmodeled_mass=unmodeled_mass,
            unmodeled_mass_loc=unmodeled_mass_loc,
        )
        mjcWriter.create_quadrotor0(
            mjcWriter.worldbody,
            "reference",
            np.array([0.0, 0, 0.4]),
            rgb=[1.0, 0.0, 0.4],
            alpha=0.25,
            unmodeled_mass=unmodeled_mass,
            unmodeled_mass_loc=unmodeled_mass_loc,
        )

    else:
        mjcWriter.create_quadrotor0(
            mjcWriter.worldbody,
            "plant",
            np.array([0.0, 0, 0.4]),
            rgb=[1.0, 0.0, 0.0],
        )
        mjcWriter.create_quadrotor0(
            mjcWriter.worldbody,
            "reference",
            np.array([0.0, 0, 0.4]),
            rgb=[1.0, 0.0, 0.4],
            alpha=0.25,
        )
    mjcWriter.exclude_contact("plant", "reference")
    mjcWriter.save_to(filename=filename, verbose=verbosity)
    return
