#!/usr/bin/python

import numpy as np

from ...core.defaults import (
    DEFAULT_CABLE_LENGTH,
    DEFAULT_FORMATION_RADIUS,
    DEFAULT_NUM_QUADROTORS,
    DEFAULT_PAYLOAD_MASS,
    DEFAULT_PAYLOAD_RADIUS,
    DEFAULT_QUAD_MASS,
    DEFAULT_SPAWN_HEIGHT,
)
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
        )

    pyld = writer.body(writer.worldbody, "pyld", pos=np.array([0.0, 0.0, z_payload]))
    writer.sphere(
        pyld, "pyld", radius=DEFAULT_PAYLOAD_RADIUS, rgb=[0.2, 0.8, 0.4], mass=payload_mass
    )
    writer.site(
        pyld, "end2", pos=np.array([0.0, 0.0, 0.0]), type="sphere", size=[0.01], rgba=[0, 0, 0, 0]
    )
    writer.joint(pyld, "pyld_joint", type="free")

    tendon = writer.tendon(writer.root)
    for i in range(num_quadrotors):
        writer.spatial(tendon, f"quad{i}_end1", "end2", range=[0.0, cable_length])

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
