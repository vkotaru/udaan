#!/usr/bin/python

from ...core.defaults import (
    DEFAULT_CABLE_LENGTH,
    DEFAULT_FORMATION_RADIUS,
    DEFAULT_GENERATED_QUAD_MASS,
    DEFAULT_MAX_THRUST,
    DEFAULT_MAX_TORQUE,
    DEFAULT_MIN_THRUST,
    DEFAULT_MIN_TORQUE,
    DEFAULT_NUM_QUADROTORS,
    DEFAULT_PAYLOAD_MASS,
    DEFAULT_PAYLOAD_RADIUS,
    DEFAULT_PHYSICS_RATE_HZ,
    DEFAULT_SPAWN_HEIGHT,
    DEFAULT_TENDON_LIMIT_TIMESTEPS,
)
from ..formation import check_formation, default_formation_radius, default_spawn_height
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
