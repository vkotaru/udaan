#!/usr/bin/python

import math
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation as sp_rot

from ..logging import get_logger

_logger = get_logger(__name__)

#: Version of the MJCF *structure* emitted by this module and the generators
#: built on it (:mod:`udaan.utils.assets.xml_model_generator`,
#: :class:`~udaan.models.mujoco.QuadrotorFleet`,
#: :class:`~udaan.models.mujoco.QuadrotorCsPayloadFleet`).
#:
#: **Bump this whenever the emitted XML changes shape** — geom masses, actuator
#: names or order, body/site layout, tendon wiring, anything a file on disk
#: would freeze. Every runtime asset cache hashes it into the generated
#: filename, so a bump makes stale cached XML unreachable instead of silently
#: reused by an upgraded ``udaan`` with a warm asset directory.
#:
#: History:
#:
#: * ``3`` — two shared-code changes that both alter emitted text:
#:
#:   1. :meth:`MujocoAssetCreator.sphere` began emitting the ``pos`` it had
#:      always accepted but silently dropped. Every pre-existing caller passes
#:      the default ``[0, 0, 0]``, which is MJCF's own, so this is text-only:
#:      ``multi_quad_pointmass`` (1q–4q), ``quad_flexcable`` (1/2/3/5 links)
#:      and ``cspayload_fleet`` (2q–4q) each gain one ``pos="0.0 0.0 0.0"`` per
#:      sphere geom, and a compile-and-compare of every pair over
#:      ``body_mass``/``inertia``/``pos``/``quat``/``ipos``/``iquat``,
#:      ``geom_pos``/``quat``/``size``, ``jnt_pos``, ``dof_damping`` and
#:      ``tendon_range`` finds *zero* difference.
#:   2. ``body``, ``geom`` and ``site`` quaternions went from ``%g`` to
#:      ``%.17g``. Unlike (1) this is not purely textual: the four rotor-arm
#:      boxes of ``create_quadrotor0`` sit at ±45°/±135°, so *every* generated
#:      family's text changes and the compiled model moves — ``geom_quat`` by
#:      5.3e-7, ``body_inertia`` by 6e-10 (relative ~1e-7). The motion is
#:      *toward* the exact value: six significant digits of a rotation is
#:      ~1e-7 of placement error, and down a chain of ball joints that is a
#:      backend disagreement no amount of integration care can remove.
#:
#:   Either change alone means a file frozen on disk under the version-2 hash
#:   is no longer what this module emits, which is what the bump is for.
MJCF_GENERATOR_VERSION = 3

# Per-geom masses of the rotor assembly emitted by ``create_quadrotor0`` at the
# default total mass. A requested total scales all four geoms uniformly (see
# `create_quadrotor0`), so these are reference values, not fixed ones.
_ARM_MASS = 0.01  # kg, one rotor arm
_PROP_MASS = 0.05  # kg, one propeller disc
_N_ROTORS = 4
#: Mass of the four arm + propeller pairs at the default total mass.
_ROTOR_ASSEMBLY_MASS = _N_ROTORS * (_ARM_MASS + _PROP_MASS)
#: Chassis geom mass used when no total quadrotor mass is requested.
_DEFAULT_CHASSIS_MASS = 0.75  # kg
#: Total quadrotor mass the geom masses above were tuned for.
_DEFAULT_QUAD_TOTAL_MASS = _DEFAULT_CHASSIS_MASS + _ROTOR_ASSEMBLY_MASS  # 0.99 kg
#: Smallest total quadrotor mass accepted by ``create_quadrotor0``. Well below
#: any real airframe, but a strictly positive floor: without it a total just
#: above the rotor assembly used to leave a ~1e-7 kg chassis geom, and MuJoCo
#: would happily build a body whose inertia is pure rounding noise.
_MIN_QUAD_MASS = 0.05  # kg


class MujocoAssetCreator:
    """Adapted from IsaacGymEnvs

    Args:
        object (_type_): _description_
    """

    def __init__(self, name="mujoco_asset"):
        self.name = name
        self.root = ET.Element("mujoco")
        self.root.attrib["model"] = name
        include = ET.SubElement(self.root, "include")
        include.attrib["file"] = "scene.xml"
        compiler = ET.SubElement(self.root, "compiler")
        compiler.attrib["angle"] = "degree"
        compiler.attrib["coordinate"] = "local"
        compiler.attrib["inertiafromgeom"] = "true"
        self.worldbody = ET.SubElement(self.root, "worldbody")
        # One shared <actuator> block, created lazily by `actuator()`.
        self._actuator = None

    def save_to(self, filename="mujoco_asset.xml", verbose=False):
        """Save the asset to a file

        Args:
            filename (str, optional): Name of the file to save to. Defaults to "mujoco_asset.xml".
        """
        # Pretty printing to Python shell for testing purposes
        from xml.dom import minidom

        xmlstr = minidom.parseString(ET.tostring(self.root)).toprettyxml(indent="   ")
        if verbose:
            _logger.debug("Generated XML:\n%s", xmlstr)
        with open(filename, "w") as f:
            f.write(xmlstr)
        return

    def option(self, timestep: float, **flags):
        """Emit ``<option timestep=...>``, overriding MuJoCo's default 0.002 s.

        Worth setting whenever the asset is stepped by a backend with its own
        outer ``dt``: that backend computes ``round(dt / timestep)`` substeps,
        so a timestep that does not divide ``dt`` makes it advance a different
        amount of time from the one it reports. At udaan's ``dt = 5 ms`` the
        MuJoCo default gives ``round(2.5) = 2`` — 4 ms of physics per 5 ms
        step.

        Keyword Args:
            **flags: MuJoCo ``<flag>`` switches, as ``name=True/False`` (or a
                literal ``"enable"``/``"disable"``). ``contact=False`` is the
                one udaan uses: an analytical model with no contact term and an
                asset that can collide with itself are two different machines,
                and for a model whose bodies are deliberately allowed to
                overlap — see :func:`udaan.utils.assets.multi_quad_hose` — the
                difference is not subtle.
        """
        if float(timestep) <= 0.0:
            raise ValueError(f"timestep must be positive, got {timestep}")
        option = ET.SubElement(self.root, "option")
        option.attrib["timestep"] = repr(float(timestep))
        if flags:
            flag = ET.SubElement(option, "flag")
            for name, value in flags.items():
                if isinstance(value, bool):
                    value = "enable" if value else "disable"
                flag.attrib[name] = str(value)
        return option

    def asset(self):
        asset = ET.SubElement(self.root, "asset")
        texture = ET.SubElement(asset, "texture")
        texture.attrib["type"] = "skybox"
        texture.attrib["builtin"] = "gradient"
        texture.attrib["rgb1"] = "1.0 1.0 1.0"
        texture.attrib["rgb2"] = "0.6 0.8 1.0"
        texture.attrib["width"] = "256"
        texture.attrib["height"] = "256"
        return asset

    def fancy_asset(self):
        visual = ET.SubElement(self.root, "visual")
        map = ET.SubElement(visual, "map")
        map.attrib["fogstart"] = "3.0"
        map.attrib["fogend"] = "5.0"
        map.attrib["force"] = "0.1"
        map.attrib["znear"] = "0.1"
        asset = ET.SubElement(self.root, "asset")
        texture1 = ET.SubElement(asset, "texture")
        texture1.attrib["type"] = "skybox"
        texture1.attrib["builtin"] = "gradient"
        texture1.attrib["rgb1"] = "1.0 1.0 1.0"
        texture1.attrib["rgb2"] = "0.6 0.8 1.0"
        texture1.attrib["width"] = "127"
        texture1.attrib["height"] = "1278"
        texture2 = ET.SubElement(asset, "texture")
        texture2.attrib["name"] = "texplane"
        texture2.attrib["type"] = "2d"
        texture2.attrib["builtin"] = "checker"
        texture2.attrib["rgb1"] = "0.2 0.3 0.4"
        texture2.attrib["rgb2"] = ".1 0.15 0.2"
        texture2.attrib["width"] = "512"
        texture2.attrib["height"] = "512"

        mat1 = ET.SubElement(asset, "material")
        mat1.attrib["name"] = "MatPlane"
        mat1.attrib["texture"] = "texplane"
        mat1.attrib["reflectance"] = "0.5"
        mat1.attrib["texrepeat"] = "1 1"
        mat1.attrib["texuniform"] = "true"
        return asset

    def exclude_contact(self, body1: str, body2: str):
        contact = ET.SubElement(self.root, "contact")
        exclude = ET.SubElement(contact, "exclude")
        exclude.attrib["body1"] = body1
        exclude.attrib["body2"] = body2
        return contact

    def tendon(self, parent):
        tendon = ET.SubElement(parent, "tendon")
        return tendon

    def spatial(
        self,
        parent,
        site1,
        site2,
        range=[0.0, 1.0],
        width=0.003,
        damping=0.0,
        stiffness=0.0,
        rgba=[0.9, 0.9, 0.9, 1.0],
    ):
        spatial = ET.SubElement(parent, "spatial")
        spatial.attrib["limited"] = "true"
        spatial.attrib["range"] = " ".join([str(x) for x in range])
        spatial.attrib["width"] = str(width)
        spatial.attrib["damping"] = str(damping)
        spatial.attrib["stiffness"] = str(stiffness)
        spatial.attrib["rgba"] = " ".join([str(x) for x in rgba])
        s1 = ET.SubElement(spatial, "site")
        s1.attrib["site"] = site1
        s2 = ET.SubElement(spatial, "site")
        s2.attrib["site"] = site2
        return spatial

    def ground_plane(
        self,
        pos=np.array([0.0, 0, 0.0]),
        size=2,
        friction=1.0,
        rgba=[0.8, 0.9, 0.8, 1.0],
    ):
        """Create a ground plane

        Args:
            size (int, optional): Size of the ground plane. Defaults to 10.
            friction (float, optional): Friction of the ground plane. Defaults to 1.0.
            rgba (list, optional): RGBA values of the ground plane. Defaults to [0.8, 0.9, 0.8, 1.0].
        """
        geom = ET.SubElement(self.worldbody, "geom")
        geom.attrib["name"] = "ground"
        geom.attrib["type"] = "plane"
        geom.attrib["size"] = str(size) + " " + str(size) + " 0.02"
        geom.attrib["friction"] = str(friction)
        geom.attrib["pos"] = " ".join([str(x) for x in pos])
        geom.attrib["rgba"] = " ".join([str(x) for x in rgba])
        return geom

    def light(self):
        light = ET.SubElement(self.worldbody, "light")
        light.attrib["directional"] = "true"
        light.attrib["diffuse"] = ".9 .9 .9"
        light.attrib["specular"] = ".3 .3 .3"
        light.attrib["pos"] = "0 0 4.0"
        light.attrib["dir"] = "0 0.15 -1"
        return light

    def body(
        self,
        parent,
        name,
        pos=np.array([0.0, 0.0, 0.0]),
        quat=np.array([0.0, 0.0, 0.0, 1.0]),
        mass=1.0,
        inertia=np.array([1.0, 1.0, 1.0]),
        rgba=[0.8, 0.9, 0.8, 1.0],
    ):
        body = ET.SubElement(parent, "body")
        body.attrib["name"] = name
        body.attrib["pos"] = " ".join([str(x) for x in pos])
        # Full precision, not ``%g``. A body's ``quat`` is a *structural*
        # rotation: MuJoCo composes it into the world orientation and, for a
        # child with a ball joint, into the joint anchor as well, so six
        # significant digits put the anchor ~1e-7 m out of place — which
        # accumulates down a chain and shows up as a backend disagreement no
        # amount of integration care can remove. M6 is the first caller to pass
        # a non-identity quaternion here; every pre-existing one passes the
        # identity, whose ``%.17g`` and ``%g`` renderings are the same text, so
        # this line on its own changes nothing already on disk. (The same
        # change to ``geom`` below is *not* text-neutral — see the note there
        # and :data:`MJCF_GENERATOR_VERSION`.)
        body.attrib["quat"] = "%.17g %.17g %.17g %.17g" % (quat[3], quat[0], quat[1], quat[2])
        # body.attrib["mass"] = str(mass)
        # body.attrib["inertia"] = " ".join([str(x) for x in inertia])
        return body

    def cylinder(
        self,
        parent,
        name,
        pos=np.array([0.0, 0.0, 0.0]),
        mass=0.01,
        density=1000,
        radius=0.1,
        length=0.1,
        alpha=1.0,
        rgb=[0.5, 0.1, 0.1],
    ):
        geom = ET.SubElement(parent, "geom")
        geom.attrib["name"] = name
        geom.attrib["type"] = "cylinder"
        geom.attrib["size"] = "%g %g" % (radius, 0.5 * length)
        geom.attrib["pos"] = " ".join([str(x) for x in pos])
        geom.attrib["mass"] = str(mass)
        geom.attrib["density"] = str(density)
        geom.attrib["rgba"] = " ".join([str(x) for x in rgb + [alpha]])
        return geom

    def sphere(
        self,
        parent,
        name,
        pos=np.array([0.0, 0.0, 0.0]),
        quat=np.array([0.0, 0.0, 0.0, 1.0]),
        radius=0.1,
        mass=0.01,
        alpha=1.0,
        rgb=[0.1, 0.5, 0.1],
        density=1000,
    ):
        geom = ET.SubElement(parent, "geom")
        geom.attrib["name"] = name
        geom.attrib["type"] = "sphere"
        geom.attrib["size"] = "%g" % (radius)
        # `pos` was accepted and silently dropped until M6 needed a sphere
        # offset inside its body (the vertex mass at a link's distal end).
        # Emitting it matches `box`/`cylinder`; every pre-existing caller passes
        # the default, which is MJCF's own, so no shipped asset changes meaning
        # — but the emitted *text* does gain a `pos="0.0 0.0 0.0"` per sphere,
        # which is why `MJCF_GENERATOR_VERSION` moved to 3.
        geom.attrib["pos"] = " ".join([str(x) for x in pos])
        geom.attrib["mass"] = str(mass)
        geom.attrib["density"] = str(density)
        geom.attrib["rgba"] = " ".join([str(x) for x in rgb + [alpha]])
        return geom

    def box(
        self,
        parent,
        name,
        pos=np.array([0.0, 0.0, 0.0]),
        quat=np.array([0.0, 0.0, 0.0, 1.0]),
        size=np.array([0.1, 0.1, 0.1]),
        mass=0.01,
        alpha=1.0,
        rgb=[0.1, 0.1, 0.5],
        density=1000,
    ):
        geom = ET.SubElement(parent, "geom")
        geom.attrib["name"] = name
        geom.attrib["type"] = "box"
        geom.attrib["pos"] = " ".join([str(x) for x in pos])
        # Full precision, for the same reason as ``body`` above: six digits of
        # a rotation is ~1e-7 of placement error. Unlike ``body``, this one
        # *does* have non-identity callers today — the four rotor-arm boxes of
        # ``create_quadrotor0`` at ±45°/±135° — so every generated family's
        # text changes, and its compiled ``geom_quat`` moves 5.3e-7 (and
        # ``body_inertia`` 6e-10) closer to the exact value. Hence
        # ``MJCF_GENERATOR_VERSION`` 3.
        geom.attrib["quat"] = "%.17g %.17g %.17g %.17g" % (quat[3], quat[0], quat[1], quat[2])
        geom.attrib["size"] = " ".join([str(x) for x in size])
        geom.attrib["mass"] = str(mass)
        geom.attrib["density"] = str(density)
        geom.attrib["rgba"] = " ".join([str(x) for x in rgb + [alpha]])
        return geom

    def joint(
        self,
        parent,
        name,
        type,
        pos=np.array([0.0, 0.0, 0.0]),
        axis=np.array([0.0, 0.0, 1.0]),
        range=np.array([-1.0, 1.0]),
        damping=0.1,
        stiffness=0.0,
        armature=0.0,
        limited=False,
    ):
        joint = ET.SubElement(parent, "joint")
        joint.attrib["name"] = name
        joint.attrib["type"] = type
        if limited:
            joint.attrib["limited"] = "true"
        else:
            joint.attrib["limited"] = "false"
        joint.attrib["pos"] = " ".join([str(x) for x in pos])
        if type != "ball":
            joint.attrib["axis"] = " ".join([str(x) for x in axis])
            joint.attrib["range"] = " ".join([str(x) for x in range])
            joint.attrib["armature"] = str(armature)
        joint.attrib["damping"] = str(damping)
        joint.attrib["stiffness"] = str(stiffness)
        return joint

    def site(
        self,
        parent,
        name,
        type="box",
        pos=np.array([0.0, 0.0, 0.0]),
        quat=np.array([0.0, 0.0, 0.0, 1.0]),
        size=np.array([0.035, 0.035, 0.035]),
        rgba=[0.1, 0.1, 0.5, 1.0],
    ):
        site = ET.SubElement(parent, "site")
        site.attrib["name"] = name
        site.attrib["type"] = type
        site.attrib["pos"] = " ".join([str(x) for x in pos])
        # Full precision — see ``body`` above.
        site.attrib["quat"] = "%.17g %.17g %.17g %.17g" % (quat[3], quat[0], quat[1], quat[2])
        site.attrib["size"] = " ".join([str(x) for x in size])
        site.attrib["rgba"] = " ".join([str(x) for x in rgba])
        return site

    def actuator(self, parent=None):
        """Return the model's single ``<actuator>`` block, creating it on demand.

        MJCF allows sibling ``<actuator>`` sections, but one merged block is
        idiomatic: every caller (notably :meth:`create_quadrotor0`, once per
        quadrotor) appends into the same element. Actuator order — and hence
        the layout of ``data.ctrl`` — is unchanged, since sibling blocks were
        concatenated in document order anyway.
        """
        if self._actuator is None:
            self._actuator = ET.SubElement(self.root if parent is None else parent, "actuator")
        return self._actuator

    def motor(
        self,
        parent,
        site,
        range=[],
        gear=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        name=None,
    ):
        """Append a ``<motor>``; ``name`` defaults to the site it acts on.

        Naming every actuator is what makes the ``data.ctrl`` layout
        *assertable*: models write ``data.ctrl`` positionally while resolving
        bodies by name, so without names nothing pins the two orders together
        (see ``tests/test_actuator_order.py``).
        """
        motor = ET.SubElement(parent, "motor")
        motor.attrib["name"] = site if name is None else name
        motor.attrib["ctrllimited"] = "true"
        motor.attrib["site"] = site
        motor.attrib["ctrlrange"] = " ".join([str(x) for x in range])
        motor.attrib["gear"] = " ".join([str(x) for x in gear])
        return motor

    def velocity(
        self,
        parent,
        site,
        range=[-1.0, 1.0],
        kv=0.1,
        gear=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ):
        velocity = ET.SubElement(parent, "velocity")
        velocity.attrib["ctrllimited"] = "true"
        velocity.attrib["site"] = site
        velocity.attrib["ctrlrange"] = " ".join([str(x) for x in range])
        velocity.attrib["gear"] = " ".join([str(x) for x in gear])
        velocity.attrib["kv"] = str(kv)
        return velocity

    def create_quadrotor0(self, parent, name, pos, **kwargs):
        """Emit a quadrotor body (chassis + 4 arm/prop pairs) and its actuators.

        Mass and inertia
        ----------------
        The geometry is fixed (0.2 m arms, 0.1 m propeller discs, a
        0.16 × 0.08 × 0.05 m chassis box), and ``inertiafromgeom`` means MuJoCo
        derives the body inertia from the geom masses. A requested ``mass``
        therefore scales **every** geom by the same factor rather than dumping
        the difference into the chassis box: at fixed geometry that is a uniform
        density change, so the body inertia scales *linearly* with the total
        mass. The alternative — a fixed rotor assembly plus a variable chassis
        — is what made ``mass=3.0`` (3.3× the default) yield only a 1.3–1.8×
        inertia, because the propeller discs at the arm tips dominate the
        tensor and they never grew.

        Two consequences worth stating out loud:

        * The emitted inertia is a property of *this geometry*, and it is
          several times :data:`udaan.core.defaults.DEFAULT_QUAD_INERTIA` (a
          smaller airframe). Backends that call ``_read_params_from_mjcf``
          adopt it, so changing ``mass`` retunes their attitude loop — by
          design, and now proportionally.
        * ``unmodeled_mass`` is *added on top* of ``mass``: it models an
          unmodelled payload, so it is deliberately not part of the scaling.

        Keyword Args:
            mass: total quadrotor mass in kg, at least :data:`_MIN_QUAD_MASS`.
                Omitted -> :data:`_DEFAULT_QUAD_TOTAL_MASS`.
            root_damping: viscous damping on the free root joint, in N·s/m and
                N·m·s/rad. Defaults to :meth:`joint`'s own ``0.1``, which is a
                *physics* term with no counterpart in any of udaan's analytical
                quadrotor models: it drags every translational and rotational
                degree of freedom of the airframe. Generators whose asset is
                meant to be compared against the analytical dynamics should
                pass ``0.0`` — see
                :func:`udaan.utils.assets.multi_quad_rigidbody`.
            root_joint_type: ``"free"`` (default) for a free-flying airframe, or
                ``"ball"`` for one pinned to its parent at its own centre of
                mass — free to rotate, carried translationally by whatever it
                is bolted to. The geometry emitted here is symmetric about the
                body origin, so that origin *is* the centre of mass and a ball
                joint there transmits force but no torque: exactly the
                frictionless spherical attachment
                :class:`~udaan.models.multi_quad_hose.MultiQuadHoseBase`
                assumes between an airframe and its hose vertex.
            quat: orientation of the chassis relative to its parent, as
                ``(x, y, z, w)``. Defaults to identity.
        """
        xtype = kwargs["xtype"] if "xtype" in kwargs else True
        alpha = kwargs["alpha"] if "alpha" in kwargs else 1.0
        rgb = kwargs["rgb"] if "rgb" in kwargs else [0.95, 0.45, 0.1]
        mass = kwargs.get("mass")
        if mass is None:
            mass_scale = 1.0
        else:
            if float(mass) < _MIN_QUAD_MASS:
                raise ValueError(
                    f"quadrotor mass {mass} must be at least {_MIN_QUAD_MASS} kg; "
                    "lighter airframes make the derived inertia rounding noise"
                )
            mass_scale = float(mass) / _DEFAULT_QUAD_TOTAL_MASS
        chassis_mass = _DEFAULT_CHASSIS_MASS * mass_scale
        arm_mass = _ARM_MASS * mass_scale
        prop_mass = _PROP_MASS * mass_scale
        unmodeled_dynamics = False
        if "unmodeled_mass" in kwargs:
            unmodeled_dynamics = True
            unmodeled_mass = kwargs["unmodeled_mass"]
        if "unmodeled_mass_loc" in kwargs:
            unmodeled_mass_loc = kwargs["unmodeled_mass_loc"]
        else:
            unmodeled_mass_loc = np.array([0.0, 0.0, 0.0])

        chassis = self.body(
            parent, name, pos, quat=np.asarray(kwargs.get("quat", [0.0, 0.0, 0.0, 1.0]))
        )
        self.box(
            chassis,
            name + "_geom",
            size=np.array([0.08, 0.04, 0.025]),
            mass=chassis_mass,
            density=1000,
            alpha=alpha,
            rgb=[0.15, 0.15, 0.15],
        )
        self.joint(
            chassis,
            name + "_root_joint",
            str(kwargs.get("root_joint_type", "free")),
            damping=float(kwargs.get("root_damping", 0.1)),
        )

        zaxis = np.array([0, 0, 1])
        l = 0.2
        rotor_arm_offset = np.array([l, 0, 0])

        rotor_angles = [0.25 * math.pi, 0.75 * math.pi, 1.25 * math.pi, 1.75 * math.pi]
        if not xtype:
            rotor_angles = [0, 0.5 * math.pi, math.pi, 1.5 * math.pi]

        for i in range(len(rotor_angles)):
            angle = rotor_angles[i]
            r = sp_rot.from_rotvec(zaxis * angle)
            rotor_arm_quat = r.as_quat()
            rotor_prop_pos = r.apply(rotor_arm_offset)
            self.box(
                chassis,
                name + "_rotor_arm_geom_%d" % i,
                quat=rotor_arm_quat,
                size=np.array([l, 0.01, 0.01]),
                mass=arm_mass,
                alpha=alpha,
                rgb=[0.4, 0.4, 0.45],
            )
            self.cylinder(
                chassis,
                name + "_rotor_prop_geom_%d" % i,
                pos=rotor_prop_pos,
                radius=0.1,
                length=0.01,
                mass=prop_mass,
                rgb=rgb,
                alpha=alpha,
            )

        if unmodeled_dynamics:
            self.box(
                chassis,
                name + "_unmodeled_mass",
                size=np.array([0.025, 0.025, 0.025]),
                mass=unmodeled_mass,
                density=1000,
                alpha=alpha,
                rgb=[0.85, 0.2, 0.2],
                pos=unmodeled_mass_loc,
            )

        # Creating actuator sites (invisible)
        self.site(
            chassis,
            name + "_end1",
            pos=np.array([0.0, 0.0, 0.0]),
            type="sphere",
            size=[0.01],
            rgba=[0, 0, 0, 0],
        )
        self.site(chassis, name + "_thrust", pos=np.array([0.0, 0.0, 0.0]), rgba=[0, 0, 0, 0])
        self.site(chassis, name + "_Mx", pos=np.array([0.0, 0.0, 0.0]), rgba=[0, 0, 0, 0])
        self.site(chassis, name + "_My", pos=np.array([0.0, 0.0, 0.0]), rgba=[0, 0, 0, 0])
        self.site(chassis, name + "_Mz", pos=np.array([0.0, 0.0, 0.0]), rgba=[0, 0, 0, 0])
        # Creating actuators
        actuator = self.actuator(self.root)
        self.motor(
            actuator,
            site=name + "_thrust",
            range=[0.0, 30.0],
            gear=np.array([0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        )
        self.motor(
            actuator,
            site=name + "_Mx",
            range=[-3.0, 3.0],
            gear=np.array([0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        )
        self.motor(
            actuator,
            site=name + "_My",
            range=[-3.0, 3.0],
            gear=np.array([0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        )
        self.motor(
            actuator,
            site=name + "_Mz",
            range=[-3.0, 3.0],
            gear=np.array([0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        )
        return chassis, actuator

    def create_cable_payload(self, parent, name, pos, length=1, mass=0.15):
        cable = self.body(parent, name, pos)
        cable_joint = self.joint(
            cable,
            "cable_quad_joint",
            "ball",
            pos=np.array([0.0, 0.0, 0.5 * length]),
            damping=0.01,
        )
        cable_geom = self.cylinder(
            cable,
            "cable",
            radius=0.005,
            length=length,
            mass=0.1,
            density=50,
            rgb=[0.2, 0.2, 0.2],
        )
        pyld = self.body(cable, "pyld", pos=np.array([0.0, 0.0, -0.5 * length]))
        pyld_geom = self.sphere(
            pyld, "pyld", radius=0.05, mass=mass, density=50, rgb=[0.0, 0.8, 0.2]
        )
        pyld_joint = self.joint(
            pyld,
            "pyld_joint",
            "hinge",
            pos=np.array([0.0, 0.0, 0]),
            range=np.array([-0.001, 0.001]),
            stiffness=1000,
            damping=1000,
            armature=0.0,
        )
        return cable

    def create_flexible_cable_payload(
        self,
        parent,
        name,
        pos=None,
        N=5,
        length=1,
        mass=0.15,
        payload_rgb=(0.0, 0.8, 0.2),
        cable_mass=0.01,
        joint_damping=1e-3,
        payload_joint_damping=1000.0,
        payload_joint_stiffness=1000.0,
        payload_radius=0.05,
        link_radius=0.005,
    ):
        """``N``-link cable + payload subtree. Child names are prefixed with ``name``.

        Geometry.  Link ``i`` is a cylinder of length ``dl = length / N``
        centred on its own body origin, hinged to its parent by a ball joint at
        its **proximal** end; link 0's joint therefore sits exactly at
        ``parent``'s origin, and the payload body sits at the distal end of
        link ``N-1``.  The chain spans ``0.5 dl + (N-1) dl + 0.5 dl = length``
        from the parent's origin to the payload centre.

        The payload's own ball joint is anchored at its centre, so a
        (rotationally isotropic) sphere spinning freely about it is exactly a
        **point mass** — which is what the analytical models suspend.  Leave
        ``payload_joint_damping``/``payload_joint_stiffness`` at their defaults
        to reproduce the historical near-welded behaviour, or pass ``0.0`` for
        a genuine free point mass.

        Args:
            parent: XML element to hang the chain from.
            name: prefix for every emitted body/joint/geom name.
            pos: accepted and ignored — link 0 is placed relative to ``parent``
                by the chain geometry, not by a caller-chosen offset.  Kept so
                existing positional callers keep working.
            N: number of links.
            length: total cable length, in metres.
            mass: payload mass, in kg.
            payload_rgb: payload colour.
            cable_mass: total mass of the cable itself, spread evenly over the
                ``N`` cylinder geoms.
            joint_damping: viscous damping on each cable ball joint.  Pass
                ``0.0`` when the asset is to be compared against an analytical
                model with no drag term — see
                :func:`udaan.utils.assets.quad_flexcable`.
            payload_joint_damping: viscous damping on the payload ball joint.
            payload_joint_stiffness: stiffness of the payload ball joint.
            payload_radius: radius of the payload sphere, in metres.
            link_radius: radius of the link cylinders, in metres.
        """
        del pos  # documented as ignored; the chain geometry places link 0
        dl = length / N
        dm = cable_mass / N
        cable = self.body(parent, f"{name}_link0", np.array([0.0, 0.0, -0.5 * dl]))
        self.joint(
            cable,
            f"{name}_joint0",
            "ball",
            pos=np.array([0.0, 0.0, 0.5 * dl]),
            damping=joint_damping,
            stiffness=0,
        )
        self.cylinder(
            cable,
            f"{name}_geom0",
            radius=link_radius,
            length=dl,
            mass=dm,
            density=50,
            rgb=[0.2, 0.2, 0.2],
        )
        for i in range(1, N):
            cable = self.body(cable, f"{name}_link{i}", np.array([0.0, 0.0, -dl]))
            self.joint(
                cable,
                f"{name}_joint{i}",
                "ball",
                pos=np.array([0.0, 0.0, 0.5 * dl]),
                damping=joint_damping,
                stiffness=0.0,
            )
            self.cylinder(
                cable,
                f"{name}_geom{i}",
                radius=link_radius,
                length=dl,
                mass=dm,
                density=50,
                rgb=[0.2, 0.2, 0.2],
            )
        pyld = self.body(cable, f"{name}_payload", pos=np.array([0.0, 0.0, -0.5 * dl]))
        self.sphere(
            pyld,
            f"{name}_payload_geom",
            radius=payload_radius,
            mass=mass,
            density=50,
            rgb=list(payload_rgb),
        )
        self.joint(
            pyld,
            f"{name}_payload_joint",
            "ball",
            pos=np.array([0.0, 0.0, 0]),
            range=np.array([-0.001, 0.001]),
            stiffness=payload_joint_stiffness,
            damping=payload_joint_damping,
            armature=0.0,
        )
        return cable
