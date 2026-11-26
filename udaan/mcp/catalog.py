"""The systems the MCP server exposes, their parameters, and argument validation.

Each :class:`System` names a scenario builder in :mod:`udaan.cli.scenarios`
(the one ``udaan run`` uses) and declares the parameters the server accepts for
it, with types, defaults and bounds. :func:`validate` turns a tool call's
arguments into builder keyword arguments, refusing anything out of range
before a model is constructed, so no call can ask for an unbounded run.

This module does not need the ``mcp`` SDK. The choice lists (trajectories,
cable models, controllers, demos) are imported from the builders, so the two
cannot disagree.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field
from typing import Any

from udaan.cli import scenarios as sc
from udaan.cli.demos import CSPAYLOAD_DEMOS, DEMOS
from udaan.core import defaults as D

#: Longest simulated duration a call may ask for, in s.
MAX_DURATION = 60.0
#: Most vehicles a multi-vehicle system may be built with.
MAX_VEHICLES = 8
#: Most links a cable or hose may be split into.
MAX_LINKS = 12

#: Parameter kinds :func:`validate` understands.
KINDS = ("int", "float", "bool", "choice", "vec3", "ints")


class ValidationError(ValueError):
    """A tool argument outside what the server accepts; the message says what is."""


@dataclass(frozen=True)
class Param:
    """One system parameter as the server accepts it.

    ``default=None`` means the model's (or the scenario's) own default, which
    ``help`` states. ``low``/``high`` bound an ``int``/``float`` (inclusive)
    and the length of ``ints``; ``choices`` lists a ``choice``'s values.
    ``backends`` limits the parameter to those backends.
    """

    name: str
    kind: str
    help: str
    default: Any = None
    low: float | None = None
    high: float | None = None
    choices: tuple[str, ...] = ()
    backends: tuple[str, ...] = ("base", "mujoco")
    #: Whether the parameter shapes the MJCF ``generate_xml`` returns. A start
    #: position, a reference knob or a value set on the loaded model does not.
    mjcf: bool = True

    def describe(self) -> dict[str, Any]:
        """JSON-serialisable description for ``list_systems``/``describe_system``."""
        out: dict[str, Any] = {"name": self.name, "type": self.kind, "help": self.help}
        out["default"] = self.default
        if self.low is not None:
            out["min"] = self.low
        if self.high is not None:
            out["max"] = self.high
        if self.choices:
            out["choices"] = list(self.choices)
        if self.backends != ("base", "mujoco"):
            out["backends"] = list(self.backends)
        if not self.mjcf:
            out["shapes_mjcf"] = False
        return out


@dataclass(frozen=True)
class System:
    """One simulated system.

    ``builder`` is the name of its function in :mod:`udaan.cli.scenarios`;
    ``None`` marks a system that only has an MJCF (``generate_xml``) and no
    closed loop to fly. ``trajectories[0]`` is the default.
    """

    id: str
    label: str
    description: str
    backends: tuple[str, ...]
    trajectories: tuple[str, ...]
    builder: str | None
    params: tuple[Param, ...] = ()
    tracked: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)
    check_reference: bool = False

    @property
    def default_backend(self) -> str:
        """``base`` where the analytical model exists, else ``mujoco``."""
        return "base" if "base" in self.backends else self.backends[0]

    def describe(self, full: bool = True) -> dict[str, Any]:
        """JSON-serialisable description; ``full=False`` drops the notes."""
        out: dict[str, Any] = {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "backends": list(self.backends),
            "default_backend": self.default_backend if self.backends else None,
            "trajectories": list(self.trajectories),
            "simulate": self.builder is not None,
            "tracked": self.tracked,
            "parameters": [p.describe() for p in self.params],
        }
        if self.check_reference:
            out["check_reference"] = True
        if full and self.notes:
            out["notes"] = list(self.notes)
        return out


def _position(help_: str) -> Param:
    return Param("position", "vec3", help_, mjcf=False)


_NUM_LINKS_CHAIN = Param(
    "num_links",
    "int",
    f"Rigid links the cable is split into (model default {D.DEFAULT_NUM_CABLE_LINKS}).",
    low=1,
    high=MAX_LINKS,
)


SYSTEMS: dict[str, System] = {
    s.id: s
    for s in (
        System(
            id="quadrotor",
            label="M1",
            description="One rigid-body quadrotor under geometric SE(3) control.",
            backends=("base", "mujoco"),
            trajectories=sc.QUADROTOR_TRAJECTORIES,
            builder="quadrotor",
            tracked="airframe position",
            params=(
                _position(
                    "Start (circle/figure-8 centre, mission take-off point) as [x, y, z] in m; "
                    "default [1, 1, 0] for hover/lissajous/mission, [1, 1, 2] otherwise."
                ),
            ),
            notes=(
                "flip, spiral, mission and figure8 set their own duration.",
                "Every trajectory but hover and step carries jerk and snap, so the "
                "flatness attitude feed-forward engages.",
            ),
        ),
        System(
            id="quad-payload",
            label="M2",
            description="One quadrotor with a cable-suspended point-mass payload "
            "(geometric cascade on SE(3) x S2).",
            backends=("base", "mujoco"),
            trajectories=sc.QUAD_PAYLOAD_TRAJECTORIES,
            builder="quad_payload",
            tracked="payload position",
            check_reference=True,
            params=(
                Param(
                    "cable_model",
                    "choice",
                    "MuJoCo cable: 'tendon' (spatial tendon, fast) or 'links' "
                    "(rigid link chain with ball joints).",
                    default="tendon",
                    choices=sc.CABLE_MODELS,
                    backends=("mujoco",),
                ),
                Param(
                    "cable_length",
                    "float",
                    f"Cable length in m (model default {D.DEFAULT_M2_CABLE_LENGTH:g}); "
                    "the link chain accepts only its MJCF's length. Set on the loaded model, "
                    "not in the MJCF.",
                    low=0.1,
                    high=5.0,
                    mjcf=False,
                ),
                _position(
                    "Payload start (circle centre, helix start) as [x, y, z] in m; "
                    "default [-1, 2, 0.5] for hover, [0, 0, 1.5] otherwise."
                ),
            ),
            notes=(
                "spiral is a horizontal helix (0.8 m radius, 2 m/s, two loops climbing 1 m) "
                "and sets its own duration.",
                "circle is 1 m radius at 1 m/s after a 2 s ramp from rest.",
            ),
        ),
        System(
            id="multi-quad-payload",
            label="M3",
            description="nQ quadrotors carrying one cable-suspended point-mass payload "
            "(geometric cascade, ring tension split).",
            backends=("base", "mujoco"),
            trajectories=("hover",),
            builder="multi_quad_payload",
            tracked="payload position",
            params=(
                Param(
                    "num_quadrotors",
                    "int",
                    "Quadrotors in the ring.",
                    default=D.DEFAULT_NUM_QUADROTORS,
                    low=1,
                    high=MAX_VEHICLES,
                ),
                Param("cable_length", "float", "Taut cable length in m.", low=0.2, high=5.0),
                Param("payload_mass", "float", "Payload mass in kg.", low=0.01, high=10.0),
                Param(
                    "formation_radius",
                    "float",
                    f"Ring radius in m (default {D.DEFAULT_FORMATION_RADIUS:g}, grown with nQ "
                    "so neighbouring airframes clear each other).",
                    low=0.05,
                    high=5.0,
                ),
                _position("Payload start as [x, y, z] in m; default [-1, 2, 0.5]."),
            ),
            notes=("'hover' flies the payload to the controller's own setpoint [0, 0, 1].",),
        ),
        System(
            id="multi-quad-rigidbody",
            label="M4",
            description="nQ quadrotors carrying one rigid-body payload "
            "(geometric cascade, wrench allocation).",
            backends=("base", "mujoco"),
            trajectories=("hover",),
            builder="multi_quad_rigidbody",
            tracked="payload position",
            params=(
                Param(
                    "num_quadrotors",
                    "int",
                    "Quadrotors around the payload; payload attitude is controllable from 3.",
                    default=4,
                    low=1,
                    high=MAX_VEHICLES,
                ),
                Param("cable_length", "float", "Taut cable length in m.", low=0.2, high=5.0),
                Param("payload_mass", "float", "Payload mass in kg.", low=0.01, high=20.0),
                Param(
                    "attachment_radius",
                    "float",
                    "Radius of the attachment ring on the payload, m.",
                    low=0.01,
                    high=3.0,
                    backends=("mujoco",),
                ),
                Param(
                    "formation_radius",
                    "float",
                    "Horizontal offset of each quadrotor from its attachment point, m.",
                    low=0.0,
                    high=5.0,
                ),
                Param(
                    "spawn_height",
                    "float",
                    "Height of the quadrotor ring in the generated MJCF, m.",
                    low=0.5,
                    high=10.0,
                    backends=("mujoco",),
                ),
                _position(
                    "Payload start as [x, y, z] in m; default the MJCF spawn (MuJoCo) "
                    "or [0, 0, 0.5] (base)."
                ),
            ),
            notes=("'hover' flies the payload to the controller's own setpoint [0, 0, 1].",),
        ),
        System(
            id="quad-flexcable",
            label="M5",
            description="One quadrotor with a payload on an n-link flexible cable "
            "(infinite-horizon LQR with flat-map feed-forward).",
            backends=("base", "mujoco"),
            trajectories=sc.CHAIN_TRAJECTORIES,
            builder="quad_flexcable",
            tracked="payload position",
            params=(
                _NUM_LINKS_CHAIN,
                Param("cable_length", "float", "Total taut cable length in m.", low=0.1, high=5.0),
                Param("cable_mass", "float", "Cable mass in kg.", low=0.001, high=5.0),
                Param("payload_mass", "float", "Payload mass in kg.", low=0.01, high=5.0),
                _position("Payload start as [x, y, z] in m; default [0, 0, 1]."),
            ),
            notes=(
                "'hover' is a step: the payload is commanded [0.5, 0.3, 0.2] m from its start.",
                "The analytical model warns when the chain is stiff enough to overflow; "
                "past about 5 links prefer the MuJoCo backend.",
            ),
        ),
        System(
            id="hose",
            label="M6",
            description="Several quadrotors in series on one n-link flexible hose "
            "(hose LQR, or per-vehicle station-keeping PDs).",
            backends=("base", "mujoco"),
            trajectories=sc.CHAIN_TRAJECTORIES,
            builder="hose",
            tracked="hose root (vertex 0) with 'lqr'; every vehicle with 'station'",
            params=(
                Param(
                    "num_links",
                    "int",
                    "Links the hose is split into.",
                    default=D.DEFAULT_NUM_HOSE_LINKS,
                    low=1,
                    high=MAX_LINKS,
                ),
                Param(
                    "attachments",
                    "ints",
                    "Vertex indices carrying a quadrotor, e.g. [0, 2, 4]; default both ends.",
                    low=1,
                    high=MAX_VEHICLES,
                ),
                Param("hose_length", "float", "Total hose length in m.", low=0.2, high=10.0),
                Param("hose_mass", "float", "Total hose mass in kg.", low=0.001, high=10.0),
                Param("quad_mass", "float", "Mass of each airframe in kg.", low=0.1, high=5.0),
                Param(
                    "sag_ratio",
                    "float",
                    "Spawn span of each hose segment as a fraction of its length.",
                    low=0.05,
                    high=1.0,
                ),
                Param(
                    "spawn_height",
                    "float",
                    "Height of vertex 0 at spawn, m.",
                    low=0.5,
                    high=10.0,
                    backends=("mujoco",),
                ),
                Param(
                    "controller",
                    "choice",
                    "'lqr' (needs a vehicle on the last vertex) or 'station' (any pattern).",
                    default="lqr",
                    mjcf=False,
                    choices=sc.HOSE_CONTROLLERS,
                ),
                _position(
                    "Vertex-0 start as [x, y, z] in m; default the MJCF spawn (MuJoCo) "
                    "or [0, 0, 2] (base)."
                ),
            ),
            notes=("'hover' is a step: the root is commanded [0.3, 0.2, 0.2] m from its start.",),
        ),
        System(
            id="tethered-quad",
            label="M7",
            description="One quadrotor tethered to a ground anchor by an n-link cable "
            "(hose LQR carried round the anchor).",
            backends=("base", "mujoco"),
            trajectories=sc.TETHER_TRAJECTORIES,
            builder="tethered_quad",
            tracked="top vehicle position",
            params=(
                Param(
                    "num_links",
                    "int",
                    f"Tether links (default {D.DEFAULT_TETHER_NUM_LINKS}).",
                    low=1,
                    high=MAX_LINKS,
                ),
                Param(
                    "length",
                    "float",
                    f"Tether length in m (default {D.DEFAULT_TETHER_LENGTH:g}).",
                    low=0.2,
                    high=10.0,
                ),
                Param("tether_mass", "float", "Tether mass in kg.", low=0.001, high=10.0),
                Param("anchor_height", "float", "Anchor height in m.", low=0.0, high=10.0),
                Param(
                    "standoff",
                    "float",
                    "Horizontal distance from the anchor, as a fraction of the length.",
                    low=0.0,
                    high=1.0,
                    mjcf=False,
                ),
                Param(
                    "height",
                    "float",
                    "Height above the anchor, as a fraction of the length.",
                    low=0.0,
                    high=1.0,
                    mjcf=False,
                ),
                Param(
                    "rate",
                    "float",
                    "Circle rate in rad/s; default the largest the rule allows.",
                    low=0.01,
                    high=10.0,
                    mjcf=False,
                ),
                Param(
                    "ramp",
                    "float",
                    f"Circle ramp / climb blend duration in s (default "
                    f"{D.DEFAULT_TETHER_RAMP_TIME:g}).",
                    low=0.1,
                    high=MAX_DURATION,
                    mjcf=False,
                ),
            ),
        ),
        System(
            id="tethered-hose",
            label="M8",
            description="Several quadrotors in series on one hose pinned to a ground anchor "
            "(hose LQR).",
            backends=("base", "mujoco"),
            trajectories=sc.TETHER_TRAJECTORIES,
            builder="tethered_hose",
            tracked="top vehicle position",
            params=(
                Param(
                    "num_quadrotors",
                    "int",
                    f"Vehicles on the hose (default {D.DEFAULT_TETHERED_HOSE_NUM_QUADROTORS}).",
                    low=1,
                    high=MAX_VEHICLES,
                ),
                Param(
                    "num_links",
                    "int",
                    "Hose links (default two per vehicle).",
                    low=1,
                    high=MAX_LINKS,
                ),
                Param(
                    "length",
                    "float",
                    f"Hose length in m (default {D.DEFAULT_TETHERED_HOSE_LENGTH:g}).",
                    low=0.2,
                    high=10.0,
                ),
                Param("hose_mass", "float", "Hose mass in kg.", low=0.001, high=10.0),
                Param("anchor_height", "float", "Anchor height in m.", low=0.0, high=10.0),
                Param(
                    "standoff",
                    "float",
                    "Top vehicle's distance from the anchor, fraction of the length.",
                    low=0.0,
                    high=1.0,
                    mjcf=False,
                ),
                Param(
                    "height",
                    "float",
                    "Top vehicle's height above the anchor, fraction of the length.",
                    low=0.0,
                    high=1.0,
                    mjcf=False,
                ),
                Param(
                    "rate",
                    "float",
                    "Circle rate in rad/s; default the largest the rule allows.",
                    low=0.01,
                    high=10.0,
                    mjcf=False,
                ),
                Param(
                    "ramp",
                    "float",
                    "Circle ramp / climb blend duration in s.",
                    low=0.1,
                    high=MAX_DURATION,
                    mjcf=False,
                ),
            ),
        ),
        System(
            id="multi-quad",
            label="multi-quad",
            description="MultiQuadrotorCSPointmass: quadrotors station-keeping a "
            "cable-suspended point-mass payload (MuJoCo-only, PD per vehicle).",
            backends=("mujoco",),
            trajectories=("hover",),
            builder="multi_quad",
            tracked="every quadrotor (each holds its spawn)",
            params=(
                Param(
                    "num_quadrotors",
                    "int",
                    "Quadrotors.",
                    default=3,
                    low=1,
                    high=MAX_VEHICLES,
                ),
                _position(
                    "Offset added to every body's spawn, [x, y, z] in m; default [-1, 2, 0.5]."
                ),
            ),
        ),
        System(
            id="multi-quad-rigid",
            label="multi-quad-rigid",
            description="MultiQuadRigidbody: four quadrotors station-keeping a rigid-body "
            "payload (MuJoCo-only, PD per vehicle).",
            backends=("mujoco",),
            trajectories=("hover",),
            builder="multi_quad_rigid",
            tracked="every quadrotor (each holds its constructed position)",
            params=(
                _position(
                    "Offset added to every body's spawn, [x, y, z] in m; default [0, 0, 0.5]."
                ),
            ),
        ),
        System(
            id="fleet",
            label="fleet",
            description="QuadrotorFleet: independent quadrotors side by side, each with "
            "its own controllers (MuJoCo-only).",
            backends=("mujoco",),
            trajectories=("hover",),
            builder="fleet",
            tracked="every quadrotor",
            params=(
                Param(
                    "num_quadrotors",
                    "int",
                    "Quadrotors (ignored by a demo, which fixes its own count).",
                    default=3,
                    low=1,
                    high=MAX_VEHICLES,
                ),
                Param(
                    "demo",
                    "choice",
                    "Preset: 'l1-comparison' (PD clean / PD disturbed / L1 disturbed) or "
                    "'gain-sweep' (PD from soft to aggressive). Default: all fly to [0, 0, 2].",
                    choices=tuple(DEMOS),
                ),
                _position("First quadrotor's start, [x, y, z] in m; default [0, 0, 1]."),
            ),
        ),
        System(
            id="cspayload-fleet",
            label="cspayload-fleet",
            description="QuadrotorCsPayloadFleet: quadrotor-payload agents side by side on "
            "link-chain cables (MuJoCo-only).",
            backends=("mujoco",),
            trajectories=("hover",),
            builder="cspayload_fleet",
            tracked="every agent's payload",
            params=(
                Param(
                    "num_agents",
                    "int",
                    "Agents (ignored by a demo).",
                    default=2,
                    low=1,
                    high=MAX_VEHICLES,
                ),
                Param(
                    "demo",
                    "choice",
                    "Preset: 'same-gains' or 'gain-sweep' (cable kp/kd x 0.5..2). "
                    "Default: payloads from [+-1, +-1, 0.5] to [0, 0, 1.5].",
                    choices=tuple(CSPAYLOAD_DEMOS),
                ),
                Param(
                    "cable_links",
                    "int",
                    "Cable discretisation (links).",
                    default=10,
                    low=1,
                    high=MAX_LINKS,
                ),
                Param(
                    "same_start",
                    "bool",
                    "Every agent on one absolute start (demo only).",
                    default=False,
                    mjcf=False,
                ),
            ),
        ),
        System(
            id="comparison",
            label="comparison",
            description="QuadrotorComparison: a plant and a reference quadrotor in one "
            "MuJoCo scene. MJCF only; it has no closed loop to simulate.",
            backends=("mujoco",),
            trajectories=(),
            builder=None,
        ),
    )
}


def get_system(system: str) -> System:
    """The :class:`System` named ``system``; raises :class:`ValidationError` naming the choices."""
    try:
        return SYSTEMS[system]
    except KeyError:
        raise ValidationError(
            f"Unknown system '{system}'. Expected one of: {', '.join(SYSTEMS)}."
        ) from None


def _check_number(p: Param, value: Any) -> float | int:
    if p.kind == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            else:
                raise ValidationError(f"{p.name} must be an integer, got {value!r}.")
    else:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValidationError(f"{p.name} must be a number, got {value!r}.")
        value = float(value)
        if not math.isfinite(value):
            raise ValidationError(f"{p.name} must be finite, got {value!r}.")
    if p.low is not None and value < p.low:
        raise ValidationError(f"{p.name} must be >= {p.low:g}, got {value!r}.")
    if p.high is not None and value > p.high:
        raise ValidationError(f"{p.name} must be <= {p.high:g}, got {value!r}.")
    number: float | int = value
    return number


def _check_value(p: Param, value: Any) -> Any:
    if p.kind in ("int", "float"):
        return _check_number(p, value)
    if p.kind == "bool":
        if not isinstance(value, bool):
            raise ValidationError(f"{p.name} must be true or false, got {value!r}.")
        return value
    if p.kind == "choice":
        if value not in p.choices:
            raise ValidationError(
                f"{p.name} must be one of: {', '.join(p.choices)}; got {value!r}."
            )
        return value
    if p.kind == "vec3":
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValidationError(f"{p.name} must be [x, y, z], got {value!r}.")
        out = [_check_number(Param(p.name, "float", ""), v) for v in value]
        if any(abs(v) > 100.0 for v in out):
            raise ValidationError(f"{p.name} components must lie within +-100 m, got {value!r}.")
        return out
    if p.kind == "ints":
        if not isinstance(value, (list, tuple)):
            raise ValidationError(f"{p.name} must be a list of integers, got {value!r}.")
        if p.low is not None and len(value) < p.low or p.high is not None and len(value) > p.high:
            raise ValidationError(
                f"{p.name} must list {p.low:g} to {p.high:g} entries, got {len(value)}."
            )
        return [_check_number(Param(p.name, "int", "", low=0, high=MAX_LINKS), v) for v in value]
    raise AssertionError(f"unknown parameter kind {p.kind!r}")  # pragma: no cover


def validate(
    system: str,
    backend: str | None = None,
    trajectory: str | None = None,
    duration: float | None = None,
    params: dict[str, Any] | None = None,
) -> tuple[System, dict[str, Any]]:
    """Check a call's arguments and return the system with its builder keyword arguments.

    ``backend``/``trajectory`` default to the system's own; ``duration`` to
    the builder's default (the CLI's), bounded by :data:`MAX_DURATION`. Unknown parameter names, a
    parameter the chosen backend does not take, and any out-of-range value
    are refused with :class:`ValidationError`. The ``vfx`` backend is never
    accepted.
    """
    spec = get_system(system)
    if spec.builder is None:
        raise ValidationError(
            f"System '{system}' has an MJCF only and cannot be simulated; use generate_xml."
        )
    backend = spec.default_backend if backend is None else backend
    if backend not in spec.backends:
        hint = " (vfx is never offered: it blocks)" if backend == "vfx" else ""
        raise ValidationError(
            f"System '{system}' runs on: {', '.join(spec.backends)}; got backend '{backend}'{hint}."
        )
    trajectory = spec.trajectories[0] if trajectory is None else trajectory
    if trajectory not in spec.trajectories:
        raise ValidationError(
            f"System '{system}' flies: {', '.join(spec.trajectories)}; got trajectory "
            f"'{trajectory}'."
        )
    if duration is None:
        duration = _builder_signature(spec).parameters["duration"].default
    duration = _check_number(Param("duration", "float", "", low=0.0, high=MAX_DURATION), duration)
    if duration <= 0.0:
        raise ValidationError(f"duration must be > 0, got {duration!r}.")

    known = {p.name: p for p in spec.params}
    kwargs: dict[str, Any] = {}
    for name, value in (params or {}).items():
        p = known.get(name)
        if p is None:
            accepted = ", ".join(known) or "none"
            raise ValidationError(
                f"System '{system}' has no parameter '{name}'. Accepted: {accepted}."
            )
        if value is None:
            continue
        if backend not in p.backends:
            raise ValidationError(
                f"Parameter '{name}' applies only to backend {', '.join(p.backends)}."
            )
        kwargs[name] = _check_value(p, value)

    if "backend" in _builder_params(spec):
        kwargs["backend"] = backend
    if len(spec.trajectories) > 1:
        kwargs["trajectory"] = trajectory
    kwargs["duration"] = duration
    return spec, kwargs


def _builder_signature(spec: System) -> inspect.Signature:
    """Signature of the system's scenario builder."""
    assert spec.builder is not None
    return inspect.signature(getattr(sc, spec.builder))


def _builder_params(spec: System) -> set[str]:
    """Keyword names of the system's scenario builder."""
    return set(_builder_signature(spec).parameters)
