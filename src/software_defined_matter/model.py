"""Data model for additively manufactured parts in the Emergent Matter framework.

Hierarchy
---------
  Param            - a scalar design variable (free or fixed)
  MaterialRegion   - material identity + SDF sub-tree
  Port     - interface port in local part coordinates
  Objective        - symbolic expression to minimise/maximise
  Constraint       - symbolic expression compared to an RHS
  Part             - top-level object: params + materials +
                     ports + objectives + constraints + metadata

All objects are JSON-serializable via to_dict() / from_dict() and can be
saved to / loaded from a ``.sdm`` file (JSON on the wire).

SDF trees and expression trees are stored as nested dicts (DSL expression
trees). The SDF tree is evaluated by ``software_defined_matter.sdf.compile``; the
expression tree is evaluated by ``software_defined_matter.dsl.expr``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Literal

from software_defined_matter.ports import Port, scalar_refs

# The newest schema_version this package emits and validates against by
# default. Distinct from the package's own version (0.3.0 today).
LATEST_SCHEMA_VERSION = "0.6"

# Fallback when a document's schema_version is missing or unrecognised.
# 0.1's `schema_version` property is a permissive pattern rather than a
# `const`, so a well-formed unknown version still validates structurally
# against it (see io._schema_version_for).
FALLBACK_SCHEMA_VERSION = "0.1"

# Every version this package can load and validate. Absent `prior` /
# `tolerance` means a delta at the authored value, so 0.1 and 0.2 documents
# load unchanged; 0.3 is a pure superset.
KNOWN_SCHEMA_VERSIONS = ("0.1", "0.2", "0.3", "0.4", "0.5", "0.6")

# Capability -> the schema_version that introduced it; min_schema_version_for
# takes the newest entry among the capabilities a part actually uses.
# "kinematics" is a compatibility floor, independent of block presence.
_CAPABILITY_MIN_VERSION: dict[str, str] = {
    "kinematics": "0.2",
    "ports": "0.5",
    "trig_expr": "0.5",
    "raster_field": "0.3",
    "axis_ramp": "0.3",
    "radial_hermite": "0.4",
    # Generated schemas enumerate deform names. A reader needs the first
    # released schema whose vocabulary includes this warp.
    "shear_linear": "0.6",
    "taper_linear": "0.6",
    "scale_axis": "0.6",
    "param_expr": "0.3",
    "prior_params": "0.3",
}


def _parse_version(sv: str) -> tuple[int, int]:
    """Parse ``"MAJOR.MINOR"`` into a numeric tuple for ordering comparisons.

    A bare string compare puts ``"0.10"`` before ``"0.2"``; this makes the
    comparison numeric so schema_version ordering survives a two-digit minor.

    Raises:
        ValueError: If ``sv`` is not two dot-separated non-negative integers.
    """
    parts = sv.split(".")
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        raise ValueError(f"Malformed schema version {sv!r}; expected 'MAJOR.MINOR'.")
    return (int(parts[0]), int(parts[1]))


def supports_version(sv: str) -> bool:
    """True if this package can read a document declaring ``schema_version=sv``.

    Membership in :data:`KNOWN_SCHEMA_VERSIONS`, not a comparison against
    :data:`LATEST_SCHEMA_VERSION`: "supported" means a schema file exists to
    validate the document against. That is the same test
    ``io._schema_version_for`` dispatches on, so the two cannot disagree. A
    ceiling comparison would call an invented ``"0.15"`` supported while
    :func:`software_defined_matter.io.validate` quietly checked it against
    the ``0.1`` fallback schema, which is the version confusion this
    contract exists to close; it would also accept a below-floor ``"0.0"``
    that no schema file has ever covered.

    A malformed version string is simply unsupported, not an error. Callers
    that need a raised error should use :func:`assert_supported`.
    """
    return sv in KNOWN_SCHEMA_VERSIONS


def assert_supported(doc: dict[str, Any]) -> None:
    """Raise ``ValueError`` unless ``doc`` declares a version this package supports.

    Names the document's declared version and every version this package
    knows (:data:`KNOWN_SCHEMA_VERSIONS`). A separate, opt-in strict check:
    not called from :func:`software_defined_matter.io.validate`, whose
    policy is to warn and load unknown versions on a best-effort basis.
    """
    sv = doc.get("schema_version")
    if sv is None or not supports_version(sv):
        raise ValueError(
            f".sdm document declares schema_version={sv!r}, which this "
            f"package cannot read. Known versions: {list(KNOWN_SCHEMA_VERSIONS)}."
        )


Number = int | float
Vector3 = tuple[Number, Number, Number]
# A DSL tree is any nested dict/list/scalar produced by the builders below.
SDFTree = dict[str, Any]
ExprTree = dict[str, Any]

# Controlled unit vocabulary for Param.unit. The .sdm wire format REQUIRES a
# unit from this set on every param (schema-enforced); in-memory Params may
# use "" while under construction, but a non-empty unit must come from here:
# free-form strings drift ("milimeter", "MM") and silently break downstream
# consumers that dispatch on unit. Geometry is authored in mm by org
# convention. Grow the set by PR, and mirror any addition in the schema enum.
ALLOWED_UNITS = frozenset(
    {
        # geometry
        "mm",
        "mm^2",
        "mm^3",
        # angle
        "rad",
        "deg",
        # discrete / dimensionless
        "count",
        "ratio",
        # mass / density
        "g",
        "kg",
        "g/mm^3",
        "kg/m^3",
        # force / stiffness / torque
        "N",
        "N/mm",
        "N*mm",
        "N*mm/rad",
        # stress / modulus
        "MPa",
        "GPa",
        # time / frequency
        "s",
        "Hz",
    }
)


# Supported Param.prior distributions and the extra keys each one requires
# (beyond "dist"). Distributions are centred on / anchored to the authored
# ``value``: the prior describes real uncertainty AROUND the nominal (SLS
# shrinkage, as-printed clearance), it does not replace the nominal. Grow the
# set by PR and mirror any addition in the 0.3 schema's ``prior`` oneOf.
PRIOR_DISTS: dict[str, tuple[str, ...]] = {
    "delta": (),  # point mass at value (the default)
    "uniform": ("lo", "hi"),  # uniform on [lo, hi] (must contain value)
    "normal": ("sigma",),  # Normal(value, sigma), sigma > 0
    "uniform_pm": ("half_width",),  # uniform on value +/- half_width
}


def _validate_prior(param_name: str, value: Number, prior: dict[str, Any]) -> None:
    """Raise ``ValueError`` unless ``prior`` is a well-formed distribution spec."""
    if not isinstance(prior, dict):
        # TRY004 is suppressed below: ValueError, not TypeError, is the
        # documented contract --
        # this function's docstring says so, and test_prior_must_be_a_dict
        # asserts it. TypeError is not a subclass of ValueError, so switching
        # would break both the test and every caller catching ValueError.
        raise ValueError(  # noqa: TRY004
            f"Param {param_name!r}: prior must be a dict like "
            f'{{"dist": "normal", "sigma": 0.1}}, got {type(prior).__name__}'
        )
    dist = prior.get("dist")
    if dist not in PRIOR_DISTS:
        raise ValueError(
            f"Param {param_name!r}: unknown prior dist {dist!r}. Supported: {sorted(PRIOR_DISTS)}."
        )
    required = PRIOR_DISTS[dist]
    missing = [k for k in required if k not in prior]
    if missing:
        raise ValueError(
            f"Param {param_name!r}: prior dist {dist!r} requires keys "
            f"{list(required)}, missing {missing}."
        )
    extra = sorted(set(prior) - {"dist", *required})
    if extra:
        raise ValueError(
            f"Param {param_name!r}: prior dist {dist!r} accepts only keys "
            f"{['dist', *required]}, got unexpected {extra}."
        )
    for k in required:
        v = prior[k]
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            # See above: ValueError is the contract, asserted by
            # test_bad_priors_raise's "must be a number" case.
            raise ValueError(  # noqa: TRY004
                f"Param {param_name!r}: prior key {k!r} must be a number, got {v!r}."
            )
    if dist == "normal" and prior["sigma"] <= 0:
        raise ValueError(
            f"Param {param_name!r}: normal prior requires sigma > 0, got {prior['sigma']!r}."
        )
    if dist == "uniform_pm" and prior["half_width"] <= 0:
        raise ValueError(
            f"Param {param_name!r}: uniform_pm prior requires half_width > 0, "
            f"got {prior['half_width']!r}."
        )
    if dist == "uniform":
        lo, hi = prior["lo"], prior["hi"]
        if not lo < hi:
            raise ValueError(
                f"Param {param_name!r}: uniform prior requires lo < hi, got lo={lo!r}, hi={hi!r}."
            )
        if not lo <= value <= hi:
            raise ValueError(
                f"Param {param_name!r}: uniform prior [{lo}, {hi}] excludes "
                f"the authored value {value!r}; the nominal must have "
                f"support (a prior that rules out its own nominal is almost "
                f"certainly an authoring bug)."
            )


# ===========================================================================
# Param
# ===========================================================================


@dataclass
class Param:
    """A scalar design variable.

    Attributes:
        name (str): Identifier used as key in ``Part.params`` and in DSL
            ``$ref`` leaves.
        value (float): Current numeric value.
        free (bool): If ``True``, this variable is an optimisation target
            and its value flows into the SDF / expression closures via the
            free-parameter vector.
        bounds (tuple[float, float] | None): ``(lower, upper)`` optimisation
            bounds. ``None`` means unbounded.
        unit (str): Physical unit from :data:`ALLOWED_UNITS` (e.g. ``"mm"``,
            ``"rad"``, ``"count"``). ``""`` is tolerated in-memory but the
            ``.sdm`` schema requires a real unit on every param -- a typo
            raises immediately here rather than surfacing as a
            viewer/optimiser mis-scale later.
        ui (dict | None): Optional exploration/presentation spec consumed by
            viewers and the GLSL emitter's control manifest; optimisers
            ignore it entirely. Recognised keys (all optional):

            - ``step`` (number > 0) -- slider resolution
            - ``explore_bounds`` ([lo, hi]) -- scrub range for exploration when it
              should differ from the *optimiser* box in ``bounds`` (viewers
              default to ``bounds`` when absent)
            - ``group`` (str) / ``order`` (number) -- panel grouping and sort key
            - ``role`` ("topology" | "pose") -- authored hint about what KIND of
              variable this is.

              ``"topology"`` -- the param changes the number of nodes the authoring
              script emits (loop/stage counts), which no emitter can infer from the
              stamped-out tree; viewers must rebuild rather than scrub.

              ``"pose"`` -- the param is a kinematic DOF / inspection override, not a
              design variable: transient by design, and an optimiser has no business
              touching it. Consumers rely on this: a downstream viewer project
              PRESERVES pose controls across a rebuild refresh (otherwise a rebuild
              snaps the part back to rest, which is indistinguishable from a
              broken control), and
              ``sdf.bbox`` widens a bounds-mode bbox to the pose envelope via
              ``explore_bounds`` so AABB pruning does not amputate deflected
              geometry. Pose params are typically ``free=False`` with
              ``explore_bounds`` rather than optimiser ``bounds``.
            - ``rebuild`` (bool) -- commit edits through the authoring rebuild
              because this value is baked into generated output.
            - ``axis`` ("radial" | "axial" | "tangential") -- physical
              direction used by viewers while a rebuild is pending.
            - ``collapsed`` (bool) -- the param's panel group starts collapsed
              (composed CEMs pass a dependency's full control set up, folded)
            - ``driven`` (bool) -- the value is derived by a relation from other
              params (consumer CEM driving a dependency's dimensions); viewers
              show it read-only until an explicit override
            - ``choices`` (list[str]) -- the param is a PRESET SELECTOR: its
              value is an integer index into these labels (airfoil family,
              infill type). Viewers render the label; the generator maps the
              index to geometry on rebuild
        prior (dict | None): Optional probability distribution describing the REAL uncertainty
            around ``value`` -- SLS shrinkage, PA12 modulus scatter, as-printed
            clearance. This is what turns a ``.sdm`` into a *generative
            model*: forward Monte Carlo through the compiled closures
            (:mod:`software_defined_matter.sample`) is one consumer, and a
            separate physics/verification project's chance-constrained
            referees (not part of this repo) are another. Supported
            specs (see :data:`PRIOR_DISTS`):

            - ``{"dist": "delta"}`` -- point mass at ``value``. The DEFAULT:
              an absent ``prior`` means exactly this, so every pre-0.3 file is
              a valid degenerate generative model. An explicit delta is
              normalised back to ``None`` on construction (absent IS the
              canonical spelling), so round-trips stay byte-identical.
            - ``{"dist": "uniform", "lo": ..., "hi": ...}`` -- uniform on
              ``[lo, hi]``; must contain ``value``.
            - ``{"dist": "normal", "sigma": ...}`` -- Normal centred at
              ``value`` with ``sigma > 0`` (same unit as the param).
            - ``{"dist": "uniform_pm", "half_width": ...}`` -- uniform on
              ``value ± half_width``, ``half_width > 0``. A hard ± band is a
              defensible shape for post-compensation process residual
              (Bayesian ≠ Gaussian).

            Unknown ``dist`` names, missing/extra keys, or non-positive scale
            parameters raise ``ValueError`` at construction.
        tolerance (float | None): Shorthand for
            ``prior={"dist": "uniform_pm", "half_width": tolerance}``
            (must be ``> 0``; same unit as the param). Mutually exclusive with
            ``prior`` -- authoring both is ambiguous and raises. Kept as its own
            field (not normalised into ``prior``) so the author's spelling
            survives a round-trip; :meth:`authored_prior` is the normalised view.

            Precedence when a manufacturing process profile is in play
            (``metadata["process_profile"]``, resolved by
            :mod:`software_defined_matter.process`): an authored ``prior`` /
            ``tolerance`` always WINS over the profile-derived default -- the
            profile only fills in params the author said nothing about.
    """

    name: str
    value: Number | None = None
    free: bool = False
    bounds: tuple[Number, Number] | None = None
    unit: str = ""
    ui: dict[str, Any] | None = None
    prior: dict[str, Any] | None = None
    tolerance: Number | None = None
    expr: ExprTree | None = None

    def __post_init__(self) -> None:
        if self.expr is not None and self.free:
            raise ValueError(
                f"Param {self.name!r}: expr and free=True are mutually exclusive; "
                "a derived value is not an independent optimisation variable."
            )
        if self.expr is None and self.value is None:
            raise ValueError(f"Param {self.name!r}: value is required unless expr is provided.")
        if self.unit and self.unit not in ALLOWED_UNITS:
            raise ValueError(
                f"Param {self.name!r}: unknown unit {self.unit!r}. "
                f"Allowed: {sorted(ALLOWED_UNITS)}. "
                f"Extend ALLOWED_UNITS (and the schema enum) by PR if a new "
                f"unit is genuinely needed."
            )
        if self.prior is not None and self.tolerance is not None:
            raise ValueError(
                f"Param {self.name!r}: both prior and tolerance are set; "
                f"tolerance IS a prior (uniform_pm shorthand), so authoring "
                f"both is ambiguous. Keep one."
            )
        if self.tolerance is not None and (
            not isinstance(self.tolerance, (int, float))
            or isinstance(self.tolerance, bool)
            or self.tolerance <= 0
        ):
            raise ValueError(
                f"Param {self.name!r}: tolerance must be a number > 0 "
                f"(a ± half-width), got {self.tolerance!r}."
            )
        if self.prior is not None:
            if self.value is None:
                raise ValueError(f"Param {self.name!r}: a prior requires a cached numeric value.")
            _validate_prior(self.name, self.value, self.prior)
            if self.prior.get("dist") == "delta":
                # Absent-means-delta is the canonical spelling: normalising
                # here keeps to_dict/from_dict round-trips byte-identical and
                # keeps delta-only parts on the pre-0.3 wire version.
                self.prior = None

    def authored_prior(self) -> dict[str, Any]:
        """The author's prior as a normalised distribution spec.

        ``tolerance`` becomes its ``uniform_pm`` equivalent; nothing authored
        becomes ``{"dist": "delta"}``. This is the *authored* layer only: a
        manufacturing process profile never appears here; see
        :func:`software_defined_matter.process.effective_prior` for the full
        precedence (authored > profile-derived > delta).
        """
        if self.prior is not None:
            return dict(self.prior)
        if self.tolerance is not None:
            return {"dist": "uniform_pm", "half_width": float(self.tolerance)}
        return {"dist": "delta"}

    def has_authored_prior(self) -> bool:
        """True when the author declared a NON-delta prior (or tolerance).

        This is the per-param predicate behind the ``"prior_params"``
        capability in :func:`min_schema_version_for`: a part needs schema 0.3
        iff any param answers True (an explicit delta was normalised away in
        ``__post_init__``, so it does not count, and does not churn the file
        to 0.3).
        """
        return self.prior is not None or self.tolerance is not None

    @property
    def derived(self) -> bool:
        """Whether this parameter's value is defined by a relation."""
        return self.expr is not None

    def is_driven(self) -> bool:
        """Whether a UI should present this parameter as a read-only result."""
        return self.derived or bool((self.ui or {}).get("driven", False))

    def numeric_value(self) -> Number:
        """Return the cached scalar value.

        Raises:
            ValueError: When no cached value is available (typical for
                unevaluated relation-defined params).
        """
        if self.value is None:
            raise ValueError(f"Param {self.name!r}: no cached numeric value.")
        return self.value

    def to_dict(self) -> dict[str, Any]:
        if self.value is None:
            raise ValueError(
                f"Param {self.name!r}: no cached value is available for serialisation."
            )
        d: dict[str, Any] = {
            "name": self.name,
            "value": self.value,
            "free": self.free,
            "bounds": list(self.bounds) if self.bounds is not None else None,
            "unit": self.unit,
        }
        if self.ui is not None:
            d["ui"] = self.ui
        if self.prior is not None:
            d["prior"] = copy.deepcopy(self.prior)
        if self.tolerance is not None:
            d["tolerance"] = self.tolerance
        if self.expr is not None:
            d["expr"] = copy.deepcopy(self.expr)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Param:
        bounds = tuple(d["bounds"]) if d.get("bounds") is not None else None
        return cls(
            name=d["name"],
            value=d["value"],
            free=d.get("free", False),
            bounds=bounds,
            unit=d.get("unit", ""),
            ui=d.get("ui"),
            prior=d.get("prior"),
            tolerance=d.get("tolerance"),
            expr=d.get("expr"),
        )

    def __repr__(self) -> str:
        flag = " [FREE]" if self.free else ""
        bnd = f" bounds={self.bounds}" if self.bounds else ""
        pri = ""
        if self.prior is not None or self.tolerance is not None:
            ap = self.authored_prior()
            args = ", ".join(f"{k}={v}" for k, v in ap.items() if k != "dist")
            pri = f" ~{ap['dist']}({args})"
        return f"Param({self.name}={self.value}{self.unit}{flag}{bnd}{pri})"


# ===========================================================================
# MaterialRegion
# ===========================================================================


@dataclass
class MaterialRegion:
    """A named material and the spatial region it occupies.

    The ``sdf_tree`` is the *distribution* of the material inside the part
    and is the data we actually care about at this layer. Physical
    properties (E, nu, rho, sigma, yield, ...) are intentionally **not**
    stored here - they are looked up by ``name`` / ``material_id`` from the
    separate ``emergent_matter_materials`` package, keeping the ``.sdm``
    file focused on geometry + identity.

    Attributes:
        material_id (int): Integer key (matches voxel label / segmentation id).
        name (str): Material name, e.g. ``"Cu"``, ``"PLA"``, ``"Fe"``. Used
            as the look-up key into the Matter Library.
        sdf_tree (SDFTree): SDF expression tree describing the spatial
            region occupied by this material. Required.
    """

    material_id: int
    name: str
    sdf_tree: SDFTree

    def to_dict(self) -> dict[str, Any]:
        return {
            "material_id": self.material_id,
            "name": self.name,
            "sdf_tree": copy.deepcopy(self.sdf_tree),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MaterialRegion:
        if "sdf_tree" not in d or d["sdf_tree"] is None:
            raise ValueError(
                f"MaterialRegion {d.get('name')!r} is missing required 'sdf_tree'. "
                "Every material must define its spatial distribution as an SDF."
            )
        return cls(
            material_id=d["material_id"],
            name=d["name"],
            sdf_tree=d["sdf_tree"],
        )

    def __repr__(self) -> str:
        return f"MaterialRegion(id={self.material_id}, name={self.name!r})"

    def infer_bbox(
        self, part: Part
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Infer a conservative finite bbox for this region."""
        from software_defined_matter.sdf.bbox import infer_material_bbox

        return infer_material_bbox(self, part)


# ===========================================================================
# Port
# ===========================================================================


# ===========================================================================
# Objective
# ===========================================================================

ObjectiveSense = Literal["minimize", "maximize"]


@dataclass
class Objective:
    """A symbolic objective to be minimised or maximised.

    The ``expr`` tree uses the expression DSL in
    :mod:`software_defined_matter.dsl.expr` and can reference free params, other
    params, and metrics of the part's SDF.

    Attributes:
        name (str): Identifier.
        sense (Literal["minimize", "maximize"]): Optimisation direction.
        expr (ExprTree): Expression tree.
        weight (float): Multiplier applied to this term when the optimiser
            forms a scalarised objective.
    """

    name: str
    sense: ObjectiveSense
    expr: ExprTree
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sense": self.sense,
            "expr": copy.deepcopy(self.expr),
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Objective:
        sense = d.get("sense", "minimize")
        if sense not in ("minimize", "maximize"):
            raise ValueError(f"Objective.sense must be 'minimize' or 'maximize', got {sense!r}")
        return cls(
            name=d["name"],
            sense=sense,
            expr=d["expr"],
            weight=float(d.get("weight", 1.0)),
        )

    def __repr__(self) -> str:
        return f"Objective({self.name!r}, {self.sense}, weight={self.weight})"


# ===========================================================================
# Constraint
# ===========================================================================

ConstraintOp = Literal["<=", ">=", "==", "<", ">"]
_CONSTRAINT_OPS = {"<=", ">=", "==", "<", ">"}


@dataclass
class Constraint:
    """A symbolic constraint: ``expr <op> rhs``.

    Attributes:
        name (str): Identifier.
        expr (ExprTree): LHS expression tree.
        op (Literal["<=", ">=", "==", "<", ">"]): Comparison operator.
        rhs (float): Right-hand-side scalar.
        tolerance (float): Slack used when evaluating feasibility numerically.
    """

    name: str
    expr: ExprTree
    op: ConstraintOp
    rhs: float
    tolerance: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "expr": copy.deepcopy(self.expr),
            "op": self.op,
            "rhs": self.rhs,
            "tolerance": self.tolerance,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Constraint:
        op = d["op"]
        if op not in _CONSTRAINT_OPS:
            raise ValueError(f"Constraint.op must be one of {_CONSTRAINT_OPS}, got {op!r}")
        return cls(
            name=d["name"],
            expr=d["expr"],
            op=op,
            rhs=float(d["rhs"]),
            tolerance=float(d.get("tolerance", 0.0)),
        )

    def __repr__(self) -> str:
        return f"Constraint({self.name!r}, {self.op} {self.rhs})"


# ===========================================================================
# Part
# ===========================================================================


@dataclass
class Part:
    """Top-level object representing a single additively manufactured component.

    Attributes:
        name (str): Identifier.
        params (dict[str, Param]): The part's design variables, keyed by name.
        materials (list[MaterialRegion]): Evaluated in order; later entries
            override earlier ones where they overlap.
        ports (list[Port]): Interface ports.
        objectives (list[Objective]): Terms to minimise or maximise.
        constraints (list[Constraint]): Feasibility conditions.
        metadata (dict): Free-form dict for version, author, AM process,
            units, etc.
        history (list[dict]): Optional list of partial-state snapshots
            written during optimisation.
    """

    name: str
    params: dict[str, Param] = field(default_factory=dict)
    materials: list[MaterialRegion] = field(default_factory=list)
    ports: list[Port] = field(default_factory=list)
    objectives: list[Objective] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    kinematics: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.validate_ports()

    def validate_ports(self) -> None:
        """Reject duplicate ports, missing frame parameters, and unknown body attachments."""
        names: set[str] = set()
        bodies = {body["name"] for body in (self.kinematics or {}).get("bodies", [])}
        for port in self.ports:
            if not isinstance(port, Port) or port.name in names:
                raise ValueError(
                    f"Part {self.name!r}.ports: expected unique Port declarations, got {port!r}"
                )
            names.add(port.name)
            missing = scalar_refs(port.frame.to_dict()) - set(self.params)
            if missing:
                raise ValueError(
                    f"Port {port.name!r}.frame: undeclared parameters {sorted(missing)}"
                )
            if port.body is not None and port.body not in bodies:
                raise ValueError(f"Port {port.name!r}.body: unknown body {port.body!r}")

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def add_param(self, param: Param) -> Part:
        self.params[param.name] = param
        return self

    def add_material(self, region: MaterialRegion) -> Part:
        self.materials.append(region)
        return self

    def add_port(self, node: Port) -> Part:
        """Append an attachment port; serialization validates its complete scope."""
        self.ports.append(node)
        return self

    def add_objective(self, objective: Objective) -> Part:
        self.objectives.append(objective)
        return self

    def add_constraint(self, constraint: Constraint) -> Part:
        self.constraints.append(constraint)
        return self

    def free_params(self) -> dict[str, Param]:
        """Return only the optimisable (free) parameters, preserving order."""
        invalid = [p.name for p in self.params.values() if p.free and p.expr is not None]
        if invalid:
            raise ValueError(
                "Parameters cannot be both free and defined by a relation: "
                + ", ".join(repr(name) for name in invalid)
            )
        return {k: v for k, v in self.params.items() if v.free}

    def derived_params(self) -> dict[str, Param]:
        """Params whose value is defined by a relation, in insertion order."""
        return {k: v for k, v in self.params.items() if v.expr is not None}

    def derived_order(self) -> list[str]:
        """Derived param names in dependency order (inputs before outputs).

        Raises ``sdf.param_refs.ParamRelationError`` naming the cycle. Evaluating
        in this order means every relation reads values that are already final,
        so nothing depends on the iteration order of ``params``.

        The sort itself lives in ``sdf.param_refs.relation_order``, shared with
        the load-time check — one implementation, so a file that validates and
        a Part that evaluates cannot disagree about what a cycle is. Imported
        lazily: ``sdf`` imports this module.
        """
        from software_defined_matter.dsl.expr import expr_param_names
        from software_defined_matter.sdf.param_refs import relation_order

        derived = self.derived_params()
        deps = {n: expr_param_names(p.expr) & set(derived) for n, p in derived.items()}
        return relation_order(deps)

    def derived_values(self) -> dict[str, float]:
        """Evaluate every relation at the params' current values. Pure.

        Uses the SAME evaluator the SDF closure uses (``eval_expr_pure``
        through a :class:`ParamBinding`), so the cached number written to the
        file is bit-identical to what the traced path computes — a second,
        "simpler" evaluator here is how the cache and the geometry drift apart.
        """
        from software_defined_matter.dsl.expr import eval_expr_pure
        from software_defined_matter.dsl.resolve import make_binding

        if not self.derived_params():
            return {}
        binding = make_binding(self)
        free_vec = binding.initial_free_vector()
        out: dict[str, float] = {}
        for name in self.derived_order():
            expr = self.params[name].expr
            assert expr is not None  # derived_order only yields names with an expr
            out[name] = float(eval_expr_pure(expr, binding, free_vec))
        return out

    def refresh_derived(self) -> Part:
        """Write evaluated relations back into derived params' cached ``value``.

        Called by :meth:`to_dict`; call it directly after mutating base params
        if you are about to read ``Param.value`` yourself.
        """
        for name, val in self.derived_values().items():
            self.params[name].value = val
        return self

    def free_param_names(self) -> list[str]:
        return list(self.free_params().keys())

    def param_vector(self) -> list[Number]:
        """Ordered list of free-parameter values (for optimisers)."""
        return [p.numeric_value() for p in self.free_params().values()]

    def computed_envelope(self) -> SDFTree | None:
        """Return the smooth union of all material SDF trees.

        Uses ``smooth_union`` with ``k`` taken from
        ``metadata["envelope_smooth_k"]`` (default 0.05) to keep material
        interfaces differentiable for gradient-based optimisers.
        Returns ``None`` when no materials are defined.
        """
        if not self.materials:
            return None
        k = float(self.metadata.get("envelope_smooth_k", 0.05))
        result = self.materials[0].sdf_tree
        for m in self.materials[1:]:
            result = sdf_op("smooth_union", [result, m.sdf_tree], k=k)
        return result

    def infer_bbox_from_sdf(
        self,
        *,
        persist_to_metadata: bool = True,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Infer a conservative finite bbox from the part envelope SDF.

        The envelope is produced by :meth:`computed_envelope`. By default, the
        inferred bbox is written to ``metadata["bbox"]`` so existing sampling
        and meshing paths can consume it directly.
        """
        from software_defined_matter.sdf.bbox import infer_sdf_bbox

        envelope = self.computed_envelope()
        if envelope is None:
            raise ValueError("Cannot infer bbox: Part has no materials.")
        bbox = infer_sdf_bbox(envelope, self)
        if persist_to_metadata:
            self.metadata["bbox"] = [list(bbox[0]), list(bbox[1])]
        return bbox

    def update_from_vector(self, values: list[Number]) -> None:
        """Update free-parameter values from an optimiser vector.
        'optimiser vector' here is the flat, ordered numerical view
        of the part's free parameters: the lingua franca between
        the structured Part/Param model and any black-box numerical optimiser.
        """
        free = list(self.free_params().values())
        if len(values) != len(free):
            raise ValueError(f"update_from_vector expected {len(free)} values, got {len(values)}")
        for param, val in zip(free, values, strict=False):
            param.value = val
        # Base params moved, so every cached relation is stale. No-op when the
        # part has no relations, so an optimiser loop pays nothing for this.
        self.refresh_derived()

    def snapshot(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """Append a snapshot of the current free-param state to ``history``.

        Used by optimisers to log partial ``.sdm`` states. The returned
        dict is the entry that was appended (also a convenient return for
        callers who want to attach objective/constraint values to it before
        the file is written).
        """
        entry = {
            "params": {name: self.params[name].value for name in self.free_param_names()},
        }
        if extra:
            entry.update(extra)
        self.history.append(entry)
        return entry

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        # Refresh first: a derived param's ``value`` on the wire is a cached
        # evaluation, and it is the ONLY thing a reader that predates ``expr``
        # will see. A stale cache there is a file that renders differently
        # depending on how old the reader is.
        self.validate_ports()
        self.refresh_derived()
        schema_version = min_schema_version_for(self)
        return {
            "schema_version": schema_version,
            **({"kind": "part"} if _parse_version(schema_version) >= (0, 5) else {}),
            "name": self.name,
            "metadata": copy.deepcopy(self.metadata),
            "params": {k: v.to_dict() for k, v in self.params.items()},
            "materials": [m.to_dict() for m in self.materials],
            **({"ports": [c.to_dict() for c in self.ports]} if self.ports else {}),
            "objectives": [o.to_dict() for o in self.objectives],
            "constraints": [c.to_dict() for c in self.constraints],
            "history": copy.deepcopy(self.history),
            **(
                {"kinematics": copy.deepcopy(self.kinematics)}
                if self.kinematics is not None
                else {}
            ),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Part:
        if "couplings" in d:
            raise ValueError(
                "Part.couplings was removed; migrate explicitly to Port and Part.ports"
            )
        if "port" in d:
            raise ValueError("Part.port was renamed to Part.ports")
        if d.get("kind", "part") != "part":
            raise ValueError("Part.from_dict expected kind=part; use the assembly loader")
        sv = d.get("schema_version")
        if sv is not None and sv not in KNOWN_SCHEMA_VERSIONS:
            # Forward-compatible: warn but accept; real migrations live in io.py.
            import warnings

            warnings.warn(
                f"Loading .sdm with schema_version={sv!r} but package knows "
                f"{KNOWN_SCHEMA_VERSIONS}. Attempting best-effort load.",
                stacklevel=2,
            )
        return cls(
            name=d["name"],
            params={k: Param.from_dict(v) for k, v in d.get("params", {}).items()},
            materials=[MaterialRegion.from_dict(m) for m in d.get("materials", [])],
            ports=[Port.from_dict(c) for c in d.get("ports", [])],
            objectives=[Objective.from_dict(o) for o in d.get("objectives", [])],
            constraints=[Constraint.from_dict(c) for c in d.get("constraints", [])],
            metadata=d.get("metadata", {}),
            history=list(d.get("history", [])),
            kinematics=copy.deepcopy(d.get("kinematics")),
        )

    # Thin wrappers. Real I/O with validation lives in ``software_defined_matter.io``.
    def save(self, path: str, *, b_validate_schema: bool = True) -> None:
        from software_defined_matter.io import save as _save

        _save(self, path, b_validate_schema=b_validate_schema)

    @classmethod
    def load(cls, path: str) -> Part:
        from software_defined_matter.io import load_part as _load

        return _load(path)

    def __repr__(self) -> str:
        return (
            f"Part({self.name!r}, "
            f"params={list(self.params)}, "
            f"materials={len(self.materials)}, "
            f"ports={len(self.ports)}, "
            f"objectives={len(self.objectives)}, "
            f"constraints={len(self.constraints)})"
        )


def _capabilities_used(part: Part) -> set[str]:
    """Which entries of ``_CAPABILITY_MIN_VERSION`` ``part``'s content exercises."""
    used = {"kinematics"}
    if part.ports:
        used.add("ports")

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("type") == "unop" and value.get("op") in {"sin", "cos"}:
                used.add("trig_expr")
            if value.get("kind") in {"raster_field", "axis_ramp", "radial_hermite"}:
                used.add(value["kind"])
            if value.get("type") == "deform" and value.get("deform") in _CAPABILITY_MIN_VERSION:
                used.add(value["deform"])
            if (
                value.get("type") == "transform"
                and value.get("transform") in _CAPABILITY_MIN_VERSION
            ):
                used.add(value["transform"])
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for region in part.materials:
        visit(region.sdf_tree)
    visit(part.kinematics)
    visit([p.expr for p in part.params.values()])
    visit([o.expr for o in part.objectives])
    visit([c.expr for c in part.constraints])
    if any(p.has_authored_prior() for p in part.params.values()):
        used.add("prior_params")
    if any(p.expr is not None for p in part.params.values()):
        used.add("param_expr")
    return used


def min_schema_version_for(part: Part) -> str:
    """The minimum ``schema_version`` a reader needs to load ``part`` correctly.

    Driven by ``_CAPABILITY_MIN_VERSION`` rather than an if-chain: a future
    wire feature adds one table entry instead of another branch.
    """
    versions = [_CAPABILITY_MIN_VERSION[cap] for cap in _capabilities_used(part)]
    return max(versions, key=_parse_version)


# ===========================================================================
# SDF Expression-Tree DSL builders
# ===========================================================================
# These build the JSON-serialisable expression trees stored in
# MaterialRegion.sdf_tree and Port.sdf_tree.
# They do NOT evaluate the SDF; evaluation is done by
# software_defined_matter.sdf.compile.make_sdf_closure.
#
# Leaf values in the ``**kwargs`` accepted by these builders may be either
# plain numbers / lists of numbers OR an expression leaf of the form
# {"$ref": "param_name"} or an expression tree from ``software_defined_matter.dsl.expr``.
# Resolution is handled uniformly by ``software_defined_matter.dsl.resolve``.


def sdf_primitive(kind: str, **kwargs: Any) -> SDFTree:
    """Create a leaf node for a named primitive (e.g. ``sphere``, ``box``)."""
    return {"type": "primitive", "kind": kind, "params": kwargs}


def sdf_helix(
    major_r: Any,
    pitch: Any,
    r: Any,
    n_turns: Any,
    phase: Any = 0.0,
    handedness: Any = 1.0,
) -> SDFTree:
    """Create a ``helix`` primitive node: a circular tube wound about +Z.

    Thin wrapper over :func:`sdf_primitive` that exists to guarantee ALL six
    params are present in the node. The GLSL emitter raises on a missing arg
    while the JAX evaluator would silently fall back to its Python default,
    so a hand-built ``sdf_primitive("helix", ...)`` that omits ``phase`` meshes
    fine and then fails in the viewer. Going through this builder makes the two
    paths agree by construction.

    See :func:`software_defined_matter.sdf.sdf_shapes.helix` for the geometry.
    ``phase`` is the clocking knob: two helices sharing a ``pitch`` and
    differing only in ``phase`` are the registered male/female pair of a thread.
    """
    return sdf_primitive(
        "helix",
        major_r=major_r,
        pitch=pitch,
        r=r,
        n_turns=n_turns,
        phase=phase,
        handedness=handedness,
    )


def sdf_screw_thread(
    r_root: Any,
    depth: Any,
    pitch: Any,
    width: Any,
    n_turns: Any,
    phase: Any = 0.0,
    handedness: Any = 1.0,
    flank_deg: Any = 60.0,
) -> SDFTree:
    """Create a ``screw_thread`` primitive node: a truncated-V helical ridge.

    Same rationale as :func:`sdf_helix` for existing as a builder — every param is
    filled in, so the JAX evaluator and the GLSL emitter (which raises on a
    missing arg) cannot disagree.

    See :func:`software_defined_matter.sdf.sdf_shapes.screw_thread`.
    """
    return sdf_primitive(
        "screw_thread",
        r_root=r_root,
        depth=depth,
        pitch=pitch,
        width=width,
        n_turns=n_turns,
        phase=phase,
        handedness=handedness,
        flank_deg=flank_deg,
    )


def sdf_raster_field(
    origin: Any,
    spacing: Any,
    values: Any,
    *,
    encoding: str = "f32le",
    step_scale: float | None = None,
    provenance: dict[str, Any] | None = None,
) -> SDFTree:
    """Create a ``raster_field`` primitive from decoded samples (DR-0003).

    ``values`` is ``(nz, ny, nx)``; the builder encodes wire base64 and fills
    ``dims``. ``spacing`` may be a scalar (broadcast to vec3) or a 3-list —
    the wire form always stores a 3-list so schema validation stays exact.
    """
    import numpy as _np

    from software_defined_matter.sdf.raster import encode_raster_data

    arr = _np.asarray(values, dtype=_np.float32)
    if arr.ndim != 3:
        raise ValueError(f"values must be (nz, ny, nx), got shape {arr.shape}")
    nz, ny, nx = arr.shape
    if isinstance(spacing, (int, float)):
        spacing_v = [float(spacing)] * 3
    else:
        spacing_v = [float(v) for v in spacing]
        if len(spacing_v) != 3:
            raise ValueError(f"spacing must be scalar or 3-list, got {spacing!r}")
    params: dict[str, Any] = {
        "origin": [float(v) for v in origin],
        "spacing": spacing_v,
        "dims": [nx, ny, nz],
        "encoding": encoding,
        "data": encode_raster_data(arr, encoding),
    }
    if step_scale is not None:
        params["step_scale"] = float(step_scale)
    if provenance is not None:
        params["provenance"] = dict(provenance)
    return {"type": "primitive", "kind": "raster_field", "params": params}


def sdf_op(op: str, children: list[SDFTree], **kwargs: Any) -> SDFTree:
    """Create a CSG operation node.

    Supported ``op`` values:
      - ``union``, ``subtract``, ``intersect``
      - ``smooth_union``, ``smooth_subtract``, ``smooth_intersect``
        (take an extra keyword ``k`` for the smoothing radius)
      - ``softmin_many``: log-sum-exp softmin across N children
        (takes ``k`` for the smoothing temperature)
      - ``softmin_chunked``: memory-disciplined N-way softmin that
        evaluates children in chunks (takes ``k`` and an optional
        ``chunk_size``, default 64)
    """
    node: SDFTree = {"type": "op", "op": op, "children": children}
    if kwargs:
        node["params"] = kwargs
    return node


def sdf_transform(transform: str, child: SDFTree, **kwargs: Any) -> SDFTree:
    """Create a transform node.

    Supported ``transform`` values:
      - ``translate`` (takes ``t``: 3-vector)
      - ``scale`` (takes ``s``: scalar)
      - ``rotate_x`` / ``rotate_y`` / ``rotate_z`` (take ``angle`` in
        radians)
      - ``rotate_matrix`` (takes ``R``: 3x3 matrix)
      - ``canonical_sector_fold``: N-fold rotational symmetry about the
        Z axis. Takes ``n_sectors``, and optionally ``centered``
        (fold into ``[-sector/2, sector/2)``) and ``phase_frac``.
      - ``mirror``: reflective (bilateral) symmetry about a plane: the
        child is unioned with its reflection,
        ``min(child(p), child(reflect(p)))``, so it works wherever the
        child sits (either side of the plane, or crossing it). Takes
        ``n`` (plane normal, normalised internally) and ``o`` (point on
        the plane); both ``$ref``-able.
    """
    node: SDFTree = {"type": "transform", "transform": transform, "child": child}
    if kwargs:
        node["params"] = kwargs
    return node


def sdf_modifier(modifier: str, child: SDFTree, **kwargs: Any) -> SDFTree:
    """Create a modifier node (``round`` / ``onion`` / ``elongate``)."""
    node: SDFTree = {"type": "modifier", "modifier": modifier, "child": child}
    if kwargs:
        node["params"] = kwargs
    return node


def sdf_deform(deform: str, child: SDFTree, **kwargs: Any) -> SDFTree:
    """Create a deformation node (``twist`` / ``bend`` / ``displace``).

    For ``displace``, pass the displacement field tree as the ``field``
    keyword (built with :func:`field_primitive` / :func:`field_op`)::

        sdf_deform("displace", shape_tree, field=field_primitive("sin_xyz", ...))

    The displacement field is stored on the node in a structural ``field``
    slot (alongside ``child``); other kwargs go into ``params``.
    """
    node: SDFTree = {"type": "deform", "deform": deform, "child": child}
    if deform == "displace":
        if "field" not in kwargs:
            raise ValueError(
                "sdf_deform('displace', ...) requires a 'field' kwarg "
                "(a field tree built with field_primitive / field_op)"
            )
        node["field"] = kwargs.pop("field")
    if kwargs:
        node["params"] = kwargs
    return node


def field_primitive(kind: str, **kwargs: Any) -> SDFTree:
    """Create a leaf node for a named scalar-field primitive.

    Supported ``kind`` values:
      - ``sin_xyz``: separable 3-D sinusoidal corrugation
        (kwargs: ``freq=[fx, fy, fz]``, ``amplitude``, ``phase=[px, py, pz]``)
      - ``radial``: axially symmetric ripple in the XY plane
        (kwargs: ``freq``, ``amplitude``, ``phase``)
      - ``angular``: azimuthal ripple about the Z axis; ``freq`` evenly
        spaced lobes, constant along any radius
        (kwargs: ``freq`` (use an integer lobe count for a seam-free
        closed pattern), ``amplitude``, ``phase``)

    Field primitives are consumed by ``sdf_deform("displace", ...)``. Their
    output is a displacement amplitude, NOT a distance.
    """
    return {"type": "field", "kind": kind, "params": kwargs}


def field_op(op: str, children: list[SDFTree], **kwargs: Any) -> SDFTree:
    """Create a field-tree combinator node.

    Supported ``op`` values:
      - ``add``: sum of child fields
    """
    node: SDFTree = {"type": "field_op", "op": op, "children": children}
    if kwargs:
        node["params"] = kwargs
    return node


def sdf_2d_to_3d(method: str, child_2d: SDFTree, **kwargs: Any) -> SDFTree:
    """Create a 2-D to 3-D lifting node (``revolution`` or ``extrusion``)."""
    node: SDFTree = {"type": "2d_to_3d", "method": method, "child": child_2d}
    if kwargs:
        node["params"] = kwargs
    return node


def sdf_sweep(
    profile_2d: SDFTree,
    path: Any,
    path_kind: str = "bspline",
    closed: bool = False,
    frame: str = "rmf",
    normal0: Any = None,
    **kwargs: Any,
) -> SDFTree:
    """Create a 3-D ``sweep`` node: a 2-D profile swept along a 3-D path.

    ``profile_2d`` is any 2-D SDF node (e.g. ``sdf_primitive("circle_2d", r=...)``
    or ``sdf_primitive("bspline_2d", ...)``). ``path`` is a list of ``[x, y, z]``
    control points; ``path_kind`` is ``"bspline"`` (periodic cubic loop, the
    default), ``"bezier"`` (composite cubic; open needs ``3K+1`` points, closed
    ``3K``), or ``"polyline"`` (used verbatim). ``closed`` wraps an open kind
    into a loop (``bspline`` is always a loop). Exact for circular profiles;
    surface-accurate (mild faceting) for general profiles.

    ``frame`` orients the cross-section: ``"rmf"`` (default) rotation-minimising,
    or ``"cylindrical"`` which locks the profile's first axis to the radial
    direction (width outward, height ~axial), for coil windings whose section
    must not twist off the cylindrical frame as an RMF does over many turns.
    ``normal0`` (``"rmf"`` only) is an optional ``[x, y, z]`` reference seeding
    the initial normal, so a swept ribbon can start in a chosen orientation
    (e.g. radial, to continue a winding). ``path`` may be passed as an ndarray to
    skip the (slow) element-wise list resolution for dense sampled paths.
    """
    params = {
        "path": ([list(pt) for pt in path] if isinstance(path, (list, tuple)) else path),
        "path_kind": str(path_kind),
        "closed": bool(closed),
        "frame": str(frame),
        **kwargs,
    }
    if normal0 is not None:
        params["normal0"] = normal0
    return {"type": "sweep", "child": profile_2d, "params": params}


def sdf_loft(
    profiles_2d: list[SDFTree],
    z: list[Number],
    smooth: bool = False,
    interp: str = "field",
    **kwargs: Any,
) -> SDFTree:
    """Create a loft node: N 2-D cross-section profiles lofted along Z.

    ``profiles_2d`` is a list of 2-D SDF nodes (e.g. :func:`sdf_primitive`
    ``"polygon_2d"``); ``z`` is the ascending axial station of each profile
    (``len(z) == len(profiles_2d) >= 2``). The cross-section is interpolated
    between sections and capped to ``[z[0], z[-1]]``, a Z-varying generalisation
    of ``extrusion``.

    ``z`` stations must be STRICTLY ASCENDING. Literal stations are validated at
    compile; ``$ref`` (param-driven) stations are the caller's responsibility to
    keep ascending: their values aren't known at compile time, so a
    non-ascending resolved ``z`` produces empty/degenerate geometry, not an error.

    interp : how to interpolate between sections.
      - ``"field"`` (default): blend the per-section 2-D distance FIELDS. Works
        for any 2-D children and needs no vertex correspondence, but the zero-set
        BULGES at sharp convex features (a swept leading edge scallops between
        stations).
      - ``"shape"``: interpolate the section OUTLINE, then take the exact polygon
        SDF of the interpolated outline; no bulge. Children must all be the SAME
        kind with matching counts (feature ``k`` = same feature across sections):
        ``polygon_2d`` (interpolates ``vertices``) OR ``bspline_2d`` / ``bezier_2d``
        (samples each section's ``control_points`` to its outline, then interpolates
        those outline vertices). The curve case gives a COMPACT, C2-smooth,
        scallop-free loft: the DSL stores only the handful of control points per
        section. (For ``smooth=False`` the sample-then-interpolate order is exactly
        equivalent to interpolating the control points then sampling: sampling is a
        linear map; for ``smooth=True`` the PCHIP interpolation is nonlinear, so the
        two orders differ and this uses sample-then-interpolate.)
    smooth : ``False`` -> piecewise-LINEAR interp (C0); ``True`` -> C1 monotone
        PCHIP (Fritsch-Carlson) interp: smooth through the sections, no joint
        creases, and no overshoot. See :func:`software_defined_matter.sdf.sdf_ops.loft`.
    """
    return {
        "type": "loft",
        "children": list(profiles_2d),
        "params": {"z": list(z), "smooth": bool(smooth), "interp": str(interp), **kwargs},
    }


def make_param_ref(name: str) -> dict[str, str]:
    """Leaf referencing a Param by name. Resolvable in any DSL leaf slot."""
    return {"$ref": name}


__all__ = [
    "FALLBACK_SCHEMA_VERSION",
    "KNOWN_SCHEMA_VERSIONS",
    "LATEST_SCHEMA_VERSION",
    "PRIOR_DISTS",
    "Constraint",
    "Port",
    "ExprTree",
    "MaterialRegion",
    "Number",
    "Objective",
    "Param",
    "Part",
    "SDFTree",
    "Vector3",
    "assert_supported",
    "field_op",
    "field_primitive",
    "make_param_ref",
    "min_schema_version_for",
    "sdf_2d_to_3d",
    "sdf_deform",
    "sdf_helix",
    "sdf_screw_thread",
    "sdf_loft",
    "sdf_modifier",
    "sdf_op",
    "sdf_primitive",
    "sdf_sweep",
    "sdf_transform",
    "supports_version",
]
