"""Dimension audit: every dimension in a part is free, an input, or driven, and nothing dangles.

Each number a CEM puts in a part gets its value in exactly one way:

- **free**: declared with limits; the designer picked it and an optimizer may move it.
- **input**: declared, set from outside the part: a requirement, a bought part, a standard,
  a process limit, or a parent assembly. An optimizer may move it only when the CEM marks
  it ``tradeable`` (a trade study); by default it is fixed, so an optimizer cannot "solve" a
  design by changing what it was asked to meet.
- **driven**: computed from other dimensions. Never declared: it lives in the CEM's code.

**Driving** is not declared either. A dimension is driving when something else moves with it,
and that is found by nudging, not written down: a written label would drift from what the
code does. Any of the three can be driving; a driven dimension that drives another is a link
in a chain.

The audit rebuilds the part with each declared dimension nudged on its own and watches which
numbers in the emitted SDF trees move. From that:

- a number that no declared dimension moves is **dangling** (typed into the shape code);
- a declared dimension that moves nothing is **unused**, unless its role says it is a limit
  (``derived``: a floor feeding ``validate()``) or a tool setting (``tool``);
- a length that neither stays put nor scales exactly when every length is scaled together is
  an **offset**: a dimension with a typed-in length added to it (``outer_radius + 1.3``).

Declarations are read from a CEM's ``[cem.params]`` table (its ``cem.toml``, already parsed):
each needs ``origin`` (``free`` or ``input``; not needed for ``role = "tool"``), a ``source``
of one of :data:`SOURCE_KINDS`, and, when free, ``bounds``. Typical use in a CEM's tests::

    cem = tomllib.loads(Path("cem.toml").read_text())["cem"]
    assert_dimensions(build_widget_part, WidgetParameters(), cem["params"])

What it cannot see: that an input is tied to the right thing. ``source =
"parent:motor:case_radius"`` is checked for form, not against the motor's own file. A clean
audit says nothing dangles, not that every link points where it should.
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from software_defined_matter.model import Part

#: Where a declared dimension's value comes from.
ORIGINS = ("free", "input")

#: The kinds of ``source`` a declared dimension may name, as regular expressions.
SOURCE_KINDS: dict[str, str] = {
    "requirement": r"requirement:\S+",
    "part": r"part:[^:\s]+:\S+",
    "standard": r"standard:.+",
    "process": r"process:[^:\s]+:\S+",
    "parent": r"parent:[^:\s]+:\S+",
    "choice": r"choice: .+",
    "tool": r"tool: .+",
}
_SOURCE = re.compile("^(" + "|".join(SOURCE_KINDS.values()) + ")")

#: Roles whose dimensions need not move the geometry: limits feeding validate(), tool settings.
ROLES_OFF_GEOMETRY = ("derived", "tool")
#: A ``choice`` source has to give a reason of at least this many words.
N_REASON_WORDS = 5
#: Numbers that may appear in a tree without coming from a dimension.
PLAIN = (0.0, 0.5, 1.0, 2.0)
#: How much a dimension is scaled when nudged.
D_SCALE = 1.07

_Numbers = list[tuple[str, Any]]
_Declared = Mapping[str, Mapping[str, Any]]


class DimensionAuditError(AssertionError):
    """Raised by :func:`assert_dimensions` when the audit finds anything."""


@dataclass(frozen=True)
class DimensionFinding:
    """One thing the audit found: which check, where, and what to do about it."""

    kind: str  # declaration | dangling | unused | offset | shape_changed
    where: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.kind}] {self.where}: {self.detail}"


@dataclass(frozen=True)
class DimensionReport:
    """What :func:`dimension_audit` found, plus what it learned about the part.

    ``drives`` maps each declared dimension to the places in the emitted trees that move with
    it; the ones with any are driving. ``optimizable`` lists what an optimizer may move: every
    free dimension and every input marked ``tradeable``.
    """

    findings: list[DimensionFinding] = field(default_factory=list)
    drives: dict[str, list[str]] = field(default_factory=dict)
    optimizable: list[str] = field(default_factory=list)

    @property
    def driving(self) -> list[str]:
        return [n for n, places in self.drives.items() if places]


def _decl(name: str, detail: str) -> DimensionFinding:
    return DimensionFinding("declaration", name, detail)


def find_declaration_problems(declared: _Declared) -> list[DimensionFinding]:
    """Check each declared dimension's ``origin``, ``source``, ``bounds`` and ``tradeable``."""
    out: list[DimensionFinding] = []
    for name, spec in declared.items():
        origin, source = spec.get("origin"), spec.get("source")
        if spec.get("role") != "tool" and origin not in ORIGINS:
            out.append(_decl(name, f"`origin` must be one of {ORIGINS}, not {origin!r}"))
        if not source:
            out.append(_decl(name, "no `source`: say where the value comes from"))
        elif not _SOURCE.match(str(source)):
            out.append(_decl(name, f"`source` {source!r} is not one of: {', '.join(SOURCE_KINDS)}"))
        elif str(source).startswith("choice:") and len(str(source).split()) - 1 < N_REASON_WORDS:
            out.append(_decl(name, f"`choice` needs a reason of {N_REASON_WORDS}+ words"))
        bounds = spec.get("bounds")
        if origin == "free" and not (
            isinstance(bounds, (list, tuple)) and len(bounds) == 2 and bounds[0] < bounds[1]
        ):
            out.append(_decl(name, "a free dimension needs `bounds = [low, high]`, low < high"))
        if "tradeable" in spec and origin != "input":
            out.append(_decl(name, "`tradeable` applies to inputs only (free ones always are)"))
    return out


def optimizable(declared: _Declared) -> list[str]:
    """What an optimizer may move: free dimensions, and inputs marked ``tradeable = true``."""
    return [
        n
        for n, s in declared.items()
        if s.get("origin") == "free" or (s.get("origin") == "input" and s.get("tradeable") is True)
    ]


def _numbers(tree: Any, where: str = "") -> _Numbers:
    """Every number in a tree with its place, plus every place's shape, so trees line up."""
    out: _Numbers = []
    if isinstance(tree, dict):
        out.append((f"{where}{{}}", tuple(sorted(tree))))
        for k in sorted(tree):
            out += _numbers(tree[k], f"{where}/{k}")
    elif isinstance(tree, (list, tuple)):
        out.append((f"{where}[]", len(tree)))
        for i, v in enumerate(tree):
            out += _numbers(v, f"{where}/{i}")
    elif isinstance(tree, (int, float)) and not isinstance(tree, bool):
        out.append((where, float(tree)))
    return out


def _is_plain(x: float) -> bool:
    if any(math.isclose(abs(x), c, abs_tol=1e-12) for c in PLAIN):
        return True
    steps = x / (math.pi / 12)  # whole 15 degree steps: an angle set by a count
    return math.isclose(steps, round(steps), abs_tol=1e-9)


def _tree_numbers[P](build: Callable[[P], Part], params: P) -> _Numbers:
    part = build(params)
    return _numbers({m.name or str(i): m.sdf_tree for i, m in enumerate(part.materials)})


def _scaled[P](params: P, names: Iterable[str], k: float) -> P:
    changes = {n: getattr(params, n) * k for n in names}
    return dataclasses.replace(params, **changes)  # type: ignore[type-var]


def _lined_up(a: _Numbers, b: _Numbers) -> list[tuple[str, float, float]] | None:
    """Pair the numbers of two trees by place, or None when the trees differ in shape."""
    if [w for w, _ in a] != [w for w, _ in b]:
        return None
    return [(w, va, vb) for (w, va), (_, vb) in zip(a, b, strict=True) if isinstance(va, float)]


def _nudgeable(params: Any, declared: _Declared) -> list[str]:
    """Declared dimensions that are non-zero float fields of ``params`` (a zero cannot be
    scaled, and an int is a count, which may legitimately add or remove features)."""
    fields = {f.name for f in dataclasses.fields(params)}
    return [
        n
        for n in declared
        if n in fields and isinstance(getattr(params, n), float) and getattr(params, n) != 0.0
    ]


def dimension_audit[P](
    build: Callable[[P], Part],
    params: P,
    declared: _Declared,
    *,
    length_fields: Iterable[str] | None = None,
    d_scale: float = D_SCALE,
) -> DimensionReport:
    """Run every check on ``build(params)`` and return the findings without raising.

    ``build`` turns a parameters dataclass into a Part; ``params`` is the instance to audit
    around (usually the defaults); ``declared`` is the CEM's ``[cem.params]`` table, keyed by
    the dataclass field names. ``length_fields`` defaults to the declared dimensions with
    ``unit = "mm"``, tool settings included (a cut's overshoot sized from the voxel is a
    length like any other).
    """
    findings = find_declaration_problems(declared)
    base = _tree_numbers(build, params)
    drives: dict[str, list[str]] = {}
    for name in _nudgeable(params, declared):
        pairs = _lined_up(base, _tree_numbers(build, _scaled(params, [name], d_scale)))
        if pairs is None:
            findings.append(
                DimensionFinding(
                    "shape_changed", name, "nudging it added or removed features; a count is an int"
                )
            )
            continue
        drives[name] = [w for w, va, vb in pairs if va != vb]
    moved = {w for places in drives.values() for w in places}
    for w, v in base:
        if isinstance(v, float) and w not in moved and not _is_plain(v):
            findings.append(
                DimensionFinding(
                    "dangling",
                    w,
                    f"{v:g} moves with no declared dimension: it is typed in. Declare it (free or "
                    "input) or compute it from what drives it",
                )
            )
    for name, places in drives.items():
        if not places and declared[name].get("role") not in ROLES_OFF_GEOMETRY:
            findings.append(
                DimensionFinding(
                    "unused", name, "declared but moves nothing in the part; remove it or use it"
                )
            )
    if length_fields is None:
        length_fields = [n for n, s in declared.items() if s.get("unit") == "mm"]
    lengths = [n for n in length_fields if n in _nudgeable(params, declared)]
    pairs = _lined_up(base, _tree_numbers(build, _scaled(params, lengths, d_scale)))
    if pairs is None:
        findings.append(
            DimensionFinding("shape_changed", "lengths", "scaling every length changed features")
        )
    else:
        findings += [
            DimensionFinding(
                "offset",
                w,
                f"{va:g} -> {vb:g} with every length scaled by {d_scale}: a typed-in length is "
                "added to a dimension. Declare it or compute it",
            )
            for w, va, vb in pairs
            if not math.isclose(vb, va, rel_tol=1e-9, abs_tol=1e-12)
            and not math.isclose(vb, d_scale * va, rel_tol=1e-9, abs_tol=1e-12)
        ]
    return DimensionReport(findings, drives, optimizable(declared))


def assert_dimensions[P](
    build: Callable[[P], Part],
    params: P,
    declared: _Declared,
    *,
    length_fields: Iterable[str] | None = None,
    d_scale: float = D_SCALE,
) -> DimensionReport:
    """:func:`dimension_audit` that raises :class:`DimensionAuditError` listing every finding."""
    report = dimension_audit(build, params, declared, length_fields=length_fields, d_scale=d_scale)
    if report.findings:
        raise DimensionAuditError(
            f"dimension audit found {len(report.findings)} problem(s):\n"
            + "\n".join(f"  {f}" for f in report.findings[:40])
        )
    return report
