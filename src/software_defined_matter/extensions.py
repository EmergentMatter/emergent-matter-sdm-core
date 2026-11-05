"""Extension nodes: node kinds a separate package adds without forking this one.

A document carries an extension as one node::

    {"type": "extension", "id": "acme.taper", "version": 1,
     "payload": {...}, "children": [...], "children_dim": 3, "fallback": <public tree>}

``id`` names the kind (``vendor.kind``; the prefix ``em.`` is reserved for
Emergent Matter packages). ``version`` is the version of that kind's own payload
shape. ``payload`` is an object this package never interprets: it only finds
the ``$ref`` leaves in it and keeps it unchanged on save. ``children`` are
ordinary SDF subtrees of dimension ``children_dim``. ``fallback`` is an
ordinary SDF tree built only from this package's own node kinds.

A package that knows a kind registers an :class:`ExtensionSpec` for it with
:func:`register_extension`, explicitly and never as an import side effect. Where
no spec is registered for a node's ``id`` and ``version``, every consumer uses
the node's ``fallback`` in its place, so a document always opens.

Serialisation never consults the registry: the bytes a part saves to do not
depend on what is installed.

This module does not import JAX.
"""

from __future__ import annotations

import contextlib
import contextvars
import re
import warnings
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from software_defined_matter.model import Part, SDFTree

__all__ = [
    "ExtensionError",
    "ExtensionFallbackWarning",
    "ExtensionRegistrationError",
    "ExtensionSpec",
    "ExtensionUnavailableError",
    "ExtensionUse",
    "PROMOTIONS",
    "Promotion",
    "BBoxContext",
    "CompileContext",
    "EnvelopeContext",
    "FeatureContext",
    "RateContext",
    "ValidateContext",
    "extension_uses",
    "is_extension",
    "lookup",
    "policy",
    "promote_tree",
    "promoted_ids",
    "refresh_fallbacks",
    "register_extension",
    "registered_extensions",
    "unregister_extension",
]

#: ``vendor.kind``: lower-case words joined by dots, at least two parts.
ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")

#: The fields a node of ``type: "extension"`` may carry.
NODE_KEYS = frozenset(
    {"type", "id", "version", "payload", "children", "children_dim", "fallback", "name"}
)

#: A payload deeper or larger than this is refused by the validator: it is
#: untrusted input to code this package does not own.
MAX_PAYLOAD_DEPTH = 32
MAX_PAYLOAD_ITEMS = 10_000_000


class ExtensionError(Exception):
    """Base class for extension failures."""


class ExtensionRegistrationError(ExtensionError, ValueError):
    """A spec is malformed or its id is taken by a different spec."""


class ExtensionUnavailableError(ExtensionError):
    """An unregistered extension's fallback would silently freeze live parameters."""


class ExtensionFallbackWarning(UserWarning):
    """An extension is not available, so its fallback tree stands in for it."""


# ---------------------------------------------------------------------------
# Contexts: the only surface a callback sees of this package
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompileContext:
    """What :attr:`ExtensionSpec.compile` may use.

    ``binding`` is the part's :class:`~software_defined_matter.dsl.resolve.ParamBinding`.
    ``resolve(value, free_vec)`` turns a payload leaf (a number, a list, a
    ``$ref``, an expression) into a JAX value. ``compile_child(tree, binding)``
    compiles an SDF subtree, with ``binding`` when given. ``nominal`` maps each
    non-derived parameter to its value now, for plans made once at compile time.
    """

    binding: Any
    resolve: Callable[[Any, Any], Any]
    compile_child: Callable[..., Any]
    smooth_csg: bool
    smooth_k: float
    nominal: Mapping[str, float]


@dataclass(frozen=True)
class BBoxContext:
    """What :attr:`ExtensionSpec.bbox` may use. ``intervals`` maps param names to value ranges."""

    intervals: Mapping[str, tuple[float, float]]
    bbox: Callable[[Any], Any]
    bbox_2d_primitive: Callable[[Any], Any]
    vec_intervals: Callable[[Any, int], Any]
    scalar_abs_max: Callable[[Any], float]


@dataclass(frozen=True)
class RateContext:
    """What :attr:`ExtensionSpec.max_rate` may use. ``rate(tree)`` bounds a subtree's slope."""

    intervals: Mapping[str, tuple[float, float]]
    domain: Any
    rate: Callable[[Any], float]


@dataclass(frozen=True)
class FeatureContext:
    """What :attr:`ExtensionSpec.features` may use."""

    intervals: Mapping[str, tuple[float, float]]
    features: Callable[[Any], list[float | None]]


@dataclass(frozen=True)
class EnvelopeContext:
    """What :attr:`ExtensionSpec.envelope` may use. ``envelope(tree)`` fills a subtree's holes."""

    part: Any
    envelope: Callable[[Any], Any]


@dataclass(frozen=True)
class ValidateContext:
    """What :attr:`ExtensionSpec.validate` may use.

    ``validate_subtree(tree, dim, path)`` runs the ordinary validator on a public
    tree; ``fail(path, message)`` raises the validator's error.
    """

    known_params: frozenset[str]
    path: str
    expected_dim: Literal[2, 3]
    validate_subtree: Callable[[Any, Literal[2, 3], str], None]
    fail: Callable[[str, str], Any]


# ---------------------------------------------------------------------------
# Spec and registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtensionSpec:
    """Everything a package supplies for one extension kind.

    ``compile``, ``bbox`` and ``max_rate`` are required, so a kind cannot be
    registered without saying how to evaluate, bound and rate it.
    ``max_rate`` may return ``math.inf`` (no bound known); it must say so.

    ``csg_like`` makes the node's ``children`` follow the unboundedness rules
    of ``union``, ``subtract`` or ``intersect``. ``unbounded`` marks a kind valid
    only as a cutter, like ``plane``. ``envelope`` is optional; without it an
    envelope request on a tree that holds the node raises, naming the kind.
    ``make_fallback(node, part)`` builds a fallback tree for :func:`refresh_fallbacks`.
    """

    id: str
    versions: frozenset[int]
    dim: Literal[2, 3]
    compile: Callable[[dict[str, Any], CompileContext], Any]
    bbox: Callable[[dict[str, Any], BBoxContext], Any]
    max_rate: Callable[[dict[str, Any], RateContext], float]
    csg_like: Literal["union", "subtract", "intersect"] | None = None
    unbounded: bool = False
    validate: Callable[[dict[str, Any], ValidateContext], None] | None = None
    features: Callable[[dict[str, Any], FeatureContext], list[float | None]] | None = None
    envelope: Callable[[dict[str, Any], EnvelopeContext], Any] | None = None
    make_fallback: Callable[[dict[str, Any], Any], Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not ID_PATTERN.match(self.id):
            raise ExtensionRegistrationError(
                f"extension id {self.id!r}: expected lower-case words joined by dots, "
                "for example 'vendor.kind'"
            )
        if not self.versions or any(
            not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in self.versions
        ):
            raise ExtensionRegistrationError(
                f"extension {self.id!r}: versions must be a non-empty set of integers >= 1"
            )
        if self.dim not in (2, 3):
            raise ExtensionRegistrationError(f"extension {self.id!r}: dim must be 2 or 3")
        for name in ("compile", "bbox", "max_rate"):
            if not callable(getattr(self, name)):
                raise ExtensionRegistrationError(f"extension {self.id!r}: {name} must be callable")


@dataclass(frozen=True)
class Promotion:
    """A kind that became a native node kind in this package.

    ``to_native(node)`` returns the native node for an extension node whose
    ``version`` is at most ``max_version``, or ``None`` when it cannot convert
    that payload (the extension node then stays as it is).
    """

    max_version: int
    to_native: Callable[[dict[str, Any]], dict[str, Any] | None]


#: Extension ids this package has made native. Empty until the first one is.
PROMOTIONS: dict[str, Promotion] = {}

_REGISTRY: dict[str, ExtensionSpec] = {}
_FROZEN_PARAMS: contextvars.ContextVar[str] = contextvars.ContextVar(
    "sdm_extension_frozen_params", default="raise"
)


def promoted_ids() -> frozenset[str]:
    """The extension ids that are now native node kinds."""
    return frozenset(PROMOTIONS)


def register_extension(spec: ExtensionSpec) -> bool:
    """Register ``spec``; return ``True`` if it is now active.

    Registering the same spec again does nothing. A different spec for a taken
    id raises :class:`ExtensionRegistrationError`. An id this package has made
    native is skipped and returns ``False``: the native kind replaces it, and a
    package still registering the old id must not fail on import.
    """
    if not isinstance(spec, ExtensionSpec):
        raise ExtensionRegistrationError(f"expected an ExtensionSpec, got {type(spec).__name__}")
    if spec.id in PROMOTIONS:
        return False
    taken = _REGISTRY.get(spec.id)
    if taken is not None and taken is not spec:
        raise ExtensionRegistrationError(f"extension {spec.id!r} is already registered")
    _REGISTRY[spec.id] = spec
    return True


def unregister_extension(extension_id: str) -> None:
    """Remove a registration (tests). Unknown ids are ignored."""
    _REGISTRY.pop(extension_id, None)


def registered_extensions() -> Mapping[str, ExtensionSpec]:
    """A snapshot of the registry."""
    return dict(_REGISTRY)


def is_extension(node: Any) -> bool:
    return isinstance(node, dict) and node.get("type") == "extension"


def lookup(node: dict[str, Any]) -> ExtensionSpec | None:
    """The spec that handles ``node``, or ``None`` when its fallback must stand in.

    ``None`` when the id is not registered or the node's version is not one the
    registered spec reads.
    """
    spec = _REGISTRY.get(node.get("id"))  # type: ignore[arg-type]
    if spec is None or node.get("version") not in spec.versions:
        return None
    return spec


@contextlib.contextmanager
def policy(*, frozen_params: Literal["raise", "allow"]) -> Iterator[None]:
    """Choose what compiling an unavailable extension does about live parameters.

    By default (``"raise"``) compiling an unregistered extension whose payload
    refers to a free or derived parameter raises
    :class:`ExtensionUnavailableError`: the fallback is a fixed tree, so the
    parameter would do nothing, and an optimiser would stop at a zero gradient.
    A viewer that only displays the geometry opts in with ``"allow"``.
    """
    if frozen_params not in ("raise", "allow"):
        raise ValueError("frozen_params must be 'raise' or 'allow'")
    token = _FROZEN_PARAMS.set(frozen_params)
    try:
        yield
    finally:
        _FROZEN_PARAMS.reset(token)


def payload_refs(node: dict[str, Any]) -> set[str]:
    """Parameter names the node's payload refers to."""
    from software_defined_matter.sdf.param_refs import _walk_refs

    out: list[tuple[str, str]] = []
    _walk_refs(node.get("payload"), "", out)
    return {name for name, _ in out}


def use_fallback(node: dict[str, Any], binding: Any = None) -> SDFTree:
    """The tree that stands in for an unavailable ``node``; warn, and guard live parameters.

    ``binding`` (a ``ParamBinding``), when given, enables the frozen-parameter
    guard: payload refs to a free or derived parameter raise unless
    ``policy(frozen_params="allow")`` is active.
    """
    ext_id = node.get("id")
    if binding is not None and _FROZEN_PARAMS.get() == "raise":
        live = set(binding.free_names) | set(binding.derived)
        frozen = sorted(payload_refs(node) & live)
        if frozen:
            raise ExtensionUnavailableError(
                f"extension {ext_id!r} (version {node.get('version')!r}) is not available, and "
                f"its fallback is fixed, so parameters {frozen} would do nothing. Register the "
                "extension, or compile inside extensions.policy(frozen_params='allow') "
                "to view the fallback."
            )
    warnings.warn(
        f"extension {ext_id!r} (version {node.get('version')!r}) is not available; "
        "using its fallback",
        ExtensionFallbackWarning,
        stacklevel=3,
    )
    return node["fallback"]


# ---------------------------------------------------------------------------
# Reports and maintenance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtensionUse:
    """One extension node in a part: where, what, whether it is available, what it freezes."""

    id: str
    version: int
    path: str
    registered: bool
    frozen_params: tuple[str, ...] = field(default=())


def _walk_extensions(tree: Any, path: str) -> Iterator[tuple[str, dict[str, Any]]]:
    if isinstance(tree, dict):
        if tree.get("type") == "extension":
            yield path, tree
            for i, child in enumerate(tree.get("children") or []):
                yield from _walk_extensions(child, f"{path}.children[{i}]")
            return
        for key in ("child", "field"):
            if key in tree:
                yield from _walk_extensions(tree[key], f"{path}.{key}")
        for i, child in enumerate(tree.get("children") or []):
            yield from _walk_extensions(child, f"{path}.children[{i}]")


def extension_uses(part: Part) -> list[ExtensionUse]:
    """Every extension node in ``part``'s material regions, outermost first.

    For an unavailable node, ``frozen_params`` lists the free or derived
    parameters its payload refers to: the ones its fallback ignores.
    """
    out: list[ExtensionUse] = []
    live = {n for n, p in part.params.items() if p.free or p.expr is not None}
    for i, region in enumerate(part.materials):
        for path, node in _walk_extensions(region.sdf_tree, f"materials[{i}].sdf_tree"):
            spec = lookup(node)
            frozen = () if spec else tuple(sorted(payload_refs(node) & live))
            out.append(
                ExtensionUse(
                    str(node.get("id")), int(node.get("version", 0)), path, spec is not None, frozen
                )
            )
    return out


def refresh_fallbacks(part: Part) -> int:
    """Rebuild the fallback of every registered extension node that can; return how many.

    Explicit and never part of saving. A kind without ``make_fallback`` keeps its
    fallback.
    """
    n = 0
    for region in part.materials:
        for _path, node in _walk_extensions(region.sdf_tree, ""):
            spec = lookup(node)
            if spec is not None and spec.make_fallback is not None:
                node["fallback"] = spec.make_fallback(node, part)
                n += 1
    return n


def promote_tree(tree: Any) -> Any:
    """``tree`` with every promotable extension node replaced by its native node.

    Returns the same object when nothing promotes. Runs on read; a read never
    rewrites a file.
    """
    if not PROMOTIONS:
        return tree
    return _promote(tree)


def _promote(node: Any) -> Any:
    if isinstance(node, list):
        return [_promote(n) for n in node]
    if not isinstance(node, dict):
        return node
    if node.get("type") == "extension":
        promotion = PROMOTIONS.get(node.get("id"))  # type: ignore[arg-type]
        version = node.get("version")
        if promotion is not None and isinstance(version, int) and version <= promotion.max_version:
            native = promotion.to_native(node)
            if native is not None:
                return _promote(native)
        out = dict(node)
        if "children" in out:
            out["children"] = [_promote(c) for c in out["children"]]
        return out
    return {k: (_promote(v) if k in ("child", "children", "field") else v) for k, v in node.items()}
