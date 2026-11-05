"""Extension nodes: registry, validation, fallback, hooks, capability and promotion."""

from __future__ import annotations

import copy
import json
import math
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from software_defined_matter import (
    MaterialRegion,
    Param,
    Part,
    extensions,
    min_schema_version_for,
    sdf_extension,
    sdf_op,
    sdf_primitive,
    validate,
)
from software_defined_matter.dsl.resolve import make_binding
from software_defined_matter.extensions import (
    ExtensionRegistrationError,
    ExtensionSpec,
    ExtensionUnavailableError,
)
from software_defined_matter.glsl.emit import emit_glsl
from software_defined_matter.sdf import bbox as bbox_mod
from software_defined_matter.sdf.bbox import infer_sdf_bbox
from software_defined_matter.sdf.compile import make_sdf_closure
from software_defined_matter.sdf.envelope import EnvelopeInferenceError, infer_sdf_envelope
from software_defined_matter.sdf.features import infer_min_feature_size
from software_defined_matter.sdf.lipschitz import infer_sdf_max_rate
from software_defined_matter.sdf.validate import (
    SemanticValidationError,
    UnboundedRootError,
    validate_document_semantics,
)

EXT = "acme.slab"


@pytest.fixture(autouse=True)
def _clean_registry():
    before = dict(extensions._REGISTRY)
    promotions = dict(extensions.PROMOTIONS)
    yield
    extensions._REGISTRY.clear()
    extensions._REGISTRY.update(before)
    extensions.PROMOTIONS.clear()
    extensions.PROMOTIONS.update(promotions)


def _compile(node, ctx):
    half = node["payload"]["half"]

    def fn(p, free_vec):
        h = ctx.resolve(half, free_vec)
        return jnp.max(jnp.abs(p) - h, axis=-1)

    return fn


def _bbox(node, ctx):
    h = ctx.scalar_abs_max(node["payload"]["half"])
    return ((-h, -h, -h), (h, h, h))


def _spec(**over):
    kw = {
        "id": EXT,
        "versions": frozenset({1}),
        "dim": 3,
        "compile": _compile,
        "bbox": _bbox,
        "max_rate": lambda node, ctx: 1.0,
        **over,
    }
    return ExtensionSpec(**kw)


def _node(half=2.0, *, fallback_half=1.0, **kw):
    fb = sdf_primitive("box", b=[fallback_half] * 3)
    return sdf_extension(EXT, 1, {"half": half}, fb, **kw)


def _part(tree, **params):
    return Part(
        name="p",
        params=params,
        materials=[MaterialRegion(material_id=1, name="m", sdf_tree=tree)],
    )


def _at(tree, point, part=None):
    fn = make_sdf_closure(tree, part or _part(tree))
    return float(fn(jnp.asarray([point], dtype=jnp.float32), None)[0])


# -- registration ---------------------------------------------------------


@pytest.mark.parametrize("bad", ["slab", "Acme.slab", "acme.", ".slab", "acme slab", ""])
def test_a_malformed_id_is_refused(bad):
    with pytest.raises(ExtensionRegistrationError):
        _spec(id=bad)


def test_the_mandatory_callbacks_cannot_be_left_out():
    for name in ("compile", "bbox", "max_rate"):
        kw = {
            "id": EXT,
            "versions": frozenset({1}),
            "dim": 3,
            "compile": _compile,
            "bbox": _bbox,
            "max_rate": lambda n, c: 1.0,
        }
        del kw[name]
        with pytest.raises(TypeError):
            ExtensionSpec(**kw)


def test_versions_must_be_positive_integers():
    for bad in (frozenset(), frozenset({0}), frozenset({True}), frozenset({1.5})):
        with pytest.raises(ExtensionRegistrationError):
            _spec(versions=bad)


def test_registering_again_is_a_noop_or_a_replacement_and_a_different_kind_is_refused():
    spec = _spec()
    assert extensions.register_extension(spec) is True
    assert extensions.register_extension(spec) is True
    # a reloaded module builds a new spec with the same versions: that replaces, it does not fail
    again = _spec(compile=lambda n, c: None)
    assert extensions.register_extension(again) is True
    assert extensions.registered_extensions()[EXT] is again
    with pytest.raises(ExtensionRegistrationError):
        extensions.register_extension(_spec(versions=frozenset({1, 2})))


def test_a_promoted_id_is_skipped_silently():
    extensions.PROMOTIONS[EXT] = extensions.Promotion(1, lambda n: None)
    assert extensions.register_extension(_spec()) is False
    assert EXT not in extensions.registered_extensions()


# -- the node on the wire --------------------------------------------------


def test_builder_makes_the_documented_node():
    node = _node()
    assert node["type"] == "extension" and node["id"] == EXT and node["version"] == 1
    assert "children" not in node
    child = sdf_primitive("sphere", r=1.0)
    with_children = _node(children=[child], children_dim=3)
    assert with_children["children_dim"] == 3
    with pytest.raises(ValueError):
        _node(children=[child])


def test_a_part_with_an_extension_validates_with_nothing_registered():
    validate(_part(_node()))


def test_an_extension_needs_schema_0_7_and_nothing_else_does():
    plain = _part(sdf_primitive("sphere", r=1.0))
    assert min_schema_version_for(plain) != "0.7"
    assert min_schema_version_for(_part(_node())) == "0.7"
    assert _part(_node()).to_dict()["schema_version"] == "0.7"


def test_payload_content_does_not_raise_the_schema_version():
    node = sdf_extension(
        EXT, 1, {"kind": "raster_field", "deform": "shear_linear"}, sdf_primitive("sphere", r=1.0)
    )
    assert min_schema_version_for(_part(node)) == "0.7"
    # the fallback's own needs still count
    fallback = {
        "type": "deform",
        "deform": "shear_linear",
        "params": {"axis": [1.0, 0.0], "u0": -1.0, "u1": 1.0, "dz_0": 0.0, "dz_1": 1.0},
        "child": sdf_primitive("sphere", r=1.0),
    }
    assert min_schema_version_for(_part(sdf_extension(EXT, 1, {}, fallback))) == "0.7"


def test_serialisation_does_not_depend_on_the_registry():
    part = _part(_node())
    before = json.dumps(part.to_dict(), sort_keys=True)
    extensions.register_extension(_spec())
    assert json.dumps(part.to_dict(), sort_keys=True) == before


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda n: n.pop("fallback"), "fallback"),
        (lambda n: n.update(id="Bad Id"), "vendor.kind"),
        (lambda n: n.update(version=0), "version"),
        (lambda n: n.update(version=True), "version"),
        (lambda n: n.update(payload=[1]), "payload"),
        (lambda n: n.update(extra=1), "unrecognised"),
        (lambda n: n.update(children_dim=3), "children_dim"),
        (lambda n: n.update(children=[sdf_primitive("sphere", r=1.0)]), "children_dim"),
        (lambda n: n.update(fallback=_node()), "may not contain an extension"),
        (
            lambda n: n.update(fallback=sdf_op("union", [sdf_primitive("sphere", r=1.0), _node()])),
            "may not contain an extension",
        ),
        (lambda n: n.update(fallback=sdf_primitive("circle_2d", r=1.0)), "3-D"),
    ],
)
def test_a_malformed_node_is_rejected(mutate, message):
    node = _node()
    mutate(node)
    with pytest.raises(SemanticValidationError, match=message):
        validate_document_semantics(_part(node).to_dict())


def test_an_oversized_payload_is_rejected():
    deep: object = 1
    for _ in range(extensions.MAX_PAYLOAD_DEPTH + 2):
        deep = [deep]
    with pytest.raises(SemanticValidationError, match="nested deeper"):
        validate_document_semantics(
            _part(sdf_extension(EXT, 1, {"x": deep}, sdf_primitive("sphere", r=1.0))).to_dict()
        )


def test_the_json_schema_requires_a_fallback():
    import jsonschema

    from software_defined_matter.io import load_schema

    doc = _part(_node()).to_dict()
    jsonschema.validate(doc, load_schema("0.7"))
    del doc["materials"][0]["sdf_tree"]["fallback"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, load_schema("0.7"))


def test_a_payload_ref_must_name_a_declared_param():
    with pytest.raises(Exception, match="ghost"):
        validate(_part(_node(half={"$ref": "ghost"})))
    validate(
        _part(_node(half={"$ref": "h"}), h=Param("h", 2.0, unit="mm", bounds=(1.0, 3.0), free=True))
    )


def test_children_are_validated_at_their_declared_dimension():
    ok = _node(children=[sdf_primitive("circle_2d", r=1.0)], children_dim=2)
    validate(_part(ok))
    bad = _node(children=[sdf_primitive("sphere", r=1.0)], children_dim=2)
    with pytest.raises(SemanticValidationError):
        validate(_part(bad))


def test_a_registered_kind_in_the_wrong_dimension_is_rejected():
    extensions.register_extension(_spec(dim=2))
    with pytest.raises(SemanticValidationError, match="2-D"):
        validate(_part(_node()))


def test_the_kinds_own_validator_runs_and_can_refuse():
    def check(node, ctx):
        if node["payload"]["half"] < 0:
            ctx.fail(f"{ctx.path}.payload.half", "must not be negative")

    extensions.register_extension(_spec(validate=check))
    validate(_part(_node(2.0)))
    with pytest.raises(SemanticValidationError, match="negative"):
        validate(_part(_node(-1.0)))


def test_the_kinds_validator_can_check_a_public_subtree():
    def check(node, ctx):
        ctx.validate_subtree(node["payload"]["profile"], 2, f"{ctx.path}.payload.profile")

    extensions.register_extension(_spec(validate=check))
    good = sdf_extension(
        EXT, 1, {"half": 1.0, "profile": sdf_primitive("circle_2d", r=1.0)}, _node()["fallback"]
    )
    validate(_part(good))
    bad = sdf_extension(
        EXT, 1, {"half": 1.0, "profile": sdf_primitive("sphere", r=1.0)}, _node()["fallback"]
    )
    with pytest.raises(SemanticValidationError):
        validate(_part(bad))


# -- unavailable: the fallback stands in ----------------------------------


def test_unregistered_compile_uses_the_fallback_and_warns():
    node = _node(half=5.0, fallback_half=1.0)
    with pytest.warns(extensions.ExtensionFallbackWarning, match=EXT):
        d = _at(node, [3.0, 0.0, 0.0])
    assert d == pytest.approx(2.0)  # the fallback box, half 1, not the registered half 5


def test_registered_compile_uses_the_kind():
    extensions.register_extension(_spec())
    node = _node(half=5.0, fallback_half=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _at(node, [3.0, 0.0, 0.0]) == pytest.approx(-2.0)


def test_a_version_the_spec_does_not_read_falls_back():
    extensions.register_extension(_spec())
    node = _node(half=5.0, fallback_half=1.0)
    node["version"] = 2
    with pytest.warns(extensions.ExtensionFallbackWarning):
        assert _at(node, [3.0, 0.0, 0.0]) == pytest.approx(2.0)


def test_unavailable_with_a_live_payload_param_refuses_by_default():
    node = _node(half={"$ref": "h"})
    part = _part(node, h=Param("h", 2.0, unit="mm", bounds=(1.0, 3.0), free=True))
    with pytest.raises(ExtensionUnavailableError, match="'h'"):
        make_sdf_closure(node, part)
    with (
        extensions.policy(frozen_params="allow"),
        pytest.warns(extensions.ExtensionFallbackWarning),
    ):
        make_sdf_closure(node, part)


def test_unavailable_with_a_fixed_payload_param_does_not_refuse():
    node = _node(half={"$ref": "h"})
    part = _part(node, h=Param("h", 2.0, unit="mm"))
    with pytest.warns(extensions.ExtensionFallbackWarning):
        make_sdf_closure(node, part)


def test_extension_uses_reports_what_a_fallback_freezes():
    node = _node(half={"$ref": "h"})
    part = _part(node, h=Param("h", 2.0, unit="mm", bounds=(1.0, 3.0), free=True))
    (use,) = extensions.extension_uses(part)
    assert (use.id, use.version, use.registered, use.frozen_params) == (EXT, 1, False, ("h",))
    assert use.path == "materials[0].sdf_tree"
    extensions.register_extension(_spec())
    assert extensions.extension_uses(part)[0].frozen_params == ()


def test_unavailable_bbox_rate_features_and_envelope_use_the_fallback():
    node = _node(half=5.0, fallback_half=1.0)
    part = _part(node)
    assert infer_sdf_bbox(node, part) == ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    assert infer_sdf_max_rate(node, part) == 1.0
    assert infer_min_feature_size(node, part) == pytest.approx(2.0)
    env = infer_sdf_envelope(node, part)
    assert env["type"] == "primitive"


def test_a_cutter_fallback_follows_the_same_position_rule():
    plane = sdf_primitive("plane", n=[0.0, 0.0, 1.0], h=0.0)
    node = sdf_extension(EXT, 1, {}, plane)
    with pytest.raises(UnboundedRootError):
        validate(_part(node))
    validate(_part(sdf_op("subtract", [sdf_primitive("sphere", r=1.0), node])))


# -- registered: the hooks ------------------------------------------------


def test_registered_bbox_and_rate_come_from_the_kind():
    extensions.register_extension(_spec(max_rate=lambda n, c: math.inf))
    node = _node(half=5.0, fallback_half=1.0)
    part = _part(node)
    assert infer_sdf_bbox(node, part) == ((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0))
    assert infer_sdf_max_rate(node, part) == math.inf


def test_bbox_context_supplies_interval_helpers():
    seen = {}

    def bbox(node, ctx):
        seen["abs"] = ctx.scalar_abs_max({"$ref": "h"})
        seen["vec"] = ctx.vec_intervals([{"$ref": "h"}, 1.0], 2)
        seen["tree"] = ctx.bbox(sdf_primitive("sphere", r=2.0))
        seen["p2"] = ctx.bbox_2d_primitive(sdf_primitive("circle_2d", r=1.5))
        return ((-1.0,) * 3, (1.0,) * 3)

    extensions.register_extension(_spec(bbox=bbox))
    node = _node()
    infer_sdf_bbox(node, _part(node, h=Param("h", 2.0, unit="mm", bounds=(1.0, 3.0), free=True)))
    assert seen["abs"] == 3.0
    assert seen["vec"] == ((1.0, 3.0), (1.0, 1.0))
    assert seen["tree"] == ((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0))
    assert seen["p2"] == ((-1.5, -1.5), (1.5, 1.5))


def test_registered_features_and_envelope_hooks():
    extensions.register_extension(
        _spec(
            features=lambda n, c: [0.25],
            envelope=lambda n, c: sdf_primitive("sphere", r=9.0),
        )
    )
    node = _node()
    part = _part(node)
    assert infer_min_feature_size(node, part) == 0.25
    assert infer_sdf_envelope(node, part)["params"]["r"] == 9.0


def test_registered_without_envelope_or_features_is_loud_and_quiet_respectively():
    extensions.register_extension(_spec())
    node = _node()
    part = _part(node)
    with pytest.raises(EnvelopeInferenceError, match=EXT):
        infer_sdf_envelope(node, part)
    assert infer_min_feature_size(node, part) is None


def test_an_unbounded_kind_is_valid_only_as_a_cutter():
    extensions.register_extension(_spec(unbounded=True))
    with pytest.raises(UnboundedRootError):
        validate(_part(_node()))
    validate(_part(sdf_op("subtract", [sdf_primitive("sphere", r=1.0), _node()])))


def test_csg_like_children_follow_the_rules_of_that_op():
    plane = sdf_primitive("plane", n=[0.0, 0.0, 1.0], h=0.0)
    sphere = sdf_primitive("sphere", r=1.0)
    node = _node(children=[sphere, plane], children_dim=3)
    with pytest.raises(UnboundedRootError):
        validate(_part(node))
    extensions.register_extension(_spec(csg_like="subtract"))
    validate(_part(node))


def test_compile_child_and_nominal_are_available_to_the_kind():
    seen = {}

    def compile_(node, ctx):
        seen["nominal"] = dict(ctx.nominal)
        inner = ctx.compile_child(node["children"][0])
        return lambda p, free_vec: inner(p, free_vec) - 0.5

    extensions.register_extension(_spec(compile=compile_))
    node = _node(children=[sdf_primitive("sphere", r=1.0)], children_dim=3)
    part = _part(node, k=Param("k", 3.0), f=Param("f", 4.0, bounds=(1.0, 5.0), free=True))
    d = _at(node, [0.0, 0.0, 0.0], part)
    assert d == pytest.approx(-1.5)
    assert seen["nominal"] == {"k": 3.0, "f": 4.0}


def test_payload_param_gradient_flows_through_a_registered_kind():
    import jax

    extensions.register_extension(_spec())
    node = _node(half={"$ref": "h"})
    part = _part(node, h=Param("h", 2.0, unit="mm", bounds=(1.0, 3.0), free=True))
    fn = make_sdf_closure(node, part)
    g = jax.grad(lambda v: fn(jnp.asarray([[0.0, 0.0, 0.0]]), v)[0])(jnp.asarray([2.0]))
    assert float(g[0]) == pytest.approx(-1.0)


def test_layered_binding_serves_its_names_and_delegates_the_rest():
    part = _part(
        sdf_primitive("sphere", r=1.0),
        a=Param("a", 2.0, bounds=(1.0, 3.0), free=True),
        b=Param("b", 5.0),
    )
    binding = make_binding(part)
    free = binding.initial_free_vector()
    layered = binding.layered(["extra"], lambda name, v: jnp.asarray(7.0))
    assert float(layered.get("extra", free)) == 7.0
    assert float(layered.get("a", free)) == 2.0
    assert float(layered.get("b", free)) == 5.0
    assert layered.free_names == binding.free_names
    assert binding.nominal() == {"a": 2.0, "b": 5.0}


def test_refresh_fallbacks_rebuilds_only_where_the_kind_can():
    extensions.register_extension(
        _spec(make_fallback=lambda node, part: sdf_primitive("sphere", r=7.0))
    )
    part = _part(_node())
    assert extensions.refresh_fallbacks(part) == 1
    assert part.materials[0].sdf_tree["fallback"]["params"]["r"] == 7.0


# -- GLSL -----------------------------------------------------------------


def test_glsl_draws_the_fallback_and_reports_it():
    node = _node(half=5.0, fallback_half=1.0)
    part = _part(node)
    em = emit_glsl(part)
    assert em.fallbacks == [EXT]
    plain = _part(sdf_primitive("box", b=[1.0, 1.0, 1.0]))
    assert emit_glsl(plain).fallbacks == []
    assert em.scene_source.count("sdf_box") == emit_glsl(plain).scene_source.count("sdf_box")


def test_glsl_uses_the_fallback_even_when_the_kind_is_registered():
    extensions.register_extension(_spec())
    em = emit_glsl(_part(_node()))
    assert em.fallbacks == [EXT]


# -- nodes that refuse an extension ---------------------------------------


def test_regions_nested_in_an_extension_and_surface_finding_refuse_loudly():
    from software_defined_matter.kinematics import resolve_region_tree
    from software_defined_matter.material_surfaces import _supported

    inner = dict(sdf_primitive("sphere", r=1.0), name="inner")
    node = _node(children=[inner], children_dim=3)
    with pytest.raises(ValueError, match="unsupported"):
        resolve_region_tree(_part(node), {"$node": "inner"})
    with pytest.raises(ValueError, match="rate contract"):
        _supported(node)


# -- promotion ------------------------------------------------------------


def _to_native(node):
    if node["payload"].get("bad"):
        return None
    return sdf_primitive("sphere", r=node["payload"]["half"])


def test_a_promoted_extension_becomes_native_on_read_and_never_on_write(tmp_path):
    part = _part(_node(half=3.0))
    path = tmp_path / "p.sdm"
    part.save(str(path))
    raw = path.read_bytes()
    extensions.PROMOTIONS[EXT] = extensions.Promotion(1, _to_native)
    loaded = Part.load(str(path))
    assert loaded.materials[0].sdf_tree == sdf_primitive("sphere", r=3.0)
    assert path.read_bytes() == raw  # a read rewrote nothing
    assert min_schema_version_for(loaded) != "0.7"


def test_promotion_leaves_what_it_cannot_convert_and_newer_versions():
    extensions.PROMOTIONS[EXT] = extensions.Promotion(1, _to_native)
    unconvertible = sdf_extension(EXT, 1, {"half": 1.0, "bad": True}, sdf_primitive("sphere", r=1))
    newer = sdf_extension(EXT, 2, {"half": 1.0}, sdf_primitive("sphere", r=1))
    for node in (unconvertible, newer):
        out = extensions.promote_tree(copy.deepcopy(node))
        assert out["type"] == "extension"


def test_promotion_reaches_extensions_inside_other_nodes():
    extensions.PROMOTIONS[EXT] = extensions.Promotion(1, _to_native)
    tree = sdf_op("union", [sdf_primitive("sphere", r=1.0), _node(half=4.0)])
    out = extensions.promote_tree(tree)
    assert out["children"][1] == sdf_primitive("sphere", r=4.0)


def test_numpy_payload_values_do_not_break_the_validator():
    node = sdf_extension(EXT, 1, {"half": float(np.float32(2.0))}, sdf_primitive("sphere", r=1))
    validate(_part(node))
    assert bbox_mod is not None


# -- review fixes ----------------------------------------------------------


def test_the_guard_sees_params_referenced_only_from_children():
    import jax

    child = sdf_primitive("sphere", r={"$ref": "c"})
    node = _node(children=[child], children_dim=3)
    part = _part(node, c=Param("c", 2.0, unit="mm", bounds=(1.0, 3.0), free=True))
    with pytest.raises(ExtensionUnavailableError, match="'c'"):
        make_sdf_closure(node, part)
    (use,) = extensions.extension_uses(part)
    assert use.frozen_params == ("c",)
    with (
        extensions.policy(frozen_params="allow"),
        pytest.warns(extensions.ExtensionFallbackWarning),
    ):
        fn = make_sdf_closure(node, part)
    g = jax.grad(lambda v: fn(jnp.asarray([[0.0, 0.0, 0.0]]), v)[0])(jnp.asarray([2.0]))
    assert float(g[0]) == 0.0  # what the guard exists to prevent going unnoticed


def test_the_guard_ignores_params_the_fallback_also_uses():
    fallback = sdf_primitive("sphere", r={"$ref": "h"})
    node = sdf_extension(EXT, 1, {"half": {"$ref": "h"}}, fallback)
    part = _part(node, h=Param("h", 2.0, unit="mm", bounds=(1.0, 3.0), free=True))
    with pytest.warns(extensions.ExtensionFallbackWarning):
        make_sdf_closure(node, part)
    (use,) = extensions.extension_uses(part)
    assert use.frozen_params == ()


@pytest.mark.parametrize("rate", [0.0, -1.0, float("nan")])
def test_a_kind_cannot_report_a_nonsense_slope_bound(rate):
    extensions.register_extension(_spec(max_rate=lambda n, c: rate))
    node = _node()
    with pytest.raises(extensions.ExtensionError, match=EXT):
        infer_sdf_max_rate(node, _part(node))


def test_a_kind_cannot_report_an_inverted_bbox():
    extensions.register_extension(_spec(bbox=lambda n, c: ((1.0, 0.0, 0.0), (0.0, 1.0, 1.0))))
    node = _node()
    with pytest.raises(extensions.ExtensionError, match=EXT):
        infer_sdf_bbox(node, _part(node))


def test_refresh_refuses_a_fallback_that_is_not_a_valid_public_tree():
    part = _part(_node())
    extensions.register_extension(_spec(make_fallback=lambda n, p: _node()))
    with pytest.raises(extensions.ExtensionError, match="extension"):
        extensions.refresh_fallbacks(part)
    extensions.unregister_extension(EXT)
    extensions.register_extension(
        _spec(make_fallback=lambda n, p: sdf_primitive("circle_2d", r=1.0))
    )
    with pytest.raises(extensions.ExtensionError, match="3-D|dimension"):
        extensions.refresh_fallbacks(part)
