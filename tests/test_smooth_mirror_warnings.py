"""Smooth mirrors require exact input distances on both sides, just like smooth unions."""

from __future__ import annotations

import json
import warnings

import jax
import jax.numpy as jnp
import pytest

from software_defined_matter import MaterialRegion, Part, sdf_primitive, sdf_transform
from software_defined_matter.dsl.resolve import make_binding
from software_defined_matter.io import load_part
from software_defined_matter.sdf.compile import make_sdf_closure, make_sdf_closure_with_binding
from software_defined_matter.sdf.exactness import DistanceAccuracyWarning


def _mirror(child):
    return sdf_transform("mirror", child, n=[1, 0, 0], o=[0, 0, 0])


def _part(tree, smooth=True):
    return Part(
        name="mirror-warning",
        materials=[MaterialRegion(material_id=1, name="body", sdf_tree=tree)],
        metadata={"smooth_csg": smooth},
    )


@pytest.mark.parametrize("entry", ["part", "binding"])
@pytest.mark.parametrize("kind", ["lower-bound", "unknown", "exact-outside-only"])
def test_smooth_mirror_compilation_warns_once_for_unproven_input_sides(entry, kind):
    if kind == "lower-bound":
        child = sdf_primitive("ellipsoid", r=[1, 2, 3])
    elif kind == "unknown":
        child = sdf_primitive("round_box", b=[1, 1, 1], r=2)
    else:
        child = sdf_primitive("capped_torus", sc=[0.6, 0.8], ra=2, rb=0.3)
    tree = _mirror(child)
    part = _part(tree)
    with pytest.warns(
        DistanceAccuracyWarning, match="sdf_tree: mirror blends non-exact fields"
    ) as caught:
        if entry == "part":
            fn = make_sdf_closure(tree, part)
        else:
            fn = make_sdf_closure_with_binding(tree, make_binding(part), b_smooth_csg=True)
    assert len(caught) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error", DistanceAccuracyWarning)
        point = jnp.array([4.0, 1, 1])
        value = fn(point)
        assert bool(jnp.isfinite(value))
        assert float(jax.jit(fn)(point)) == pytest.approx(float(value), abs=2e-6)
        assert bool(jnp.all(jnp.isfinite(jax.jit(jax.grad(fn))(point))))


def test_validated_load_warns_at_the_nested_mirror_path(tmp_path):
    tree = sdf_transform("translate", _mirror(sdf_primitive("ellipsoid", r=[1, 2, 3])), t=[1, 0, 0])
    path = tmp_path / "mirror.sdm"
    path.write_text(json.dumps(_part(tree).to_dict()))
    with pytest.warns(
        DistanceAccuracyWarning, match=r"materials\[0\].sdf_tree.child: mirror"
    ) as caught:
        load_part(path)
    assert len(caught) == 1


def test_compiler_smoothing_overrides_control_mirror_diagnostics():
    tree = _mirror(sdf_primitive("ellipsoid", r=[1, 2, 3]))
    part = _part(tree)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DistanceAccuracyWarning)
        make_sdf_closure(tree, part, b_smooth_csg=False)
        # A binding-only compiler has no metadata and defaults to hard CSG.
        make_sdf_closure_with_binding(tree, make_binding(part))
    part.metadata["smooth_csg"] = False
    with pytest.warns(DistanceAccuracyWarning, match="mirror blends non-exact fields"):
        make_sdf_closure(tree, part, b_smooth_csg=True)


def test_exact_smooth_mirrors_and_hard_approximate_mirrors_do_not_warn(tmp_path):
    for smooth, child in [
        (True, sdf_primitive("sphere", r=1)),
        (False, sdf_primitive("ellipsoid", r=[1, 2, 3])),
    ]:
        tree = _mirror(child)
        part = _part(tree, smooth=smooth)
        path = tmp_path / "mirror.sdm"
        path.write_text(json.dumps(part.to_dict()))
        with warnings.catch_warnings():
            warnings.simplefilter("error", DistanceAccuracyWarning)
            load_part(path)
            make_sdf_closure(tree, part)
            make_sdf_closure_with_binding(tree, make_binding(part), b_smooth_csg=smooth)
