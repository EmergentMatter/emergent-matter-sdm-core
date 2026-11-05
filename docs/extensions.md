# Writing an extension

An extension is a node kind a separate package adds. The format and the rules are in [adr/0010-extension-nodes.md](adr/0010-extension-nodes.md); this page is how to write one.

## The node

Build it with `sdf_extension(id, version, payload, fallback, children=None, children_dim=None)`. Choose an id with your own prefix; `em.` is a naming convention for Emergent Matter packages, and nothing enforces it. Keep everything the kind needs in `payload`. Put a subtree in `children` only when other tools should be able to see it as geometry. Make `fallback` the closest tree you can express in the public node kinds. If the kind fits the node's position as a cutter, the fallback has to be valid as a cutter there too.

## The spec

```python
from software_defined_matter.extensions import ExtensionSpec, register_extension

spec = ExtensionSpec(
    id="acme.slab",
    versions=frozenset({1}),
    dim=3,
    compile=compile_slab,    # (node, CompileContext) -> (points, free_vec) -> distance
    bbox=bbox_slab,          # (node, BBoxContext) -> ((x0, y0, z0), (x1, y1, z1))
    max_rate=lambda node, ctx: 1.0,   # return math.inf when no bound is known
)
register_extension(spec)
```

Call `register_extension` from a function the user calls, never at import. `register_extension` returns `False` for an id this package has made native; treat that as success.

Optional hooks: `validate`, `features`, `envelope`, `make_fallback`, `csg_like` (children follow the rules of a union, subtract or intersect) and `unbounded` (valid only as a cutter, like `plane`). Without `envelope`, an envelope request on a tree holding the kind raises and names it.

## Rules

- Read parameters through `ctx.resolve(value, free_vec)` so `$ref` values and gradients work.
- `ctx.binding.layered(names, getter)` evaluates a subtree with values of your own, such as a number interpolated per point.
- A payload is untrusted input. Check it in `validate`.
- Raise the `version` when the payload's shape changes. A spec reads only the versions it lists; any other version falls back.
- Do not rely on anything outside the contexts.

## Testing

Register in a fixture and unregister after it (`unregister_extension`). Test the document with nothing registered too: it should validate and compile to the fallback. `extension_uses(part)` reports which nodes are available and which parameters a fallback ignores.
