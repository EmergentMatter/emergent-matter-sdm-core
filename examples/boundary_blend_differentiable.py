"""A part only ``boundary_blend`` can make, and proof that it is differentiable
the way sdm-core expects of a primitive (docs/optimization.md: autodiff
through ``make_sdf_closure`` under ``jax.grad`` / ``jax.jit``, with
finite-difference spot checks on a new primitive's critical parameters).

THE PART: a saddle seat. One 4-curve boundary blend over the seat's plan,
with a 4 x 4 network of curves (the four top edges of the walls plus two
guides each way), cutting a block. Its edges are 3-D curves, rising to the
pommel and cantle and sagging at the sides, and there is a dished pocket to
sit in that the edges know nothing about. No stack of flat sections (a loft)
follows edges like that: this is what boundary_blend is for. (A shape that IS
a stack of flat sections, e.g. a soap bar or a bottle, is cheaper as a loft:
see the cost note in sdf.sdf_boundary_blend.)

SIX free parameters drive every control point:

    half_length, half_width   the seat's plan (mm)
    pommel, cantle            how far the front and back rise (mm)
    sag                       how far the sides fall (mm)
    pocket                    how deep the sit-bone pocket is (mm)

It then
  1. saves the part, reloads it, validates it and compiles it (jit),
  2. checks d(volume)/d(param) and d(area)/d(param) from jax.grad against
     central finite differences, for every parameter, through sdm-core's own
     `volume` / `surface_area` metrics,
  3. sweeps the pocket depth and compares the slope of the field at a point
     in the pocket from jax.grad with finite differences at every step,
  4. uses the gradients (Newton on two equations) to find the pocket depth
     and side sag that give a target volume and area at once,
  5. measures the distance rate at the surface (|grad_p d| ~ 1),
  6. writes viewing copies of three variants.

    uv run python examples/boundary_blend_differentiable.py
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from software_defined_matter import (
    MaterialRegion,
    Param,
    Part,
    load_part,
    save,
    sdf_boundary_blend,
    sdf_op,
    sdf_primitive,
    sdf_transform,
    validate,
)
from software_defined_matter.dsl.expr import (
    compile_expr,
    expr_binop,
    expr_metric,
    expr_num,
    expr_param,
)
from software_defined_matter.sdf.compile import make_sdf_closure

OUT = Path(__file__).resolve().parent / "boundary_blend_demo_out"
BASE = 12.0  # seat height before any rise or fall (mm)
BLOCK_Z = 40.0
# name: (nominal, lower, upper)
PARAMS = {
    "half_length": (55.0, 45.0, 65.0),
    "half_width": (35.0, 28.0, 42.0),
    "pommel": (10.0, 4.0, 16.0),
    "cantle": (13.0, 6.0, 20.0),
    "sag": (9.0, 3.0, 12.0),
    "pocket": (8.0, 2.0, 12.0),
}
NAMES = list(PARAMS)
XI = np.array([-1.0, -1 / 3, 1 / 3, 1.0])  # the network's stations, as fractions of the plan
T4 = np.array([0.0, 1 / 3, 2 / 3, 1.0])
MINV = np.linalg.inv(
    np.stack([(1 - T4) ** 3, 3 * T4 * (1 - T4) ** 2, 3 * T4**2 * (1 - T4), T4**3], axis=1)
)


def height_terms(xi, eta):
    """The seat height at plan fraction (xi, eta) as BASE + sum(coef * param):
    front and back rise, side fall, and a pocket (deeper toward the back)
    that is zero on the edges."""
    return {
        "pommel": ((1 + xi) / 2) ** 2,
        "cantle": ((1 - xi) / 2) ** 2,
        "sag": -(eta**2),
        "pocket": -(1 - xi**2) * (1 - eta**2) * (1 - 0.6 * xi),
    }


def _lin(const, terms):
    """const + sum(coef * param) as an expression tree."""
    e = expr_num(const)
    for name, c in terms.items():
        if abs(c) > 1e-12:
            e = expr_binop("+", e, expr_binop("*", expr_num(float(c)), expr_param(name)))
    return e


def _curve(points):
    """Cubic Bezier through 4 points (fractions xi, eta) at t = 0, 1/3, 2/3, 1:
    control points as expressions in the parameters (the solve is linear)."""
    ctrl = []
    for k in range(4):
        w = MINV[k]
        cx = float(sum(w[i] * points[i][0] for i in range(4)))
        cy = float(sum(w[i] * points[i][1] for i in range(4)))
        terms: dict[str, float] = {}
        for i in range(4):
            for name, c in height_terms(*points[i]).items():
                terms[name] = terms.get(name, 0.0) + w[i] * c
        const = BASE * float(sum(w))
        ctrl.append(
            [
                expr_binop("*", expr_num(cx), expr_param("half_length")),
                expr_binop("*", expr_num(cy), expr_param("half_width")),
                _lin(const, terms),
            ]
        )
    return ctrl


def network():
    def along(eta):  # runs front-back at fixed eta
        return _curve([(x, eta) for x in XI])

    def across(xi):  # runs side to side at fixed xi
        return _curve([(xi, y) for y in XI])

    # boundary loop, counter-clockwise from above: side, front, other side, back
    boundary = [along(-1.0), across(1.0), along(1.0)[::-1], across(-1.0)[::-1]]
    guides = [along(XI[1]), along(XI[2]), across(XI[1]), across(XI[2])]
    return boundary, guides


def build(values=None):
    values = values or {}
    boundary, guides = network()
    skin = sdf_boundary_blend(boundary, guides=guides, reach=80.0)
    # The block: a fixed box trimmed to the seat's plan by four planes whose
    # offsets are the parameters. (A box sized by the parameters directly
    # would do, but sdm-core's surface_area gradient with respect to a box's
    # own size is NaN today, independent of this primitive.)
    a, b = expr_param("half_length"), expr_param("half_width")
    neg_a = expr_binop("*", expr_num(-1.0), a)
    neg_b = expr_binop("*", expr_num(-1.0), b)
    box = sdf_transform(
        "translate", sdf_primitive("box", b=[70.0, 46.0, BLOCK_Z / 2]), t=[0.0, 0.0, BLOCK_Z / 2]
    )
    walls = [
        sdf_primitive("plane", n=[1.0, 0.0, 0.0], h=neg_a),  # x <= a
        sdf_primitive("plane", n=[-1.0, 0.0, 0.0], h=neg_a),  # x >= -a
        sdf_primitive("plane", n=[0.0, 1.0, 0.0], h=neg_b),  # y <= b
        sdf_primitive("plane", n=[0.0, -1.0, 0.0], h=neg_b),  # y >= -b
    ]
    block = sdf_op("intersect", [box, *walls])
    tree = sdf_op("subtract", [block, skin])
    params = {
        n: Param(name=n, value=values.get(n, v), free=True, bounds=(lo, hi), unit="mm")
        for n, (v, lo, hi) in PARAMS.items()
    }
    return Part(
        name="saddle_seat_parametric",
        params=params,
        materials=[MaterialRegion(material_id=1, name="seat", sdf_tree=tree)],
        metadata={
            "units": "mm",
            "metric_voxel_size": 1.5,
            "bbox": [[-71.0, -47.0, -1.0], [71.0, 47.0, BLOCK_Z + 1.0]],
        },
    )


def vec(values):
    return jnp.asarray([values[n] for n in NAMES], dtype=float)


def main():
    OUT.mkdir(exist_ok=True)
    log = []

    def say(s):
        print(s, flush=True)
        log.append(s)

    # 1. save / reload / validate / compile -----------------------------------
    part = build()
    save(part, OUT / "saddle_seat_parametric.sdm")
    part = load_part(OUT / "saddle_seat_parametric.sdm")
    validate(part.to_dict())
    say(
        f"saved, reloaded and validated: schema {part.to_dict()['schema_version']}, "
        f"{len(part.free_params())} free parameters"
    )
    sdf = make_sdf_closure(part.materials[0].sdf_tree, part)
    volume = jax.jit(compile_expr(expr_metric("volume"), part, sdf))
    area = jax.jit(compile_expr(expr_metric("surface_area"), part, sdf))
    x0 = vec({n: v[0] for n, v in PARAMS.items()})
    t = time.time()
    v0, a0 = float(volume(x0)), float(area(x0))
    dt = time.time() - t
    say(f"nominal: volume {v0:.0f} mm^3, area {a0:.0f} mm^2 ({dt:.0f} s to compile + run)")

    # 2. jax.grad vs central finite differences, every parameter -------------
    t = time.time()
    gV = np.asarray(jax.grad(volume)(x0))
    gA = np.asarray(jax.grad(area)(x0))
    say(f"jax.grad of volume and area: {time.time() - t:.0f} s")
    rows = []
    for i, n in enumerate(NAMES):
        e = 0.02
        dx = jnp.zeros(len(NAMES)).at[i].set(e)
        fdV = (float(volume(x0 + dx)) - float(volume(x0 - dx))) / (2 * e)
        fdA = (float(area(x0 + dx)) - float(area(x0 - dx))) / (2 * e)
        rows.append((n, gV[i], fdV, gA[i], fdA))
        say(
            f"  d/d{n:11s} volume: grad {gV[i]:10.1f}  finite diff {fdV:10.1f}  "
            f"({100 * abs(gV[i] - fdV) / max(abs(fdV), 1e-9):.2f} % apart) | "
            f"area: grad {gA[i]:8.1f}  finite diff {fdA:8.1f}  "
            f"({100 * abs(gA[i] - fdA) / max(abs(fdA), 1e-9):.2f} % apart)"
        )

    # 3. pocket sweep: the field's slope at a point in the pocket -------------
    ik = NAMES.index("pocket")
    q = jnp.array([[-10.0, 3.0, 8.0]])
    fq = jax.jit(lambda x: sdf(q, x)[0])
    gq = jax.jit(jax.grad(fq))
    ks = np.linspace(2.0, 12.0, 21)
    point = {
        "h": list(ks),
        "p": [-10, 3, 8],
        "grad": [float(gq(x0.at[ik].set(k))[ik]) for k in ks],
        "fd": [
            (float(fq(x0.at[ik].set(k + 0.02))) - float(fq(x0.at[ik].set(k - 0.02)))) / 0.04
            for k in ks
        ],
    }
    worst = max(abs(a - b) for a, b in zip(point["grad"], point["fd"], strict=True))
    say(
        f"pocket sweep at {tuple(point['p'])}: slope {min(point['grad']):.3f} .. "
        f"{max(point['grad']):.3f}, worst |grad - fd| {worst:.5f}"
    )

    # 4. Newton with the gradients: pocket + sag for a target volume AND area
    targets = np.array([v0 * 1.04, a0 * 0.99])
    i_s = NAMES.index("sag")
    x = x0
    hist = []
    for it in range(8):
        r = np.array([float(volume(x)), float(area(x))]) - targets
        hist.append((it, float(x[ik]), float(x[i_s]), *(r + targets)))
        if np.all(np.abs(r / targets) < 1e-4):
            break
        Jv, Ja = np.asarray(jax.grad(volume)(x)), np.asarray(jax.grad(area)(x))
        J = np.array([[Jv[ik], Jv[i_s]], [Ja[ik], Ja[i_s]]])
        step = np.linalg.solve(J, -r)
        x = x.at[ik].add(step[0]).at[i_s].add(step[1])
    say(
        f"Newton on (pocket, sag) for volume {targets[0]:.0f} and area {targets[1]:.0f}: "
        f"{len(hist) - 1} steps -> pocket {float(x[ik]):.3f}, sag {float(x[i_s]):.3f}, "
        f"volume {hist[-1][3]:.0f}, area {hist[-1][4]:.0f}"
    )

    # 5. the distance rate at the surface --------------------------------------
    rng = np.random.default_rng(0)
    cand = rng.uniform([-54, -34, 1], [54, 34, 39], (40000, 3))
    d = np.asarray(sdf(jnp.asarray(cand)))
    near = cand[np.argsort(np.abs(d))[:400]]
    g = np.asarray(jax.vmap(jax.grad(lambda p: sdf(p[None])[0]))(jnp.asarray(near)))
    rate = np.linalg.norm(g, axis=1)
    say(f"distance rate at the surface (400 points): {rate.min():.4f} .. {rate.max():.4f}")

    # 6. viewing copies of three variants ---------------------------------------
    from software_defined_matter.sdf.bake import bake_raster_field

    variants = {
        "nominal": {},
        "deep_pocket_high_cantle": {"pocket": 12.0, "cantle": 19.0, "sag": 11.0},
        "long_flat": {"half_length": 64.0, "pommel": 5.0, "cantle": 7.0, "pocket": 3.0},
    }
    for name, over in variants.items():
        vp = build({n: over.get(n, PARAMS[n][0]) for n in NAMES})
        f = jax.jit(make_sdf_closure(vp.materials[0].sdf_tree, vp))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = bake_raster_field(f, ([-66, -43, -1], [66, 43, BLOCK_Z + 1]), 0.4, encoding="f16le")
        region = MaterialRegion(material_id=1, name="seat", sdf_tree=r)
        view = Part(name=f"seat_{name}", materials=[region], metadata={"units": "mm"}).to_dict()
        validate(view)
        (OUT / f"seat_{name}_view.sdm").write_text(json.dumps(view))
    say("wrote viewing copies: " + ", ".join(f"seat_{n}_view.sdm" for n in variants))

    results = {
        "point_sweep": point,
        "rows": [list(map(float, r[1:])) for r in rows],
        "names": NAMES,
        "newton": hist,
        "targets": list(targets),
    }
    (OUT / "results.json").write_text(json.dumps(results))
    (OUT / "results.txt").write_text("\n".join(log) + "\n")
    plot()


def plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    R = json.loads((OUT / "results.json").read_text())
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e6e5e1"
    blue, orange = "#2a78d6", "#eb6834"
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), facecolor="#fcfcfb")
    for ax in axs:
        ax.set_facecolor("#fcfcfb")
        ax.grid(color=grid, lw=0.6)
        for s in ax.spines.values():
            s.set_color(grid)
        ax.tick_params(colors=ink2, labelsize=8)
    ax = axs[0]
    P = R["point_sweep"]
    ax.plot(P["h"], P["grad"], "-", color=blue, lw=2, label="slope from jax.grad")
    ax.plot(P["h"], P["fd"], "o", color=orange, ms=4, label="slope from nudge-and-re-measure")
    worst = max(abs(a - b) for a, b in zip(P["grad"], P["fd"], strict=True))
    ax.set_xlabel("pocket depth (mm)", color=ink2, fontsize=9)
    ax.set_ylabel("d(field)/d(pocket) at the point", color=ink2, fontsize=9)
    ax.set_title(
        f"Deepen the seat's pocket and watch one point in it {tuple(P['p'])}:\n"
        f"the two slopes agree to {worst:.4f} at every step",
        color=ink,
        fontsize=10,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=8, labelcolor=ink, loc="best")
    ax = axs[1]
    names = R["names"]
    rows = np.array(R["rows"])
    # error as a share of that metric's largest sensitivity, so a parameter
    # that barely moves a metric is not read as a big relative miss
    err_v = 100 * np.abs(rows[:, 0] - rows[:, 1]) / np.abs(rows[:, 1]).max()
    err_a = 100 * np.abs(rows[:, 2] - rows[:, 3]) / np.abs(rows[:, 3]).max()
    y = np.arange(len(names))
    ax.barh(y + 0.18, err_v, 0.34, color=blue, label="d volume")
    ax.barh(y - 0.18, err_a, 0.34, color=orange, label="d area")
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel(
        "|gradient - finite difference|, % of that metric's largest sensitivity",
        color=ink2,
        fontsize=9,
    )
    ax.set_title(
        "All six parameters: jax.grad agrees with nudge-and-re-measure",
        color=ink,
        fontsize=10,
        loc="left",
    )
    for i in range(len(names)):
        ax.text(err_v[i], i + 0.18, f" {err_v[i]:.2f}%", va="center", fontsize=8, color=ink2)
        ax.text(err_a[i], i - 0.18, f" {err_a[i]:.2f}%", va="center", fontsize=8, color=ink2)
    ax.legend(frameon=False, fontsize=8, labelcolor=ink, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "differentiability.png", dpi=130, facecolor=fig.get_facecolor())
    print("wrote", OUT / "differentiability.png")


if __name__ == "__main__":
    main()
