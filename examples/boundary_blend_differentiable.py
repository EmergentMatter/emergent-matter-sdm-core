"""Proof that ``boundary_blend`` is differentiable, the way sdm-core expects
of a primitive (docs/optimization.md: autodiff through ``make_sdf_closure``
under ``jax.grad`` / ``jax.jit``, and finite-difference spot checks on the
critical parameters of a new primitive / objective combination).

The part: a soap bar cut from a block by one boundary-blend skin (one
quarter, folded to eight). FIVE free parameters drive its control points:

    length    half-length of the bar (mm)
    width     half-width (mm)
    crown     height of the top (mm)
    shoulder  how high up the wall the profile turns over (ratio)
    corner    how square the plan corners are (ratio)

It then
  1. saves the part, reloads it, validates it and compiles it (jit),
  2. checks d(volume)/d(param) and d(area)/d(param) from jax.grad against
     central finite differences, for every parameter, through sdm-core's own
     `volume` / `surface_area` metrics,
  3. sweeps crown height and draws the gradient as tangent lines on the
     volume curve,
  4. uses the gradients (Newton on two equations) to find the width and
     crown that give a target volume and area at once,
  5. measures the distance rate at the surface (|grad_p d| ~ 1),
  6. writes viewing copies of three variants (pushed, pulled, squeezed).

    uv run python examples/boundary_blend_differentiable.py
"""

from __future__ import annotations

import json
import time
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
WALL = 7.0  # top of the straight wall (fixed)
# name: (nominal, lower, upper, unit)
PARAMS = {
    "length": (45.0, 36.0, 54.0, "mm"),
    "width": (30.0, 22.0, 36.0, "mm"),
    "crown": (14.0, 10.0, 18.0, "mm"),
    "shoulder": (0.60, 0.35, 0.80, "ratio"),
    "corner": (0.72, 0.58, 0.88, "ratio"),
}
NAMES = list(PARAMS)


def P(n):
    return expr_param(n)


def mul(a, b):
    return expr_binop("*", a, b)


def add(a, b):
    return expr_binop("+", a, b)


def num(x):
    return expr_num(x)


def build(values=None):
    values = values or {}
    zs = add(num(WALL), mul(P("shoulder"), add(P("crown"), num(-WALL))))
    a, b, h, k = P("length"), P("width"), P("crown"), P("corner")
    c1 = [[0, 0, h], [mul(num(0.55), a), 0, h], [a, 0, zs], [a, 0, WALL]]  # long profile
    c2 = [[a, 0, WALL], [a, mul(k, b), WALL], [mul(k, a), b, WALL], [0, b, WALL]]  # plan
    c3 = [[0, b, WALL], [0, b, zs], [0, mul(num(0.55), b), h], [0, 0, h]]  # short profile
    conds = [("normal", (0, 1, 0)), ("normal", (0, 0, 1)), ("normal", (1, 0, 0))]
    skin = sdf_boundary_blend([c1, c2, c3], conds, reach=60.0, fold="xyz")
    block = sdf_primitive("box", b=[60.0, 42.0, 21.0])
    tree = sdf_op("subtract", [block, skin])
    params = {
        n: Param(name=n, value=values.get(n, v), free=True, bounds=(lo, hi), unit=u)
        for n, (v, lo, hi, u) in PARAMS.items()
    }
    return Part(
        name="soap_bar_parametric",
        params=params,
        materials=[MaterialRegion(material_id=1, name="soap", sdf_tree=tree)],
        metadata={"units": "mm", "metric_voxel_size": 1.5, "bbox": [[-61, -43, -22], [61, 43, 22]]},
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
    save(part, OUT / "soap_bar_parametric.sdm")
    part = load_part(OUT / "soap_bar_parametric.sdm")
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
        e = 0.02 if PARAMS[n][3] == "mm" else 0.002
        dx = jnp.zeros(len(NAMES)).at[i].set(e)
        fdV = (float(volume(x0 + dx)) - float(volume(x0 - dx))) / (2 * e)
        fdA = (float(area(x0 + dx)) - float(area(x0 - dx))) / (2 * e)
        rows.append((n, gV[i], fdV, gA[i], fdA))
        say(
            f"  d/d{n:9s} volume: grad {gV[i]:10.1f}  finite diff {fdV:10.1f}  "
            f"({100 * abs(gV[i] - fdV) / max(abs(fdV), 1e-9):.2f} % apart) | "
            f"area: grad {gA[i]:8.1f}  finite diff {fdA:8.1f}  "
            f"({100 * abs(gA[i] - fdA) / max(abs(fdA), 1e-9):.2f} % apart)"
        )

    # 3. crown sweep: volume curve and the gradient as tangents ---------------
    ic = NAMES.index("crown")
    hs = np.linspace(10.0, 18.0, 17)
    vs = [float(volume(x0.at[ic].set(h))) for h in hs]
    th = [11.0, 14.0, 17.0]
    tangents = [
        (h, float(volume(x0.at[ic].set(h))), float(jax.grad(volume)(x0.at[ic].set(h))[ic]))
        for h in th
    ]

    # 4. Newton with the gradients: width + crown for a target volume AND area
    targets = np.array([v0 * 0.85, a0 * 0.90])
    iw = NAMES.index("width")
    x = x0
    hist = []
    for it in range(8):
        r = np.array([float(volume(x)), float(area(x))]) - targets
        hist.append((it, float(x[iw]), float(x[ic]), *(r + targets)))
        if np.all(np.abs(r / targets) < 1e-4):
            break
        Jv, Ja = np.asarray(jax.grad(volume)(x)), np.asarray(jax.grad(area)(x))
        J = np.array([[Jv[iw], Jv[ic]], [Ja[iw], Ja[ic]]])
        step = np.linalg.solve(J, -r)
        x = x.at[iw].add(step[0]).at[ic].add(step[1])
    say(
        f"Newton on (width, crown) for volume {targets[0]:.0f} and area {targets[1]:.0f}: "
        f"{len(hist) - 1} steps -> width {float(x[iw]):.3f}, crown {float(x[ic]):.3f}, "
        f"volume {hist[-1][3]:.0f}, area {hist[-1][4]:.0f}"
    )

    # 5. the distance rate at the surface --------------------------------------
    pts = np.asarray(sdf(jnp.zeros((1, 3))))  # warm
    rng = np.random.default_rng(0)
    cand = rng.uniform([-50, -35, -16], [50, 35, 16], (40000, 3))
    d = np.asarray(sdf(jnp.asarray(cand)))
    near = cand[np.argsort(np.abs(d))[:400]]
    g = np.asarray(jax.vmap(jax.grad(lambda q: sdf(q[None])[0]))(jnp.asarray(near)))
    rate = np.linalg.norm(g, axis=1)
    say(f"distance rate at the surface (400 points): {rate.min():.4f} .. {rate.max():.4f}")
    del pts

    # 6. viewing copies of three variants ---------------------------------------
    from software_defined_matter.sdf.bake import bake_raster_field

    variants = {
        "nominal": {},
        "pushed_squeezed": {"width": 23.0, "crown": 17.5, "corner": 0.85},
        "pulled_flat": {"length": 53.0, "crown": 10.5, "shoulder": 0.4, "corner": 0.6},
    }
    import warnings

    for name, over in variants.items():
        vp = build({n: over.get(n, PARAMS[n][0]) for n in NAMES})
        f = jax.jit(make_sdf_closure(vp.materials[0].sdf_tree, vp))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = bake_raster_field(f, ([-57, -40, -19], [57, 40, 19]), 0.4, encoding="f16le")
        view = Part(
            name=f"soap_{name}",
            materials=[MaterialRegion(material_id=1, name="soap", sdf_tree=r)],
            metadata={"units": "mm"},
        ).to_dict()
        validate(view)
        (OUT / f"soap_{name}_view.sdm").write_text(json.dumps(view))
    say("wrote viewing copies: " + ", ".join(f"soap_{n}_view.sdm" for n in variants))

    # one point on the shoulder: its slope vs crown, from jax.grad and from FD
    q = jnp.array([[30.0, 22.0, 11.0]])
    fq = jax.jit(lambda x: sdf(q, x)[0])
    gq = jax.jit(jax.grad(fq))
    hs2 = np.linspace(10.0, 18.0, 33)
    point = {
        "h": list(hs2),
        "p": [30, 22, 11],
        "grad": [float(gq(x0.at[ic].set(h))[ic]) for h in hs2],
        "fd": [
            (float(fq(x0.at[ic].set(h + 0.02))) - float(fq(x0.at[ic].set(h - 0.02)))) / 0.04
            for h in hs2
        ],
    }
    (OUT / "results.json").write_text(
        json.dumps(
            {
                "point_sweep": point,
                "rows": [list(map(float, r[1:])) for r in rows],
                "names": NAMES,
                "sweep": [list(hs), vs],
                "tangents": tangents,
                "newton": hist,
                "targets": list(targets),
            }
        )
    )
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
    ax.set_xlabel("crown height (mm)", color=ink2, fontsize=9)
    ax.set_ylabel("d(field)/d(crown) at the point", color=ink2, fontsize=9)
    ax.set_title(
        f"Push the crown up and watch one point on the shoulder {tuple(P['p'])}:\n"
        f"the slope changes as the shape does; the two agree to {worst:.4f} everywhere",
        color=ink,
        fontsize=10,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=8, labelcolor=ink, loc="upper left")
    ax = axs[1]
    names = R["names"]
    rows = np.array(R["rows"])
    err_v = 100 * np.abs(rows[:, 0] - rows[:, 1]) / np.maximum(np.abs(rows[:, 1]), 1e-9)
    err_a = 100 * np.abs(rows[:, 2] - rows[:, 3]) / np.maximum(np.abs(rows[:, 3]), 1e-9)
    y = np.arange(len(names))
    ax.barh(y + 0.18, err_v, 0.34, color=blue, label="d volume")
    ax.barh(y - 0.18, err_a, 0.34, color=orange, label="d area")
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel("gradient vs finite difference, % apart", color=ink2, fontsize=9)
    ax.set_title(
        "All five parameters: jax.grad agrees with nudge-and-re-measure",
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
