"""Large shared parameter graphs retain current values without cross-call caches."""

from __future__ import annotations

import pytest

from software_defined_matter import Param, Part
from software_defined_matter.dsl.expr import expr_binop, expr_num, expr_param


def test_large_shared_graph_refreshes_every_branch_after_base_changes() -> None:
    params = {"width": Param("width", 8.0, free=True, unit="mm")}
    previous = "width"
    for index in range(1200):
        name = f"derived_{index}"
        params[name] = Param(
            name,
            0.0,
            unit="mm",
            expr=expr_binop("+", expr_param(previous), expr_num(0.25)),
        )
        previous = name
    part = Part(name="shared_graph", params=params)
    # A quarter millimetre is exactly representable at this magnitude.
    assert part.derived_values()[previous] == pytest.approx(308.0, abs=1e-6)
    part.params["width"].value = 16.0
    part.refresh_derived()
    assert part.params[previous].value == pytest.approx(316.0, abs=1e-6)


def test_shared_diamond_graph_evaluates_out_of_insertion_order() -> None:
    part = Part(
        name="diamond",
        params={
            "result": Param(
                "result",
                0.0,
                unit="mm",
                expr=expr_binop("+", expr_param("left"), expr_param("right")),
            ),
            "right": Param(
                "right", 0.0, unit="mm", expr=expr_binop("*", expr_param("shared"), expr_num(3.0))
            ),
            "left": Param("left", 0.0, unit="mm", expr=expr_param("shared")),
            "shared": Param(
                "shared", 0.0, unit="mm", expr=expr_binop("/", expr_param("width"), expr_num(2.0))
            ),
            "width": Param("width", 10.0, free=True, unit="mm"),
        },
    )
    assert part.derived_values()["result"] == pytest.approx(20.0, abs=1e-6)
    part.params["width"].value = 3.0
    assert part.derived_values()["result"] == pytest.approx(6.0, abs=1e-6)
