"""`enable_compile_cache` points JAX at a directory, honours its overrides, and caches."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

from software_defined_matter.compile_cache import ENV_VAR

SCRIPT = textwrap.dedent(
    """
    import sys
    from software_defined_matter import enable_compile_cache
    where = enable_compile_cache("sdm-test", sys.argv[1] if len(sys.argv) > 1 else None)
    import jax, jax.numpy as jnp
    print("WHERE", where, jax.config.jax_compilation_cache_dir)
    jax.jit(lambda x: jnp.sin(x) * 3.0 + x ** 2)(jnp.arange(16.0)).block_until_ready()
    """
)


def _run(args: list[str], env: dict[str, str]) -> str:
    out = subprocess.run(
        [sys.executable, "-c", SCRIPT, *args], capture_output=True, text=True, env=env, check=True
    )
    return next(line for line in out.stdout.splitlines() if line.startswith("WHERE"))


def test_a_compile_lands_in_the_named_directory(tmp_path: Path) -> None:
    import os

    env = {k: v for k, v in os.environ.items() if k != ENV_VAR}
    _run([str(tmp_path / "c")], env)
    assert any((tmp_path / "c").iterdir()), "the compiled function was written to the cache"


def test_the_environment_overrides_and_zero_disables(tmp_path: Path) -> None:
    import os

    env = {**os.environ, ENV_VAR: str(tmp_path / "env")}
    assert str(tmp_path / "env") in _run([str(tmp_path / "arg")], env)
    env[ENV_VAR] = "0"
    assert "WHERE None" in _run([str(tmp_path / "arg")], env)
    assert not (tmp_path / "arg").exists()


def test_the_default_is_under_the_home_cache(tmp_path: Path) -> None:
    import os

    env = {**{k: v for k, v in os.environ.items() if k != ENV_VAR}, "HOME": str(tmp_path)}
    line = _run([], env)
    assert str(tmp_path / ".cache" / "sdm-test" / "jax") in line
