"""JAX's persistent compilation cache, for tools that compile the same functions on every run.

A CEM's build CLI compiles its jitted geometry functions each time it runs, and a viewer replays
that build on every slider release; with the cache, a repeat run loads the compiled executables
from disk instead. Call `enable_compile_cache` from a command-line entry point, never at import:
a library does not change JAX's global settings behind its caller's back.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

__all__ = ["DISABLE", "ENV_VAR", "MAX_BYTES", "enable_compile_cache"]

#: Environment variable naming the cache directory; ``DISABLE`` turns the cache off.
ENV_VAR = "SDM_JAX_CACHE"
DISABLE = "0"
#: The cache's default size limit in bytes (1 GiB): past it, JAX deletes the least recently used
#: entries.
MAX_BYTES = 1 << 30


def enable_compile_cache(
    name: str, directory: str | os.PathLike[str] | None = None, max_bytes: int = MAX_BYTES
) -> Path | None:
    """Point JAX's persistent compilation cache at a directory and cache every compile.

    ``name`` keeps tools apart (a CEM's package name, say): the default directory is
    ``~/.cache/<name>/jax``. ``directory`` overrides that, and the ``SDM_JAX_CACHE`` environment
    variable overrides both; ``SDM_JAX_CACHE=0`` leaves the cache off. Every compile is cached,
    however quick or small, since a short build is exactly the case that repeats. Because every
    compile is kept, the directory is held to ``max_bytes``: past it, JAX deletes the least
    recently used entries (``-1`` lifts the limit).

    A test suite that runs an entry point calling this should set ``SDM_JAX_CACHE=0``, so a test
    run does not write into the user's cache.

    Returns the directory in use, or ``None`` when the cache is off.
    """
    import jax

    env = os.environ.get(ENV_VAR)
    if env == DISABLE:
        return None
    where = Path(env or directory or Path.home() / ".cache" / name / "jax").expanduser()
    jax.config.update("jax_compilation_cache_dir", str(where))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    # JAX holds the cache to its limit through `filelock`. A limit set without it stops every
    # write, so an install missing it keeps a cache with no limit rather than none at all.
    if importlib.util.find_spec("filelock") is not None:
        jax.config.update("jax_compilation_cache_max_size", max_bytes)
    return where
