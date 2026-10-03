"""JSON Schema resources for the ``.sdm`` format.

This subpackage holds the versioned JSON Schema files, the version index
built from them, and the conformance corpus. Importing it pulls in neither
JAX nor jsonschema, so a host that cannot install this package's full
dependency set (Blender's bundled interpreter, for example) can still reach
a schema, or the index, across a subprocess boundary via
``python -m software_defined_matter.schema``. See ``__main__.py``.

``_generate.py`` is the exception and is a development tool, not part of
that boundary: it imports the wire contract to build the current schema and
the index, and is only ever run from a full checkout.

Loading and parsing a schema (``load_schema``) lives in
``software_defined_matter.io``, alongside the rest of file I/O.
"""

from __future__ import annotations

__all__: list[str] = []
