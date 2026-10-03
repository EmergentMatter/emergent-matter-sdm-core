# Third-Party Software

Every software dependency of `emergent-matter-sdm-core` that is **not** EmergentMatter first-party code: the packages resolved into this repository's environment, the build backend, and the third-party GitHub Actions its CI runs.

Versions are deliberately left out. They move on every lock refresh, and a list that churns on every bump stops being read. For the exact pinned version of anything below, see `uv.lock`.

## Direct dependencies

Chosen by this repository and declared in `pyproject.toml`.

### Runtime

`[project] dependencies`, installed for anyone who installs this package.

| Package | License | Purpose |
|---|---|---|
| [`jax`](https://github.com/jax-ml/jax) | Apache-2.0 | Differentiate, compile, and transform Numpy code |
| [`jaxlib`](https://github.com/jax-ml/jax) | Apache-2.0 | XLA library for JAX |
| [`jsonschema`](https://github.com/python-jsonschema/jsonschema) | MIT | An implementation of JSON Schema validation for Python |
| [`numpy`](https://numpy.org) | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | Fundamental package for array computing in Python |
| [`pillow`](https://python-pillow.github.io) | MIT-CMU | Python Imaging Library (fork) |
| [`pyvista`](https://github.com/pyvista/pyvista) | MIT | Easier Pythonic interface to VTK |

### Optional extras

`[project.optional-dependencies]`, installed only when the extra is requested.

| Package | License | Purpose |
|---|---|---|
| [`fast-simplification`](https://github.com/pyvista/fast-simplification) | MIT | Wrapper around the Fast-Quadric-Mesh-Simplification library |
| [`rtree`](https://github.com/Toblerity/rtree) | MIT | R-Tree spatial index for Python GIS |
| [`scikit-image`](https://scikit-image.org) | BSD | Image processing in Python |
| [`trimesh`](https://github.com/mikedh/trimesh) | MIT | Import, export, process, analyze and view triangular meshes |

### Development

`[dependency-groups]`, used for tests, linting and type checking. Not shipped to consumers.

| Package | License | Purpose |
|---|---|---|
| [`mypy`](https://www.mypy-lang.org/) | MIT | Optional static typing for Python |
| [`pytest`](https://docs.pytest.org/en/latest/) | MIT | pytest: simple powerful testing with Python |
| [`pytest-xdist`](https://github.com/pytest-dev/pytest-xdist) | MIT | pytest xdist plugin for distributed testing, most importantly across multiple CPUs |
| [`ruff`](https://docs.astral.sh/ruff) | MIT | An extremely fast Python linter and code formatter, written in Rust |

## Transitive dependencies

Not requested by this repository. They arrive as dependencies of the packages above and are resolved into `uv.lock`.

| Package | License | Purpose |
|---|---|---|
| [`ast-serialize`](https://github.com/mypyc/ast_serialize) | MIT | Python bindings for mypy AST serialization |
| [`attrs`](https://pypi.org/project/attrs/) | MIT | Classes Without Boilerplate |
| [`certifi`](https://github.com/certifi/python-certifi) | MPL-2.0 | Python package for providing Mozilla's CA Bundle |
| [`charset-normalizer`](https://pypi.org/project/charset-normalizer/) | MIT | The Real First Universal Charset Detector. Open, modern and actively maintained alternativ |
| [`colorama`](https://github.com/tartley/colorama) | BSD | Cross-platform colored terminal text |
| [`contourpy`](https://github.com/contourpy/contourpy) | BSD | Python library for calculating contours of 2D quadrilateral grids |
| [`cycler`](https://matplotlib.org/cycler/) | BSD | Composable style cycles |
| [`cyclopts`](https://github.com/BrianPugh/cyclopts) | Apache-2.0 | Intuitive, easy CLIs based on type hints |
| [`docstring-parser`](https://pypi.org/project/docstring-parser/) | MIT | Parse Python docstrings in reST, Google and Numpydoc format |
| [`docutils`](https://docutils.sourceforge.io) | Public Domain; BSD License; GNU General Public License (GPL) | Docutils -- Python Documentation Utilities |
| [`execnet`](https://execnet.readthedocs.io/en/latest/) | MIT | execnet: rapid multi-Python deployment |
| [`fonttools`](http://github.com/fonttools/fonttools) | MIT | Tools to manipulate font files |
| [`idna`](https://github.com/kjd/idna) | BSD-3-Clause | Internationalized Domain Names in Applications (IDNA) |
| [`imageio`](https://github.com/imageio/imageio) | BSD-2-Clause | Read and write images and video across all major formats. Supports scientific and volumetr |
| [`iniconfig`](https://github.com/pytest-dev/iniconfig) | MIT | brain-dead simple config-ini parsing |
| [`jsonschema-specifications`](https://github.com/python-jsonschema/jsonschema-specifications) | MIT | The JSON Schema meta-schemas and vocabularies, exposed as a Registry |
| [`kiwisolver`](https://github.com/nucleic/kiwi) | BSD | A fast implementation of the Cassowary constraint solver |
| [`lazy-loader`](https://github.com/scientific-python/lazy-loader) | BSD-3-Clause | Makes it easy to load subpackages and functions on demand |
| [`librt`](https://github.com/mypyc/librt) | MIT | Mypyc runtime library |
| [`markdown-it-py`](https://github.com/executablebooks/markdown-it-py) | MIT | Python port of markdown-it. Markdown parsing, done right! |
| [`matplotlib`](https://matplotlib.org) | PSF | Python plotting package |
| [`mdurl`](https://github.com/executablebooks/mdurl) | MIT | Markdown URL utilities |
| [`ml-dtypes`](https://pypi.org/project/ml-dtypes/) | Apache-2.0 | ml_dtypes is a stand-alone implementation of several NumPy dtype extensions used in machin |
| [`mypy-extensions`](https://github.com/python/mypy_extensions) | MIT | Experimental type system extensions for mypy |
| [`networkx`](https://networkx.org/) | BSD-3-Clause | Python package for creating and manipulating graphs and networks |
| [`opt-einsum`](https://pypi.org/project/opt-einsum/) | MIT | Path optimization of einsum functions |
| [`packaging`](https://github.com/pypa/packaging) | Apache-2.0 OR BSD-2-Clause | Core utilities for Python packages |
| [`pathspec`](https://github.com/cpburnz/python-pathspec) | MPL-2.0 | Utility library for gitignore style pattern matching of file paths |
| [`platformdirs`](https://github.com/tox-dev/platformdirs) | MIT | A small Python package for determining appropriate platform-specific dirs, e.g. a `user da |
| [`pluggy`](https://pypi.org/project/pluggy/) | MIT | plugin and hook calling mechanisms for python |
| [`pooch`](https://github.com/fatiando/pooch) | BSD-3-Clause | A friend to fetch your data files |
| [`pygments`](https://pygments.org) | BSD-2-Clause | Pygments is a syntax highlighting package written in Python |
| [`pyparsing`](https://github.com/pyparsing/pyparsing/) | MIT | pyparsing - Classes and methods to define and execute parsing grammars |
| [`python-dateutil`](https://github.com/dateutil/dateutil) | BSD License; Apache Software License | Extensions to the standard Python datetime module |
| [`referencing`](https://github.com/python-jsonschema/referencing) | MIT | JSON Referencing + Python |
| [`requests`](https://github.com/psf/requests) | Apache-2.0 | Python HTTP for Humans |
| [`rich`](https://github.com/Textualize/rich) | MIT | Render rich text, tables, progress bars, syntax highlighting, markdown and more to the ter |
| [`rich-rst`](https://github.com/wasi-master/rich-rst) | MIT | A beautiful reStructuredText renderer for rich |
| [`rpds-py`](https://github.com/crate-py/rpds) | MIT | Python bindings to Rust's persistent data structures (rpds) |
| [`scipy`](https://scipy.org/) | BSD | Fundamental algorithms for scientific computing in Python |
| [`scooby`](https://github.com/banesullivan/scooby) | MIT | A Great Dane turned Python environment detective |
| [`six`](https://github.com/benjaminp/six) | MIT | Python 2 and 3 compatibility utilities |
| [`tifffile`](https://www.cgohlke.com) | BSD-3-Clause | Read and write TIFF files |
| [`typing-extensions`](https://github.com/python/typing_extensions) | PSF-2.0 | Backported and Experimental Type Hints for Python 3.9+ |
| [`urllib3`](https://pypi.org/project/urllib3/) | MIT | HTTP library with thread-safe connection pooling, file post, and more |
| [`vtk`](https://vtk.org) | BSD | VTK is an open-source toolkit for 3D computer graphics, image processing, and visualizatio |

## Build and CI toolchain

Third-party software the repository is built and tested with, rather than packages it imports.

| Component | Role | License |
|---|---|---|
| [Python](https://www.python.org/) | Language runtime | PSF-2.0 |
| [`uv`](https://github.com/astral-sh/uv) | Dependency resolver and installer | MIT OR Apache-2.0 |
| [`hatchling`](https://github.com/pypa/hatch) | Build backend, `[build-system] requires` | MIT |
| [`actions/checkout`](https://github.com/actions/checkout) | CI: checks out the repository | MIT |
| [`actions/upload-artifact`](https://github.com/actions/upload-artifact) | CI: uploads build artifacts | MIT |
| [`astral-sh/setup-uv`](https://github.com/astral-sh/setup-uv) | CI: installs `uv` | MIT |

## First-party, deliberately not listed

These are EmergentMatter's own code and are out of scope for this file:

- Any `emergent-matter-*` package, and `sdm-view`
- `EmergentMatter/actions`, the shared release and changelog workflows

## Notes

- `mypy-extensions` publishes no license metadata to PyPI. MIT is taken from the `LICENSE` file in its upstream repository.
- `certifi` and `pathspec` are MPL-2.0, which is file-level copyleft. Unmodified transitive dependencies, so the obligation is to keep the notices intact, which distributing them unmodified does.
- `docutils` is multi-licensed (Public Domain; BSD License; GNU General Public License (GPL)). The bulk of it is public domain, with some BSD and a few GPL components. It reaches this repository transitively, via `emergent-matter-sdm-core` -> `pyvista` -> `cyclopts` -> `rich-rst` -> `docutils`. It is used unmodified and is not redistributed as part of this repository's source, but it is the one package here with any GPL component, so it is worth a look before shipping a bundled artifact.
- The `setup-node` and `setup-bun` steps in `.github/workflows/` sit inside commented-out template blocks, and this repository has no `package.json`, so it has no JavaScript dependencies.

## Regenerating

Direct entries come from `pyproject.toml`; transitive entries are every remaining package in `uv.lock` that is not first-party. Licenses come from installed package metadata, falling back to PyPI.
