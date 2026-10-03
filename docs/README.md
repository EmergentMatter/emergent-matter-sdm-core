# docs/

Reference material and design records that support the code but don't
belong in the root [`README.md`](../README.md) or `CLAUDE.md`.

There is no index of the files here, deliberately. The directory listing
is the index, and it cannot fall behind the way a hand-kept list does.
What each location means:

| Location | What lives there | How it ages |
|---|---|---|
| `*.md` at this level | Reference: the reasoning behind the code, where it is too broad to sit in one docstring | Slowly. Correct it when the reasoning changes |
| [`adr/`](adr/) | One architecture decision per file, `NNNN-kebab-title.md`, with `Status` / `Context` / `Decision` / `Consequences` per [STYLE.md](../STYLE.md) | Not at all. A record is superseded by a later one, never rewritten, so it states what was true when it was accepted |
| `sdf_distances/` | The SDF distance audit and the figures generated for it | With its subject |

`csg_diagram.png` is live: it renders the CSG tree shown in the root
[`README.md`](../README.md).

## Adding to this directory

- Recording a **decision** and the reasoning behind it: write an ADR.
- Explaining **why** the code is shaped as it is, where that reasoning is
  too broad for one docstring: extend a reference page.
- Anything that is neither, such as a dated investigation or a superseded
  plan: it belongs in a GitHub issue.

Name a file so the listing reads as its own index. An ADR states its
decision in its filename, not just its number.

**Do not restate the API in prose.** Signatures, keyword arguments,
default values, per-node parameter lists, and the set of schema versions
all belong in docstrings, next to the code, where they are reviewed with
the change that moves them. A prose copy is a second source of truth that
nothing checks: it goes stale silently, and it is usually a lossier copy
than the docstring it duplicates.

The test before adding a paragraph: **would this need editing if someone
renamed a keyword argument?** If yes, it belongs in a docstring instead.

A document that reads as a proposal is a sign the decision has not been
recorded yet. Convert it once the decision is made.
