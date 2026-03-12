from __future__ import annotations

"""Generate API documentation with a small pdoc compatibility shim."""

from pathlib import Path
import shutil
import typing

import pdoc
import pdoc.doc_types
import pdoc.render


def _patch_pdoc_for_python_314() -> None:
    """Replace pdoc's vendored type evaluator with Python 3.14's version."""

    if not hasattr(typing, "_eval_type"):
        return

    def compat_eval_type(
        t: object,
        globalns: dict[str, object] | None,
        localns: dict[str, object] | None,
        recursive_guard: frozenset[str] = frozenset(),
    ) -> object:
        return typing._eval_type(
            t,
            globalns,
            localns,
            type_params=(),
            recursive_guard=recursive_guard,
        )

    pdoc.doc_types._eval_type = compat_eval_type


def main() -> None:
    """Generate the HTML API docs into ``site/api``."""

    output_directory = Path("site/api")
    if output_directory.exists():
        shutil.rmtree(output_directory)
    output_directory.mkdir(parents=True)

    _patch_pdoc_for_python_314()
    pdoc.render.configure(docformat="google")
    pdoc.pdoc("src/oak_architecture", output_directory=output_directory)


if __name__ == "__main__":
    main()
