from __future__ import annotations

"""Generate API documentation with a small pdoc compatibility shim."""

from pathlib import Path
import shutil
import typing

import pdoc
import pdoc.doc_types
import pdoc.render

MARKDOWN_SNIPPETS = {
    "oak_architecture_api_intro": Path("docs/content/oak_architecture_api_intro.md"),
}

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


def _load_markdown_sources() -> dict[str, str]:
    """Load authored Markdown fragments used by pdoc templates."""

    sources = {
        name: path.read_text()
        for name, path in MARKDOWN_SNIPPETS.items()
    }
    return sources


def _prepare_output_directory(output_directory: Path) -> None:
    """Reset generated docs while preserving pre-rendered diagram assets."""

    output_directory.mkdir(parents=True, exist_ok=True)
    for child in output_directory.iterdir():
        if child.name == "img":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    (output_directory / ".nojekyll").write_text("")

def main() -> None:
    """Generate the HTML API docs into ``docs/api``."""

    output_directory = Path("docs/api")
    _prepare_output_directory(output_directory)

    _patch_pdoc_for_python_314()
    pdoc.render.configure(
        docformat="google",
        template_directory=Path("docs/tools/pdoc_templates"),
    )
    pdoc.render.env.globals["docs_markdown"] = _load_markdown_sources()
    pdoc.pdoc("src/oak_architecture", output_directory=output_directory)


if __name__ == "__main__":
    main()
