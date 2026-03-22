from __future__ import annotations

"""Generate API documentation with a small pdoc compatibility shim."""

from pathlib import Path
import shutil
import sys
import typing

import pdoc
import pdoc.doc_types
import pdoc.render

MARKDOWN_SNIPPETS = {
    "oak_architecture_api_intro": Path("docs/content/oak_architecture_api_intro.md"),
}
CORE_MODULES = ("oak_architecture",)
DOCS_HOME_MODULE = "oak_architecture"


def _configure_import_paths() -> None:
    """Ensure repository packages are importable during doc generation."""

    repository_root = Path(__file__).resolve().parents[2]
    for path in (repository_root, repository_root / "src"):
        path_string = str(path)
        if path_string not in sys.path:
            sys.path.insert(0, path_string)


def _discover_example_modules() -> tuple[str, ...]:
    """Discover importable example modules for pdoc output."""

    module_names: set[str] = set()
    for path in Path("examples").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        if path == Path("examples/__init__.py"):
            continue

        parts = list(path.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        module_names.add(".".join(parts))

    return tuple(sorted(module_names))


def _build_sidebar_sections(example_modules: tuple[str, ...]) -> list[dict[str, object]]:
    """Build the persistent sidebar navigation for the docs site."""

    return [
        {
            "title": "Core API",
            "items": [
                {"module": "oak_architecture", "label": "oak_architecture"},
                {
                    "module": "oak_architecture.fine_grained",
                    "label": "oak_architecture.fine_grained",
                },
            ],
        },
        {
            "title": "Examples",
            "items": [
                {
                    "module": module_name,
                    "label": module_name.removeprefix("examples."),
                }
                for module_name in example_modules
            ],
        },
    ]


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

    sources = {name: path.read_text() for name, path in MARKDOWN_SNIPPETS.items()}
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
    """Generate the HTML API docs into `docs/api`."""

    output_directory = Path("docs/api")
    _prepare_output_directory(output_directory)

    _configure_import_paths()
    _patch_pdoc_for_python_314()
    pdoc.render.configure(
        docformat="google",
        template_directory=Path("docs/tools/pdoc_templates"),
    )
    example_modules = _discover_example_modules()
    pdoc.render.env.globals["docs_markdown"] = _load_markdown_sources()
    pdoc.render.env.globals["docs_home_module"] = DOCS_HOME_MODULE
    pdoc.render.env.globals["sidebar_sections"] = _build_sidebar_sections(example_modules)
    documented_modules = (*CORE_MODULES, *example_modules)
    pdoc.pdoc(*documented_modules, output_directory=output_directory)


if __name__ == "__main__":
    main()
