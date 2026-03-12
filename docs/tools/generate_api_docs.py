from __future__ import annotations

"""Generate API documentation with a small pdoc compatibility shim."""

from pathlib import Path
import re
import shutil
import typing

import pdoc
import pdoc.doc_types
import pdoc.render


GUIDE_PAGES = (
    {
        "slug": "index",
        "source": Path("docs/content/overview.md"),
        "output": "index.html",
    },
    {
        "slug": "overview",
        "source": Path("docs/content/overview.md"),
        "output": "overview.html",
    },
    {
        "slug": "implementation-guide",
        "source": Path("docs/content/implementation-guide.md"),
        "output": "implementation-guide.html",
    },
    {
        "slug": "tutorial-minimal-agent",
        "source": Path("docs/content/tutorial-minimal-agent.md"),
        "output": "tutorial-minimal-agent.html",
    },
)

MARKDOWN_SNIPPETS = {
    "oak_architecture_api_intro": Path("docs/content/oak_architecture_api_intro.md"),
}

MARKDOWN_TITLE_PATTERN = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)

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
    """Load authored Markdown fragments used by templates and guide pages."""

    sources = {
        name: path.read_text()
        for name, path in MARKDOWN_SNIPPETS.items()
    }
    return sources


def _extract_markdown_title(markdown_text: str, slug: str) -> str:
    """Use the first H1 as the page title, with a slug fallback."""

    title_match = MARKDOWN_TITLE_PATTERN.search(markdown_text)
    if title_match:
        return title_match.group(1).strip()

    return slug.replace("-", " ").title()


def _guide_pages_with_titles() -> list[dict[str, str]]:
    """Resolve guide-page titles from the authored Markdown files."""

    resolved_pages = []
    for page in GUIDE_PAGES:
        markdown_text = page["source"].read_text()
        resolved_pages.append(
            {
                "title": _extract_markdown_title(markdown_text, page["slug"]),
                "output": page["output"],
                "page_markdown": markdown_text,
            }
        )
    return resolved_pages


def _render_markdown_pages(output_directory: Path) -> None:
    """Render authored guide pages into the generated docs site."""

    template = pdoc.render.env.get_template("markdown_page.html.jinja2")
    guide_pages = [
        {"title": page["title"], "output": page["output"]}
        for page in _guide_pages_with_titles()
        if page["output"] != "index.html"
    ]
    for page in _guide_pages_with_titles():
        rendered = template.render(
            page_title=page["title"],
            page_markdown=page["page_markdown"],
            guide_pages=guide_pages,
        )
        (output_directory / page["output"]).write_text(rendered)


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
    pdoc.render.env.globals["guide_pages"] = [
        {"title": page["title"], "output": page["output"]}
        for page in _guide_pages_with_titles()
        if page["output"] != "index.html"
    ]
    pdoc.pdoc("src/oak_architecture", output_directory=output_directory)
    _render_markdown_pages(output_directory)


if __name__ == "__main__":
    main()
