"""The built docs site: every citation on its pages is rendered."""

import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def built_pages(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    """Build the site once, and return each HTML page's text by its path in the site."""
    site = tmp_path_factory.mktemp("site")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mkdocs",
            "build",
            "--site-dir",
            str(site),
        ],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
    )
    return {
        str(page.relative_to(site)): page.read_text(encoding="utf-8")
        for page in site.rglob("*.html")
    }


# "[@" is a citation the plugin did not read, such as one split across two lines; "{.nocase}" is
# a Pandoc span left in a formatted citation, such as the "et al." of an author list ending in
# "and others".
@pytest.mark.parametrize(
    "markup",
    [
        "[@",
        "{.nocase}",
    ],
)
def test_no_built_page_shows_citation_markup(
    built_pages: dict[str, str],
    markup: str,
) -> None:
    showing = sorted(path for path, text in built_pages.items() if markup in text)

    assert showing == []
