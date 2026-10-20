"""The docs pages' citations and docs/references.bib name the same sources."""

import re
from pathlib import Path

import bibtexparser

DOCS = Path(__file__).resolve().parents[2] / "docs"
# A citation group is a bracket that opens with a key, as [@key, sec. 3.2] or [@first; @second].
CITATION_GROUP = re.compile(r"\[(@[^\]\n]+)\]")
CITATION_KEY = re.compile(r"@([A-Za-z0-9_:-]+)")


def read_entry_keys() -> set[str]:
    """Return the key of every entry in docs/references.bib, which must parse whole."""
    library = bibtexparser.parse_file(
        str(DOCS / "references.bib"),
        encoding="utf-8",
    )
    assert library.failed_blocks == []
    return {entry.key for entry in library.entries}


def read_cited_keys() -> set[str]:
    """Return every key cited on a docs page, skipping the private notes in docs/plans/."""
    pages = [page for page in DOCS.rglob("*.md") if (DOCS / "plans") not in page.parents]
    groups = [
        group
        for page in pages
        for group in CITATION_GROUP.findall(page.read_text(encoding="utf-8"))
    ]
    return {key for group in groups for key in CITATION_KEY.findall(group)}


def test_every_cited_key_names_a_bibliography_entry() -> None:
    entry_keys = read_entry_keys()

    cited_keys = read_cited_keys()

    assert cited_keys - entry_keys == set()


def test_every_bibliography_entry_is_cited() -> None:
    cited_keys = read_cited_keys()

    entry_keys = read_entry_keys()

    assert entry_keys - cited_keys == set()
