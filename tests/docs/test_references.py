"""docs/references.bib: complete entries, each cited on a docs page, and no citation without one."""

import re
from pathlib import Path

import bibtexparser
from bibtexparser.model import Entry

DOCS = Path(__file__).resolve().parents[2] / "docs"
# A citation group is a bracket that opens with a key, as [@key, sec. 3.2] or [@first; @second].
CITATION_GROUP = re.compile(r"\[(@[^\]\n]+)\]")
CITATION_KEY = re.compile(r"@([A-Za-z0-9_:-]+)")


def read_entries() -> list[Entry]:
    """Return every entry in docs/references.bib, which must parse whole."""
    library = bibtexparser.parse_file(
        str(DOCS / "references.bib"),
        encoding="utf-8",
    )
    assert library.failed_blocks == []
    return library.entries


def read_entry_keys() -> set[str]:
    """Return the key of every entry in docs/references.bib."""
    return {entry.key for entry in read_entries()}


def read_author_lists() -> dict[str, str]:
    """Return the author field of each entry that has one, by key."""
    return {
        entry.key: str(entry["author"]) for entry in read_entries() if "author" in entry.fields_dict
    }


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


def test_no_author_list_ends_in_others() -> None:
    author_lists = read_author_lists()

    cut_short = {key for key, authors in author_lists.items() if authors.split()[-1] == "others"}

    assert cut_short == set()
