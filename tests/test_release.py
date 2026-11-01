"""scripts/check-release.sh: a release's version against CHANGELOG.md and CITATION.cff."""

import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check-release.sh"
INIT = '__version__ = "0.8.0"\n'
# A Keep a Changelog file whose latest release, 0.8.0, is dated and linked.
CHANGELOG = """\
# Changelog

## [Unreleased]

## [0.8.0] - 2026-11-01

### Fixed

- A rating fails closed on an image a redaction removed.

## [0.7.0] - 2026-10-30

### Added

- `ToolCall.media`.

[Unreleased]: https://github.com/omamori-lab/hyoshigi/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/omamori-lab/hyoshigi/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/omamori-lab/hyoshigi/releases/tag/v0.7.0
"""
# cff-version is not the release's version, so a reader that took it would fail.
CITATION = """\
cff-version: 1.2.0
title: hyoshigi
version: "0.8.0"
date-released: "2026-11-01"
"""


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A tree whose files agree with version 0.8.0, holding a copy of the release check."""
    package = tmp_path / "src" / "hyoshigi"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(INIT)
    (tmp_path / "CHANGELOG.md").write_text(CHANGELOG)
    (tmp_path / "CITATION.cff").write_text(CITATION)
    (tmp_path / "scripts").mkdir()
    shutil.copy(
        SCRIPT,
        tmp_path / "scripts",
    )
    return tmp_path


def check_release(
    tree: Path,
    arguments: tuple[str, ...],
) -> subprocess.CompletedProcess[str]:
    """Run the tree's copy of the release check."""
    return subprocess.run(
        [
            tree / "scripts" / "check-release.sh",
            *arguments,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.xfail(strict=True)
@pytest.mark.parametrize(
    "arguments",
    [
        ("0.8.0",),
        (),
    ],
    ids=[
        "the tag's version",
        "no version",
    ],
)
def test_a_tree_that_agrees_with_the_version_passes(
    tree: Path,
    arguments: tuple[str, ...],
) -> None:
    process = check_release(
        tree,
        arguments,
    )

    assert (
        process.returncode,
        process.stdout,
        process.stderr,
    ) == (0, "", "")


@pytest.mark.xfail(strict=True)
def test_a_tag_other_than_the_package_version_is_refused(tree: Path) -> None:
    process = check_release(
        tree,
        ("0.9.0",),
    )

    assert process.returncode == 1
    assert "__version__ is '0.8.0', not 0.9.0." in process.stderr


@pytest.mark.xfail(strict=True)
@pytest.mark.parametrize(
    ("path", "old", "new", "arguments", "problem"),
    [
        (
            "CHANGELOG.md",
            "## [0.8.0] - 2026-11-01",
            "## [0.8.0]",
            ("0.8.0",),
            "CHANGELOG.md has no '## [0.8.0] - YYYY-MM-DD' section.",
        ),
        (
            "CHANGELOG.md",
            "## [0.8.0] - 2026-11-01",
            "## [0.8.0] - 2026-11-1",
            ("0.8.0",),
            "CHANGELOG.md has no '## [0.8.0] - YYYY-MM-DD' section.",
        ),
        (
            "CHANGELOG.md",
            "### Fixed\n\n- A rating fails closed on an image a redaction removed.\n\n",
            "",
            ("0.8.0",),
            "the CHANGELOG.md section for 0.8.0 is empty.",
        ),
        (
            "CHANGELOG.md",
            "[0.8.0]: https://github.com/omamori-lab/hyoshigi/compare/v0.7.0...v0.8.0\n",
            "",
            ("0.8.0",),
            "CHANGELOG.md has no '[0.8.0]: https://...' link reference.",
        ),
        (
            "CITATION.cff",
            'version: "0.8.0"',
            'version: "0.7.0"',
            ("0.8.0",),
            "CITATION.cff has version '0.7.0', not 0.8.0.",
        ),
        (
            "CITATION.cff",
            'date-released: "2026-11-01"',
            'date-released: "2026-10-30"',
            ("0.8.0",),
            "CITATION.cff has date-released '2026-10-30', not 2026-11-01 as in CHANGELOG.md.",
        ),
        (
            "src/hyoshigi/__init__.py",
            '"0.8.0"',
            '"0.9.0"',
            (),
            "CITATION.cff has version '0.8.0', not 0.9.0.",
        ),
    ],
    ids=[
        "an undated section",
        "a date not written YYYY-MM-DD",
        "an empty section",
        "no link reference",
        "another citation version",
        "a citation date that is not the section's",
        "a package version the files do not name",
    ],
)
def test_a_tree_that_disagrees_with_the_version_is_refused_naming_the_problem(
    tree: Path,
    path: str,
    old: str,
    new: str,
    arguments: tuple[str, ...],
    problem: str,
) -> None:
    edited = tree / path
    edited.write_text(
        edited.read_text().replace(
            old,
            new,
        ),
    )

    process = check_release(
        tree,
        arguments,
    )

    assert process.returncode == 1
    assert problem in process.stderr
