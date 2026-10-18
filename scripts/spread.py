"""Spread brackets of two or more items one item per line, each item with a trailing comma.

A bracket takes a trailing comma, so that `ruff format` puts one item on each line, when it
holds a comma between items. It stays as it is when it is a subscript (an index or a type such
as `dict[str, int]`), a comprehension, a replacement field of an f-string, part of an import,
or holds only plain values (numbers, strings, None, True and False). The commas between a
lambda's parameters do not separate items.

Rewrite mode adds the commas, formats with ruff, applies ruff's COM812 fix and repeats until
the code stops changing. Check mode prints `path:row:column` for each bracket that still lacks
its comma and exits 1. The python blocks of a Markdown file are read as code.
"""

import argparse
import enum
import io
import keyword
import pathlib
import re
import subprocess
import sys
import tokenize
from collections.abc import Sequence
from dataclasses import dataclass
from typing import assert_never

# Formatting can bring a bracket into being, such as the parentheses around a long tuple, and
# that bracket then lacks its comma; a few passes reach a fixed point.
MAXIMUM_PASSES = 4
# A fence is a line of its own: "```python" opens a block, and "```" alone closes it.
PYTHON_BLOCK = re.compile(
    r"^```python[ \t]*\n(.*?\n)```[ \t]*$",
    re.DOTALL | re.MULTILINE,
)
IMPORT_KEYWORDS = frozenset({"import", "from"})
SOFT_KEYWORDS = frozenset({"match", "case"})
OPENING_BRACKETS = frozenset({"(", "[", "{"})
CLOSING_BRACKETS = frozenset({")", "]", "}"})
FSTRING_DEPTH_STEPS = {
    tokenize.FSTRING_START: 1,
    tokenize.FSTRING_END: -1,
}
LINE_KINDS = frozenset(
    {
        tokenize.NL,
        tokenize.COMMENT,
        tokenize.INDENT,
        tokenize.DEDENT,
    },
)
PLAIN_KINDS = frozenset(
    {
        tokenize.NUMBER,
        tokenize.STRING,
        tokenize.FSTRING_START,
        tokenize.FSTRING_MIDDLE,
        tokenize.FSTRING_END,
    },
)
PLAIN_NAMES = frozenset({"None", "True", "False"})
FORMAT_ARGUMENTS = ("format",)
# COM812 adds the comma to a bracket that formatting split over lines around a single item.
FIX_COMMAS_ARGUMENTS = ("check", "--select", "COM812", "--fix", "--exit-zero", "--quiet")
FILE_START = tokenize.TokenInfo(
    tokenize.ENCODING,
    "",
    (1, 0),
    (1, 0),
    "",
)


class Mode(enum.Enum):
    """What the script does with each file."""

    REWRITE = "rewrite"
    CHECK = "check"


@dataclass(frozen=True)
class Options:
    """The mode and the files named on the command line."""

    mode: Mode
    paths: tuple[pathlib.Path, ...]


@dataclass(
    frozen=True,
    order=True,
)
class Position:
    """Where a trailing comma is missing: a row from 1 and a column from 0, after the last item."""

    row: int
    column: int


@dataclass
class Bracket:
    """A bracket being read: its top-level commas and whether it stays as it is."""

    skipped: bool
    commas: int = 0
    lambda_parameters: int = 0
    plain: bool = True

    @property
    def holds_items(self) -> bool:
        """Whether the bracket holds two or more items and is not left as it is."""
        return self.commas > 0 and not self.skipped and not self.plain

    def read(
        self,
        token: tokenize.TokenInfo,
    ) -> None:
        """Count one token of the bracket's own level."""
        if token.type == tokenize.OP:
            self.read_operator(token.string)
        elif token.type == tokenize.NAME:
            self.read_name(token.string)
        elif token.type not in PLAIN_KINDS:
            self.plain = False

    def read_operator(
        self,
        text: str,
    ) -> None:
        """Count a comma, end a lambda's parameters at its colon, or note a non-plain operator."""
        if text == ",":
            self.read_comma()
        elif text == ":" and self.lambda_parameters:
            self.lambda_parameters -= 1
        elif text != "-":
            self.plain = False

    def read_comma(self) -> None:
        """Count a comma that separates items, not one that separates a lambda's parameters."""
        if not self.lambda_parameters:
            self.commas += 1

    def read_name(
        self,
        text: str,
    ) -> None:
        """Start a lambda's parameters, skip a comprehension, or note a non-plain name."""
        if text == "lambda":
            self.lambda_parameters += 1
        elif text == "for":
            self.skipped = True
        if text not in PLAIN_NAMES:
            self.plain = False


class Scanner:
    """Reads the tokens of one piece of code and collects the brackets that lack a comma."""

    def __init__(self) -> None:
        self.missing: list[Position] = []
        self.brackets: list[Bracket] = []
        self.last = FILE_START
        self.last_began_statement = False
        self.beginning_statement = True
        self.inside_import = False
        self.fstring_depth = 0

    def read(
        self,
        token: tokenize.TokenInfo,
    ) -> None:
        """Take in the next token of the code."""
        if token.type == tokenize.NEWLINE:
            self.beginning_statement = True
            self.inside_import = False
        elif token.type not in LINE_KINDS:
            self.take(token)
            self.remember(token)

    def take(
        self,
        token: tokenize.TokenInfo,
    ) -> None:
        """Open or close a bracket, or count the token in the bracket it sits in."""
        if token.type == tokenize.OP and token.string in OPENING_BRACKETS:
            self.open_bracket(token.string)
        elif token.type == tokenize.OP and token.string in CLOSING_BRACKETS:
            self.close_bracket()
        else:
            self.fstring_depth += FSTRING_DEPTH_STEPS.get(
                token.type,
                0,
            )
            self.count(token)

    def remember(
        self,
        token: tokenize.TokenInfo,
    ) -> None:
        """Note the token as the last one read, and whether it began an import."""
        if self.beginning_statement:
            self.inside_import = token.type == tokenize.NAME and token.string in IMPORT_KEYWORDS
        self.last_began_statement = self.beginning_statement
        self.beginning_statement = False
        self.last = token

    def count(
        self,
        token: tokenize.TokenInfo,
    ) -> None:
        """Count the token in the innermost open bracket, if there is one."""
        if self.brackets:
            self.brackets[-1].read(token)

    def open_bracket(
        self,
        opening: str,
    ) -> None:
        """Start reading a bracket, left as it is inside an f-string, an import or a subscript."""
        subscript = opening == "[" and self.follows_value()
        skipped = self.inside_import or self.fstring_depth > 0 or subscript
        self.brackets.append(Bracket(skipped=skipped))

    def close_bracket(self) -> None:
        """Finish a bracket and note where it lacks its trailing comma."""
        bracket = self.brackets.pop()
        if bracket.holds_items and self.last.string != ",":
            row, column = self.last.end
            self.missing.append(
                Position(
                    row=row,
                    column=column,
                ),
            )
        if self.brackets:
            self.brackets[-1].plain = False

    def follows_value(self) -> bool:
        """Whether the last token ends a value, so that a `[` after it is a subscript.

        A `[` that begins a statement follows no value, whatever ended the statement before.
        """
        if self.beginning_statement:
            return False
        last = self.last
        if last.type == tokenize.OP:
            return last.string in CLOSING_BRACKETS
        if last.type != tokenize.NAME or keyword.iskeyword(last.string):
            return False
        return not (last.string in SOFT_KEYWORDS and self.last_began_statement)


def find_missing(code: str) -> list[Position]:
    """Return where the code lacks a trailing comma after the last of two or more items."""
    scanner = Scanner()
    for token in tokenize.generate_tokens(io.StringIO(code).readline):
        scanner.read(token)
    return scanner.missing


def find_missing_in_markdown(text: str) -> list[Position]:
    """Return where the python blocks of a Markdown text lack a comma, as lines of the text."""
    missing: list[Position] = []
    for block in PYTHON_BLOCK.finditer(text):
        offset = text.count(
            "\n",
            0,
            block.start(1),
        )
        for position in find_missing(block.group(1)):
            missing.append(
                Position(
                    row=position.row + offset,
                    column=position.column,
                ),
            )
    return missing


def add_commas(code: str) -> str:
    """Return the code with a comma after the last item of each bracket that lacks one."""
    lines = code.split("\n")
    for position in sorted(
        find_missing(code),
        reverse=True,
    ):
        line = lines[position.row - 1]
        lines[position.row - 1] = f"{line[: position.column]},{line[position.column :]}"
    return "\n".join(lines)


def run_ruff(
    code: str,
    *,
    arguments: Sequence[str],
    filename: str,
) -> str:
    """Return what ruff prints for the code, which it reads as the file of that name."""
    completed = subprocess.run(
        [
            "uv",
            "run",
            "--group",
            "dev",
            "--quiet",
            "ruff",
            *arguments,
            "--stdin-filename",
            filename,
            "-",
        ],
        input=code,
        capture_output=True,
        encoding="utf-8",
        check=True,
    )
    return completed.stdout


def spread(
    code: str,
    *,
    filename: str,
) -> str:
    """Return the code with every bracket of two or more items spread one item per line."""
    for _ in range(MAXIMUM_PASSES):
        formatted = run_ruff(
            add_commas(code),
            arguments=FORMAT_ARGUMENTS,
            filename=filename,
        )
        spread_code = run_ruff(
            formatted,
            arguments=FIX_COMMAS_ARGUMENTS,
            filename=filename,
        )
        if spread_code == code:
            break
        code = spread_code
    return code


def spread_markdown(
    text: str,
    *,
    filename: str,
) -> str:
    """Return the Markdown text with each of its python blocks spread."""
    return PYTHON_BLOCK.sub(
        lambda block: f"```python\n{spread(block.group(1), filename=filename)}```",
        text,
    )


def rewrite(paths: Sequence[pathlib.Path]) -> int:
    """Spread each file in place, and return the exit status."""
    for path in paths:
        text = path.read_text()
        if path.suffix == ".md":
            spread_text = spread_markdown(
                text,
                filename=str(path.with_suffix(".py")),
            )
        else:
            spread_text = spread(
                text,
                filename=str(path),
            )
        if spread_text != text:
            path.write_text(spread_text)
    return 0


def check(paths: Sequence[pathlib.Path]) -> int:
    """Print each bracket that lacks its comma, and return 1 if there is any, otherwise 0."""
    found = False
    for path in paths:
        text = path.read_text()
        missing = find_missing_in_markdown(text) if path.suffix == ".md" else find_missing(text)
        for position in missing:
            print(
                f"{path}:{position.row}:{position.column + 1}: "
                "two or more items without a trailing comma",
            )
            found = True
    return 1 if found else 0


def read_options(arguments: Sequence[str] | None) -> Options:
    """Read the command line."""
    parser = argparse.ArgumentParser(
        description="Spread brackets of two or more items one item per line.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="print path:row:column of each bracket lacking its comma, and exit 1 if any",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=pathlib.Path,
        help="Python files, and Markdown files whose python blocks are read",
    )
    namespace = parser.parse_args(arguments)
    return Options(
        mode=Mode.CHECK if namespace.check else Mode.REWRITE,
        paths=tuple(namespace.paths),
    )


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the script in the mode the command line names, and return the exit status."""
    options = read_options(arguments)
    match options.mode:
        case Mode.CHECK:
            return check(options.paths)
        case Mode.REWRITE:
            return rewrite(options.paths)
        case _ as unreachable:
            assert_never(unreachable)


if __name__ == "__main__":
    sys.exit(main())
