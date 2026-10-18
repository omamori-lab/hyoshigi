# Contributing to hyoshigi

Thank you for helping. This guide covers how the project is set up, the gates
a change must pass and how to send a pull request.

## Where to start

- The library is in design: nothing is released yet. For anything larger
  than a fix, open an issue first, so the design is agreed before the code.
- `AGENTS.md` holds the rules for code and tests, for people and coding
  agents alike: design before code, primary sources only, canonical terms,
  failures as typed outcomes, tests first with an oracle independent of the
  code.
- To report a vulnerability, follow [SECURITY.md](SECURITY.md).

## Setup

```console
git clone https://github.com/omamori-lab/hyoshigi
cd hyoshigi
uv sync --group dev
uvx pre-commit install
```

The pre-commit hooks and `scripts/check.sh` include a secrets scan, which
needs [gitleaks](https://github.com/gitleaks/gitleaks) on your `PATH`, for
example from `brew install gitleaks`.

## The gates

One script runs every gate:

```console
scripts/check.sh
```

It scans for secrets with gitleaks and `.gitleaks.toml`, then runs ruff, the
spread check, ty (type checking), the test suite and the lanorme standards
(naming, complexity, docstrings, prose), builds the package and imports the
built wheel in a fresh environment. CI runs the same script on Python 3.12,
3.13 and 3.14, and scans the whole history for secrets.

The domain is mutation-tested: run `uv run --group dev mutmut run`, then
`uv run --group dev mutmut results`, and explain every surviving mutant in
the pull request.

The pre-commit hooks run the fast gates only: ruff, ty, lanorme, the test
suite, a gitleaks scan of the staged changes and a few file checks. Run
`scripts/check.sh` before you push.

lanorme needs Python 3.13, so the script runs it with `uvx --python 3.13`; it
only parses the code, so it checks 3.12 source correctly. To see why a rule
fired, run `uvx --python 3.13 lanorme==0.21.1 rule CODE`. Its limits in
`pyproject.toml` are a ratchet: meet them by naming steps or sharing a
helper, not by raising a limit.

## Sending a pull request

1. Branch off an up-to-date `main`: `feat/...` for a feature, `fix/...` for a
   bug, `docs/...` for documentation, `infra/...` for tooling and CI.
2. Make one focused change, with its tests, docs and a changelog entry.
3. Run `scripts/check.sh` until it passes.
4. Commit with a message that says the change's effect in the imperative
   mood, and give the net line change per file with the reason for each
   addition, as `AGENTS.md` asks.
5. Open a pull request against `main` with `Closes #N` in the description,
   and work through the template's checklist. CodeRabbit reviews it; its
   findings are advisory, and each is reproduced before anyone acts on it.

## Licence

By contributing you agree that your contribution is licensed under the MIT
licence in `LICENSE`.
