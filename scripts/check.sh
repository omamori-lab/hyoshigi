#!/usr/bin/env bash
# Run every gate: the gitleaks secrets scan, ruff, the spread check, ty, the
# test suite, the lanorme standards, a package build, and an import of the
# built wheel in a fresh environment. Run this before committing or finishing
# a change. No arguments.
#
#   scripts/check.sh
#
# Exits non-zero on the first failing gate. CI runs this same script. The
# pre-commit hooks run only the fast gates: ruff, ty, lanorme and the test
# suite, plus a gitleaks scan of the staged changes.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

echo "==> secrets (gitleaks)"
if command -v gitleaks >/dev/null 2>&1; then
  # --verbose names the rule, file and commit of a finding; --redact hides the secret.
  gitleaks_options=(--config .gitleaks.toml --redact --verbose --no-banner)
  # The history of the checked-out commit.
  gitleaks git . --log-opts=HEAD "${gitleaks_options[@]}"
  # The staged changes, then the unstaged ones.
  gitleaks git . --pre-commit --staged "${gitleaks_options[@]}"
  gitleaks git . --pre-commit "${gitleaks_options[@]}"
  # Each untracked file git does not ignore: gitleaks dir takes one path, and
  # ./ keeps a name that starts with a dash from reading as an option.
  git ls-files --others --exclude-standard -z |
    while IFS= read -r -d '' path; do
      gitleaks dir "./$path" "${gitleaks_options[@]}"
    done
else
  echo "skipped: gitleaks is not on PATH; CI's secrets job scans the whole history."
fi

echo "==> ruff"
uv run --group dev ruff check .
uv run --group dev ruff format --check .

echo "==> spread"
# Every tracked Python file and the python blocks of the Markdown files.
git ls-files -z '*.py' README.md CONTRIBUTING.md 'docs/*.md' |
  xargs -0 uv run --group dev python scripts/spread.py --check

echo "==> ty (type checking)"
uv run --group dev ty check

echo "==> test suite"
uv run --group dev pytest -q

echo "==> lanorme standards"
# lanorme needs Python 3.13+; it only parses the code, so it checks 3.12 source correctly.
uvx --python 3.13 lanorme==0.21.1 check .

echo "==> build"
uv build >/dev/null

echo "==> import the built wheel"
wheel="$(ls -t dist/hyoshigi-*.whl | head -n 1)"
# A fresh environment: `uv run --with` reuses a cached install of a wheel whose
# name and version have not changed, and would import an older build.
wheel_env="$(mktemp -d)"
trap 'rm -rf "$wheel_env"' EXIT
uv venv --quiet --python 3.12 "$wheel_env"
uv pip install --quiet --python "$wheel_env/bin/python" "$wheel"
"$wheel_env/bin/python" -c "import hyoshigi as package; print(package.__version__, len(package.__all__), 'public names')"

echo
echo "OK: all gates passed."
