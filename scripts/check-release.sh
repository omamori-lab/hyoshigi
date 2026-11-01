#!/usr/bin/env bash
# Check that the tree agrees with the version X.Y.Z of a release. It changes
# nothing, and prints nothing when the tree agrees.
#
#   scripts/check-release.sh [X.Y.Z]
#
# scripts/check.sh runs it with no version, which checks the version in
# __version__, and the Release workflow runs it with the tag's version. It
# checks that:
#   - __version__ in src/hyoshigi/__init__.py is X.Y.Z;
#   - CHANGELOG.md has a non-empty "## [X.Y.Z] - YYYY-MM-DD" section and an
#     "[X.Y.Z]: https://" link reference;
#   - CITATION.cff has version X.Y.Z and date-released equal to that date.
#
# It reports every problem it finds and exits 1 if there was any.
set -euo pipefail

cd "$(dirname "$0")/.."

package_version="$(sed -n -E 's/^__version__ = "([^"]*)"$/\1/p' src/hyoshigi/__init__.py)"
version="${1:-${package_version}}"
if [[ ! "${version}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "check-release: '${version}' is not a version X.Y.Z." >&2
  echo "usage: scripts/check-release.sh [X.Y.Z]" >&2
  exit 2
fi
escaped_version="${version//./\\.}"

problems=0
report() {
  # On GitHub Actions each problem becomes an error annotation on the run.
  if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
    echo "::error::$1" >&2
  else
    echo "check-release: $1" >&2
  fi
  problems=$((problems + 1))
}

if [[ "${package_version}" != "${version}" ]]; then
  report "__version__ is '${package_version}', not ${version}."
fi

release_date="$(sed -n -E "s/^## \[${escaped_version}\] - ([0-9]{4}-[0-9]{2}-[0-9]{2})$/\1/p" CHANGELOG.md | head -n 1)"
if [[ -z "${release_date}" ]]; then
  report "CHANGELOG.md has no '## [${version}] - YYYY-MM-DD' section."
else
  # The section runs to the next "## [" heading; its lines that are neither
  # blank nor headings are counted.
  entries="$(awk -v heading="## [${version}] - " '
    index($0, heading) == 1 { grab = 1; next }
    grab && /^## \[/ { exit }
    grab && /[^ \t]/ && !/^#/ { count++ }
    END { print count + 0 }
  ' CHANGELOG.md)"
  if [[ "${entries}" -eq 0 ]]; then
    report "the CHANGELOG.md section for ${version} is empty."
  fi
fi
if ! grep -q -E "^\[${escaped_version}\]: https://[^[:space:]]" CHANGELOG.md; then
  report "CHANGELOG.md has no '[${version}]: https://...' link reference."
fi

# A value is read in double quotes or bare; one with a lone quote reads as none.
citation_version="$(sed -n -E 's/^version: ("([^"]*)"|([^"]*))$/\2\3/p' CITATION.cff)"
if [[ "${citation_version}" != "${version}" ]]; then
  report "CITATION.cff has version '${citation_version}', not ${version}."
fi
citation_date="$(sed -n -E 's/^date-released: ("([^"]*)"|([^"]*))$/\2\3/p' CITATION.cff)"
if [[ -n "${release_date}" && "${citation_date}" != "${release_date}" ]]; then
  report "CITATION.cff has date-released '${citation_date}', not ${release_date} as in CHANGELOG.md."
fi

if [[ "${problems}" -gt 0 ]]; then
  echo "check-release: ${problems} problem(s); the tree does not agree with ${version}." >&2
  exit 1
fi
