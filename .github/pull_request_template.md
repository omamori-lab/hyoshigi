## What this changes

<!-- What the change does and why, in a few sentences. -->

Closes #

## Checklist

- [ ] The design this change depends on is agreed in the linked issue.
- [ ] `scripts/check.sh` passes: the gitleaks secrets scan, ruff, the
      spread check, ty, the test suite, lanorme, the package build and the
      wheel import.
- [ ] Each new test was committed first, marked `xfail(strict=True)`, and
      seen failing on its assertion; its expected value comes from a literal,
      a published example or a property, never from the code.
- [ ] Behaviour of a monitor or a model is tested under the sync and the
      async class in one parametrised test.
- [ ] Ideas taken from a paper, post or code base are cited at the point of
      use.
- [ ] `CHANGELOG.md` has an entry under `## [Unreleased]` for anything a user
      would notice.
