# AGENTS.md

Guidance for coding agents working in this repository (the
[agents.md](https://agents.md/) standard). `CLAUDE.md` imports this file; edit
this one.

hyoshigi is the framework-free control core, one of omamori's tools: monitors
rate an agent's action against its trajectory, and control protocols decide
what runs. The domain speaks reinforcement learning's terms (a trajectory is
a history of actions and observations) and AI control's (suspicion score,
audit, defer to trusted).
langchain-sync-monitors will be rebuilt on it.

## How we work

- **Design before code.** No module is written before the decision it
  depends on is agreed and recorded with its evidence.
- **Primary sources only**: arXiv, Science, AI lab blogs, known AI safety
  organisations, LessWrong, canonical computer science, AI and security venues.
  Never news sites, aggregators or summaries. Cite at the point of use.
- **Canonical terms.** A name comes from RL, AI control, security or software
  engineering before it is coined; a coined term is defined once.
- **No metaspeech.** Pages, docs and docstrings say what things are and do,
  never how to use the page they are on.
- **One gate.** `scripts/check.sh` runs every check; it passes before a
  commit, and CI runs the same script.

## Code

1. **Less is more.** Every added line needs a reason. Prefer simplifying,
   merging and deleting over adding. Each commit or pull request gives the net
   line change per file, with the reason for each addition. "It was simpler to
   add" is a review finding.
2. **Merge only like things.** Combine code only when it would change for the
   same reason. Never couple unrelated parts behind a new abstraction.
3. **Canonical libraries first.** Read the installed version's documentation
   and source; never trust memory for a version, an option or a URL. A new
   dependency starts at its latest release with no known vulnerability. The
   core has no runtime dependencies, only the standard library of Python 3.12
   or newer; the shipped model's httpx and stamina sit behind an extra.
4. **Types by use.**
   - Data checked at a boundary (provider payloads in the format builders, a
     stored trajectory, a model's HTTP response) is validated with `match`
     mapping and class patterns, which ignore extra keys and coerce nothing,
     and raise `FormatError` naming the path, never the input's values.
   - Frozen dataclasses for values the program builds: actions, observations,
     parts, scores.
   - A TypedDict only for an exact foreign JSON shape.
   - A named type for a tuple or dict that means something in the domain.
   - No `Any`.
5. **Names.** Functions and methods are verbs; classes and modules are nouns;
   nothing is abbreviated. Sync and async twins share names: `Monitor.rate` and
   `AsyncMonitor.rate`, never `rate_async`.
6. **Imports go at the top of the module.** A lazy import says why: a cycle,
   its cost, or an optional extra.
7. **Size limits are rules of thumb.** Split a file when the split makes the
   code clearer.
8. **A failure is an outcome, and an outcome is a type.** Outcomes the domain
   expects are values in a closed union of frozen dataclasses, matched
   exhaustively with `match` and `typing.assert_never`: a model's
   `generate` returns `Generation | Truncated | Refused | Unavailable`, and a
   monitor always returns a `SuspicionScore`, with any failure in its
   evidence. A reply the monitor cannot read, content it cannot perceive and
   a model that is unavailable all fail closed: suspicion 1.0, never "allow".
   Misuse raises: `ConfigurationError` when an object is built, or a method
   called, with arguments that cannot work, `FormatError` when a reader is
   given a payload that is not its format. Any other exception from hyoshigi's own code is a bug.
9. **Docstrings and comments say what the code does now, and stay dry.**
   History belongs in git and in the recorded decisions.
10. **Design patterns we keep.** Monitors and protocols hold configuration
    only, never state between calls; the state is the trajectory. Behaviour
    that varies is a strategy object or an enum, never a bool or a plain
    string. Composition for parts, inheritance for kinds. Sync and async
    classes drive one sans-I/O plan, so their logic is written once.
    Parameters after the first are keyword-only. Every model is a parameter;
    nothing picks one by default.

## Tests

1. **Arrange, Act, Assert.** One act per test, the three parts separated by
   blank lines. The test's name is a sentence about the behaviour; a name that
   needs "and" is two tests.
2. **Tests first.** Each new test is committed before its code, marked as
   expected to fail (`pytest.mark.xfail(strict=True)`), and seen failing on its
   assertion. The commit with the code removes the mark.
3. **Few test doubles.** The domain (trajectory, formats, views, rendering,
   reply parsing, wrappers, protocols) always runs for real. Doubles are
   allowed only for:
   - a scripted `MonitorModel` returning fixed generations, since the model is
     the outside service;
   - stand-ins on the wire for model providers (an httpx mock transport for
     OpenRouter and OpenAI-style APIs).

   Tests never check calls made to hyoshigi's own code.
4. **Every expected value has a source independent of the code:**
   - a literal written from the rule;
   - a published example (a provider's or a specification's example payload,
     a paper's worked example);
   - a property (a suspicion stays in [0, 1]; a smoothed conformal p-value is
     uniform on honest scores).

   Never the code's own output, and no logic in test code.
5. **The domain is mutation-tested.** No surviving mutant is left unexplained.
6. **Tests go through public interfaces**: the public classes and functions,
   so a refactoring breaks a test only when it would break a caller.
7. **Both families.** Every behaviour of a monitor or a model is tested under
   the sync and the async class, by parametrising one test, not by writing it
   twice.

Also:

- **Few and strong.** Delete a test that duplicates another's coverage or
  restates the code. Shared setup goes in fixtures, and data-only variations
  in one parametrised test.
- **Every test earns its place**: it protects a behaviour a caller relies on,
  an invariant, or a fixed bug.
- **Mutation-check every new test**: break the code it protects, watch it
  fail, and report how many such checks ran.
- **For results that vary between runs, compare only the parts that do not
  vary.**
