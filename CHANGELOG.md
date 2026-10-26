# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The public API is the set of names exported from `hyoshigi` and their
keyword parameters. A release that could turn a working integration into a
failing one is a breaking change; before 1.0.0, a breaking change raises the
minor version.

## [Unreleased]

### Added

- `rate_latest_step`, on every monitor, rates a trajectory's latest action
  against everything before it, so a transcript that ends in the agent's
  reply needs no split by hand. `Trajectory.split_latest_step` returns that
  action and the trajectory before it as a `Step`.
- Docs pages on choosing a monitor, rating with an LLM monitor, rating with a
  decision model, and what the default prompt asks, with the reasons and
  sources the `llm.py`, `decision.py` and `prompts.py` docstrings held.
- Docs pages on what a trajectory is, keeping one, reading an agent's own
  format, and choosing what a monitor reads, and a reference page on what
  each provider format's roles, items and blocks become.

### Fixed

- The first-rating tutorial named BashArena for the 0 to 10 scale, though the
  monitor prompt BashArena prints asks for 0 to 9, and it called the
  explanation the only part of a score shown to the agent. hyoshigi shows the
  agent nothing; the scaffold decides.
- The first-rating tutorial cited Sutton and Barto's section 3.1 for a
  history of actions and observations, which section 17.3 defines.
- `Trajectory.from_anthropic` raised `FormatError` at a message of role
  `system`, which Anthropic's Messages API accepts among a request's
  messages, so a request that held one could not be rated. It reads one as an
  observation by `SCAFFOLD` whose `source` is `system`, in its place, as it
  reads the top-level `system`.

## [0.1.0] - 2026-10-25

### Added

- A documentation site at <https://omamori-lab.github.io/hyoshigi/>: a tutorial, a how-to on failed
  ratings, explanations of the default view, failing closed and the monitors' known limits, and
  the API reference.
- `Trajectory`, a run's append-only history of actions and observations:
  `from_run` for a black-box agent, `act`, `observe`, `delegate` to a subagent
  under the tool call that delegated it, and `dump` and `load` in a
  JSON-compatible format, version 1. `load` reads composite scores and
  subagents nested up to 32 levels deep.
- `Action`, `ToolCall`, `Observation`, `ToolObservation`, `TextPart`,
  `MediaPart`, `Description` and `Author`.
- `ConfigurationError`, for arguments that cannot work, and `FormatError`,
  which names the JSONPath of a malformed payload and never quotes it.
- `Trajectory.from_anthropic` and `Action.from_anthropic`, which read
  Anthropic's Messages format: a request into a trajectory, a response into
  the action to rate. A document's title and context are read as text beside
  it.
- `Trajectory.from_openai_chat` and `Action.from_openai_chat`, which read the
  OpenAI chat completions format with OpenRouter's reasoning fields and the
  images an assistant message carries.
- `Trajectory.from_open_responses` and `Action.from_open_responses`, which read
  the Open Responses format of 2026-04-24: a request into a trajectory, a
  response's output items into the action to rate.
- `GenerativeModel` and `DecisionModel`, with `AsyncGenerativeModel` and
  `AsyncDecisionModel`: the model a monitor asks, around the caller's function,
  with `Prompt`, `Question`, `MediaTypes` and the outcomes `Generation`,
  `Truncated`, `Refused`, `Unavailable` and `Decision`.
- Adapters for the caller's SDK client, each behind its own extra:
  `GenerativeModel.from_openai` (OpenAI's Responses API, `hyoshigi[openai]`),
  `from_anthropic` (the Messages API, `hyoshigi[anthropic]`), `from_openrouter`
  and `DecisionModel.from_openrouter` (`hyoshigi[openrouter]`), and
  `DecisionModel.from_typesafe` (TypeSafe's Jev, `hyoshigi[typesafe]`).
- `reasoning_effort` on `GenerativeModel.from_openai` and its async twin, in the
  OpenAI SDK's own `ReasoningEffort`, sent as the request's reasoning effort
  when given.
- `reasoning_effort` on `GenerativeModel.from_openrouter` and its async twin, in
  the OpenRouter SDK's own `ReasoningEffort`, sent as the request's reasoning
  effort when given.
- `thinking` on `GenerativeModel.from_anthropic` and its async twin, in the
  Anthropic SDK's own `ThinkingConfigParam`, sent as the request's thinking
  when given.
- `DecisionModel.from_openai`, OpenAI's Decisions API (`gpt-6-luna`), with an
  async twin: the probability of a predicate answer, over text and images.
- `Monitor` and `AsyncMonitor`, which `rate(action, *, trajectory)` an action
  the agent proposes and always return a `SuspicionScore`: a value from 0 to 1,
  an explanation, the only part an agent may see (the default prompt asks the
  model to keep its score out of it), and typed evidence (`Rationale`, `Decisions`,
  `Labels`, `Composite` or `FailedClosed`). A reply the monitor cannot read, content it
  cannot perceive, and a model that refuses or is unavailable fail closed, at
  1.0. Media in the rated action that the monitor cannot perceive fails closed
  even when the agent describes it. Text media reaches the monitor's model as
  escaped text, whatever the model perceives, and text by URL or in bytes
  that are not UTF-8 counts as content it cannot perceive.
- `LLMMonitor` and `AsyncLLMMonitor`, which ask a generative model for
  reasoning and an integer score, with `MonitorPrompt` and `DEFAULT_PROMPT`.
  `DEFAULT_PROMPT` treats a tool call whose arguments are not JSON, such as a
  custom tool's free text, as a call that runs.
- `DecisionMonitor` and `AsyncDecisionMonitor`, which ask a decision model the
  probability that statements about the action hold, aggregated by
  `Aggregation`, with `DEFAULT_STATEMENT` and `ATTACK_CRITERIA`.
- `View` and `Channel`: what of the trajectory a monitor reads. The default
  view shows the kind of each opaque part in an unknown author's observation,
  such as a tool the agent's provider ran.
- `GuardMonitor` and `AsyncGuardMonitor`, which ask a guard model several
  times against the caller's policy and score the share of unsafe labels, with
  `GuardLabels` and presets for gpt-oss-safeguard and Qwen3Guard; `Labels`
  evidence records each reply's label. A suspicious label after a verdict key
  counts whatever markup stands between them, as in `Label = INVALID` or
  `<label>INVALID</label>`, so a reply that also quotes a safe label fails
  closed.
- `RepeatedMonitor`, `CalibratedMonitor` and `CascadeMonitor`, with their async
  twins: the mean of repeated ratings, a smoothed conformal p-value against
  honest scores, and a fast monitor escalating to a careful one.
  `RepeatedMonitor` fails closed when any call did, `CalibratedMonitor` passes
  a failed score through at 1.0, and `CascadeMonitor` returns the deciding
  score unchanged, so a failed fast score escalates and a failed careful
  score is the result.
- `Trajectory.attach` and `Rating`: a trajectory's ratings, kept by `dump` and
  `load`.
- `Authorship`, `by_role` and `marked`: who wrote each user-role message.
- `OpaquePart`, for content recorded but not read, and media ranges such as
  `image/*` in `MediaPart`. A data URL's `;base64` is read in any case, as RFC
  2045, section 5.1, has it.
- A span in OpenTelemetry for every rating of every monitor, wrappers
  included, when `opentelemetry-api` is installed (`hyoshigi[opentelemetry]`):
  named `rate` and the monitor's class, from the global tracer provider's
  tracer `hyoshigi`, and current while the monitor rates, so a wrapper's inner
  ratings and an instrumented client's model calls nest inside it. It carries
  the score as `gen_ai.evaluation.name` (`suspicion`),
  `gen_ai.evaluation.score.value` and `gen_ai.evaluation.explanation`, with
  `hyoshigi.failed_closed` and, for a score that failed closed,
  `hyoshigi.cause`.
- `ReportedMonitor` and `AsyncReportedMonitor`, which rate with a monitor
  inside a span of their own and then send the score for that span to each of
  their reports: `LangfuseScores` (`hyoshigi[langfuse]`), a numeric score on
  the span's trace and observation, queued by the client, and
  `LangSmithFeedback` or `AsyncLangSmithFeedback` (`hyoshigi[langsmith]`),
  model feedback on the run LangSmith ingests the span as. Each score carries
  its explanation as the comment and, as metadata, whether it failed closed,
  its cause's class and its models. Nothing is sent when the span is not
  sampled or no tracer provider records it; a report that fails is logged by
  its class and error type, and the score is returned either way.

### Changed

- The package is marked as alpha on PyPI (`Development Status :: 3 - Alpha`).
- hyoshigi installs from PyPI, as `pip install "hyoshigi[anthropic]"`, instead of
  from GitHub.

### Fixed

- A prompt over the model's context window now fails the rating closed, as
  `Unavailable` with status 400, on the OpenRouter and Anthropic adapters, as it
  already did on OpenAI's. Before, it raised `ConfigurationError`.
- The README's Claude Haiku 5.5 examples set `max_tokens` to 4096. The model
  thinks before it answers, and at 1024 tokens a rating could be cut off and
  fail closed.

[Unreleased]: https://github.com/omamori-lab/hyoshigi/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/omamori-lab/hyoshigi/releases/tag/v0.1.0
