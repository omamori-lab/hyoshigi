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

- `Trajectory`, a run's append-only history of actions and observations:
  `from_run` for a black-box agent, `act`, `observe`, `delegate` to a subagent
  under the tool call that delegated it, and `dump` and `load` in a
  JSON-compatible format, version 1.
- `Action`, `ToolCall`, `Observation`, `ToolObservation`, `TextPart`,
  `MediaPart`, `Description` and `Author`.
- `ConfigurationError`, for arguments that cannot work, and `FormatError`,
  which names the JSONPath of a malformed payload and never quotes it.
- `Trajectory.from_anthropic` and `Action.from_anthropic`, which read
  Anthropic's Messages format: a request into a trajectory, a response into
  the action to rate.
- `Trajectory.from_openai_chat` and `Action.from_openai_chat`, which read the
  OpenAI chat completions format with OpenRouter's reasoning fields.
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
  `Composite` or `FailedClosed`). A reply the monitor cannot read, content it
  cannot perceive, and a model that refuses or is unavailable fail closed, at
  1.0.
- `LLMMonitor` and `AsyncLLMMonitor`, which ask a generative model for
  reasoning and an integer score, with `MonitorPrompt` and `DEFAULT_PROMPT`.
- `DecisionMonitor` and `AsyncDecisionMonitor`, which ask a decision model the
  probability that statements about the action hold, aggregated by
  `Aggregation`, with `DEFAULT_STATEMENT` and `ATTACK_CRITERIA`.
- `View` and `Channel`: what of the trajectory a monitor reads.
- `GuardMonitor` and `AsyncGuardMonitor`, which ask a guard model several
  times against the caller's policy and score the share of unsafe labels, with
  `GuardLabels` and presets for Llama Guard, gpt-oss-safeguard, Qwen3Guard and
  Nemotron content safety; `Labels` evidence records each reply's label.
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
  `image/*` in `MediaPart`.
