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

### Changed

- `Redaction`'s `media` is optional. Left out, it puts
  `OpaquePart(kind=media_type)` in place of each media part, so a redaction
  that only masks text keeps and sends no media part, and the rating of an
  action whose parts or tool calls hold media fails closed. Media in a tool
  call's arguments is text, which only `text` reaches. Pass
  `media=lambda part: part` to keep media as it is.

### Fixed

- A monitor fails the rating closed on an `OpaquePart` that a redaction puts
  in the rated action's parts or tool calls, as on one the action holds. A
  model that perceives the media it replaced rated the placeholder.

## [0.5.0] - 2026-10-28

### Added

- `GenerativeModel` and `AsyncGenerativeModel` take `top_log_probabilities`, a
  whole number from 0, 0 by default: the number of likeliest alternative
  tokens the model's function returns with each generated token.
  `Generation` gains `log_probabilities`, a tuple of `TokenLogProbability`,
  each holding a token, its log-probability and its `alternatives`, a tuple of
  `AlternativeToken`. Both new classes are exported. A value of the wrong type,
  a negative count, a log-probability above 0 or NaN raises
  `ConfigurationError` when the object is built.
- `GuardMonitor` and `AsyncGuardMonitor` over a model with
  `top_log_probabilities` above 0 ask once and score the probability on the
  suspicious labels at the label's first token, in place of the share of
  `samples` labelled suspicious. The explanation states one of three findings
  and no number. A reply that carries no log-probabilities, whose tokens do not
  spell its text, or whose label token has no alternatives or does not start
  the label's kind fails the rating closed as `Unreadable`. `samples` is
  checked and not used on this path, and `HardLabelWarning` is raised only when
  the monitor samples. A model that declares 0 is sampled as before.
- `Redaction`, a text function and a media function, redacts a
  trajectory's content where it leaves the process: text parts, thoughts,
  tool-call arguments, observations and their sources, media, their
  descriptions and the UTF-8 text of text media, a score's explanation and
  the reply text in its cause, the proposals in records, and a block's or a
  halt's reason. Ids, tool names, authors, values and positions stay as they
  are. Each point where content leaves takes it as its own option, off unless
  set.
- `Trajectory(redaction=...)`, and the same option on `from_run`,
  `from_anthropic`, `from_openai_chat`, `from_open_responses` and `load`,
  keeps each entry, rating and record redacted before it is appended. Split
  copies and subagents keep the redaction, a record's action stays its
  history entry, and a monitor reads an action rated against the trajectory
  through the same redaction, once: `rate_latest_step` does not redact the
  action it takes from the history again.
- `View(redaction=...)` redacts the history and the rated action a monitor's
  model reads, after the trajectory's own, and media lifted from a tool
  call's arguments passes through both media functions. Media the model
  cannot perceive still fails the rating closed.
- `Trajectory.dump(redaction=...)` writes the trajectory, its ratings,
  records and subagents redacted, and leaves the trajectory unchanged.
  `load(data, redaction=...)` redacts what it reads. The stored format is
  unchanged.
- `LangfuseScores`, `LangSmithFeedback` and `AsyncLangSmithFeedback` take
  `redaction`, and send the explanation through it as the comment.
- `Trajectory.join(subagent, call_id=...)` holds a subagent that ran in
  another process, such as one loaded from the dump a worker returns, under
  the tool call that delegated it. It returns the copy it holds: the load
  of the subagent's dump through the trajectory's redaction, in the run's
  sequence. The joined records take the run's next numbers in the order of
  their own, so two subagents that ran at once from one loaded run replay
  one after the other, in the order they were joined. `BlockAndFeedBack`
  counts their blocks in the run's total. The trajectory given is
  unchanged. `ConfigurationError` is raised, and nothing changes,
  for a subagent that is not a `Trajectory`, that a call delegated, or that
  belongs to this run, such as its root or a copy split from it. It is
  raised too for a call that is not in the trajectory or that delegated a
  subagent already, and for a tree that would reach past 32 levels below
  the root. The stored format is unchanged.
- `Trajectory.load` takes `content`, a `StoredContent`, which is exported.
  `RAW`, the default, redacts everything loaded through the redaction, as
  before. `REDACTED` loads content stored redacted as it is and keeps the
  redaction for the entries added later, so a redaction that changes its
  own output does not compound. A `content` that is not a `StoredContent`,
  and `REDACTED` without a redaction, raise `ConfigurationError` before the
  data is read.
- `GenerativeModel.from_litellm` and `AsyncGenerativeModel.from_litellm` build
  a model on the caller's `litellm.completion` or `litellm.acompletion`, with
  `perceives` and `top_log_probabilities`. Above 0 the request asks for the
  log-probabilities and the model returns those the provider sends, so a
  `GuardMonitor` scores in one call; a reply without them fails the guard
  closed. The caller binds the key, the retries and the other options of the
  call with `functools.partial`. hyoshigi never imports or installs LiteLLM,
  and the adapter follows version 1.106.5. LiteLLM's content-policy,
  rejected-request, guardrail and blocked-entity errors are `Refused`, its
  context-window and image-fetch errors are `Rejected(status=400)`, and a
  client error that blames the request raises `ConfigurationError` naming
  LiteLLM and the status.
- `ReviewFlag`, exported, marks a step for a person to review. `name` is the
  mark's name in the backend, `flagged` by default, and `threshold` is the
  score at or above which a report's `send` flags the step, a number from 0 to
  1, `None` by default. `LangfuseScores`, `LangSmithFeedback` and
  `AsyncLangSmithFeedback` take `review_flag`. When the score reaches its
  threshold, `send` sends the flag before the score, on the same observation
  or run: a boolean score of value 1 in Langfuse and boolean feedback in
  LangSmith, with no comment, so a `redaction` does not apply to it. Their
  `flag` method sends it for any step with the ids `send` takes, for a
  `Flag` decision, and raises `ConfigurationError` on a report built without
  a `review_flag`. `AsyncLangSmithFeedback.flag` is awaited. The flag goes out
  on the report's own path: Langfuse's queue, the executor or task group when
  given, inline when not. A flag post that fails or is cancelled on a worker
  is logged as `did not send the flag`. A blank `name`, a `threshold` that is
  not a number from 0 to 1 (NaN and a bool included) and a `review_flag` that
  is not a `ReviewFlag` raise `ConfigurationError` when built.

## [0.4.0] - 2026-10-28

### Added

- `DeferToResample` and `AsyncDeferToResample` take `on_audit`, an
  `AuditResponse`. `HOLD`, the default, returns `Audit` as in 0.3.0. `FLAG`
  returns `Flag`, a new decision, wherever the protocol would return
  `Audit`, with the same action and proposals: the action runs, and a person
  audits the step afterwards, which is detection only. `FLAG` is allowed
  only for trusted monitoring, with `max_resamples=0`, equal thresholds and
  an `on_unavailable` other than `TRUSTED_STEP`, and any other settings
  raise `ConfigurationError` when the protocol is built. `Trajectory.record`
  adds a `Flag`'s action and its score to the history, as for `Run`, and a
  `Flag` ends a run of blocks. The stored records gain the kind `flag`;
  0.3.0 cannot read a file that holds a flag.

### Changed

- `resume` raises `ConfigurationError` for a request the protocol never asks
  for: a `Resample` that draws more than `max_resamples` resamples in all,
  and a `TrustedStep` under `on_audit=AuditResponse.FLAG`. This is a
  breaking change: 0.3.0 rated the resamples of a `Resample` built past
  `max_resamples`.
- `Flag` joins `ProtocolDecision`, and `Trajectory.record` names it in the
  `ConfigurationError` for a decision of another type. This is a breaking
  change: an exhaustive `match` over a `ProtocolDecision` that ends in
  `assert_never` fails under a type checker until it handles `Flag`.
- `DeferToResample` and `AsyncDeferToResample` take `resampling`, a
  `Resampling`. With `SEQUENTIAL`, the default, `decide` asks for one
  resample, and `resume` asks for one more until a resample is below
  `defer_threshold`, a resample failed closed with a cause other than
  `Unavailable`, `max_resamples` resamples are drawn, or the answer holds
  fewer proposals than the request asked for. `PARALLEL` asks for all
  `max_resamples` in one request, as 0.3.0 did. This is a breaking change:
  `decide` now asks for one resample, so a program that answers with
  `max_resamples` proposals without reading the request's `count` raises
  `ConfigurationError`, and `resume` can return another `Resample`.
  `resampling=Resampling.PARALLEL` keeps the requests of 0.3.0.
- `BlockAndFeedBack` and `AsyncBlockAndFeedBack` count `blocks_in_total`
  over the whole run: every record of the delegation tree, in the order it
  was recorded, so a subagent and its delegator count the same total. The
  trajectory that `split_latest_step` returns for a subagent counts in the
  subagent's place. The count in a row still reads the trajectory's own
  records. In 0.3.0 each trajectory counted only its own blocks, so an agent
  could stay under the total by delegating. This is a breaking change for a
  run with subagents, which can now reach `Audit` where 0.3.0 ran.
- The stored trajectory format is version 3, in which each record holds its
  `sequence`, its number in the order the run recorded it, and `load` reads
  only version 3. A repeated, decreasing or negative `sequence` raises
  `FormatError`. This is a breaking change: a version-2 file raises
  `FormatError`, since it lacks the order of records across a run; rebuild
  the trajectory from the run and dump it again.
- `Trajectory.delegate` raises `ConfigurationError` for a subagent more than
  32 levels below the root, the depth `load` reads. This is a breaking
  change for a deeper tree, which 0.3.0 built and dumped but could not load.
- The trajectory that `split_latest_step` returns holds copies of the
  subagents' trajectories, at every depth, delegated by the new trajectory,
  so a record on a copy leaves the original unchanged. 0.3.0 shared the
  subagents with the trajectory it split. This is a breaking change for code
  that records on a subagent of the split trajectory and reads the record
  from the original.

## [0.3.0] - 2026-10-27

### Added

- `LangSmithFeedback` takes an `executor`. With one, `send` submits the post
  to it and returns at once, so a rating returns without waiting for
  LangSmith's answer; a post that fails or is cancelled there is logged by
  the report's class and the error's type, never raised, and
  `executor.shutdown(wait=True)` waits for the posts left. Without one, the
  report posts and waits, as in 0.2.0.
- `AsyncLangSmithFeedback` takes a `task_group`. With one, `send` creates a
  task for the post in the group and returns at once; a post that fails or
  is cancelled there is logged by the report's class and the error's type,
  never raised, and leaving the group waits for the posts left. Without
  one, the report awaits the post, as in 0.2.0.

### Changed

- `Action` and `Observation` raise `ConfigurationError` for a list given as
  `thought`, `parts` or `tool_calls`, where 0.2.0 took one. This is a
  breaking change: code that builds one from a list passes a tuple instead.

### Fixed

- An action, a tool call, an observation or a part built with a field of the
  wrong type, such as `ToolObservation(author="monitor", ...)`, failed only
  later. `Trajectory.dump` and a monitor's rendering raised `AttributeError`
  or `TypeError`, `dump` raised an `AssertionError` that quoted a part of the
  wrong type, or `Trajectory.load` refused the stored file. Each raises
  `ConfigurationError` when built, naming the field and the value's type, and
  `MediaPart` does so for a `media_type` that is not a string, where it raised
  `TypeError`. `Trajectory.act` and `observe` raise it for an event of the
  other kind.

## [0.2.0] - 2026-10-27

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
- The docs pages, the README and the default prompt's module carry the
  training-data canary that alignment.org/canary lists, so a dataset builder
  can leave them out of training corpora.
- Docs pages on connecting a model, and on what each provider adapter
  sends and what each reply or error becomes.
- Docs pages on tracing ratings, reporting scores to Langfuse and LangSmith,
  what a calibrated score means, and why a rating is a span whose score is
  reported for it, with the reasons and sources the `CalibratedMonitor`,
  `tracing.py`, `reporting.py` and `reports.py` docstrings held.
- `LangSmithFeedback` and `AsyncLangSmithFeedback` take `stop_after_attempt`,
  the number of attempts the client's `create_feedback` makes, 10 by default.
- `Rejected`, the outcome of a prompt the provider says it cannot use, with
  the HTTP status if there was one. A monitor fails the rating closed with it
  at once, as with a refusal, and a trajectory stores it as a `rejected`
  cause, which 0.1.0 cannot read. A model function of one's own can return
  it.
- `Trajectory.record` keeps a control protocol's decision, a `Run`,
  `Replace`, `Block`, `Audit` or `Halt` with the `Proposal`s it weighed and
  its position, in `Trajectory.records`, and adds to the history what the
  decision ran or blocked. A blocked action joins the history with the
  reason as the monitor's error for each of its tool calls, so
  `ToolObservation` takes an `author`, `TOOL` by default. A monitor reads
  another author in the element, as `<tool_error call_id="call-1"
  author="monitor">`.
- `DeferToResample` and `BlockAndFeedBack`, the first control protocols, each
  with an async twin, decide what becomes of the agent's proposal from their
  monitor's score. `DeferToResample` can instead return a `Resample` or
  `TrustedStep` request, which the program answers through `resume`; each
  takes a `FailureResponse` as `on_unavailable`, `AUDIT` by default, for a
  rating that got no answer.

### Changed

- A prompt over the context window, an image the provider cannot use and a
  413 are `Rejected` with the status on every adapter, where they were
  `Unavailable`, which says that no answer came; so is a failed Responses
  reply whose error says the prompt cannot be used. A guard sample or a
  repeated call that is rejected fails the rating closed, where an
  unavailable one was set aside while most answered. This is a breaking
  change: an exhaustive `match` over `Cause`, `GenerationOutcome` or
  `DecisionOutcome` needs a `Rejected` case, and a `match` on
  `Unavailable(status=400)` for a prompt over the context window now finds
  `Rejected(status=400)`.

- A `GuardMonitor` sample or a `RepeatedMonitor` call that got no answer,
  `Unavailable`, is set aside: when more than half the samples or calls
  answered, the rating scores those that answered, and otherwise it fails
  closed as `Unavailable`. A refusal, an unreadable reply or a cut-off one
  still fails the rating closed, with its cause even after an unanswered
  sample or call.
- `Labels.unavailable` keeps a guard's unanswered samples, which
  `Trajectory.dump` writes and `load` reads. 0.1.0 cannot read a stored mean
  of the calls that answered.
- The stored trajectory format is version 2, which holds the records, each
  tool observation's author and every label's unanswered samples, and `load`
  reads only version 2. A version-1 file raises `FormatError`: rebuild the
  trajectory, from the run or with a version that reads the file, and dump it
  again. A seeded `CalibratedMonitor` keys its draw on the stored history, so
  its draw for a history that holds a tool observation changes.
- A `Composite` fails closed by the same rule, a part that is itself a
  composite counting by the cause it failed closed with, and `failed_closed`
  and the cause on a span or a report follow it.
- A completed reply from OpenAI or OpenRouter that holds no message is an
  empty generation, which a monitor reads as unreadable, as an empty
  Anthropic reply already was, so it is no longer set aside as unavailable.
- A refusal part from OpenAI or OpenRouter whose text is missing, null or not
  a string is a refusal without an explanation, so it is no longer set aside
  as unavailable.
- `GenerativeModel.from_anthropic` and `AsyncGenerativeModel.from_anthropic`
  take only the `Anthropic` and `AsyncAnthropic` clients their signatures
  declare, and their subclasses. `AnthropicBedrock`, `AnthropicVertex` and
  `AnthropicBedrockMantle`, which subclass neither, now raise
  `ConfigurationError` when the model is built.

### Fixed

- OpenAI's `invalid_prompt` raised `ConfigurationError`, so the agent's
  content could crash a rating, and a scaffold that reads an exception as
  allow would let the action pass. It is `Rejected`.
- An OpenRouter 403, such as a guardrail's block, raised `ConfigurationError`
  unless it named a refusal or a policy. Every 403 is `Refused`, so a spent
  budget, a missing permission or the wrong data region, which OpenRouter
  also answers with a 403, now scores every action 1.0.
- OpenRouter's `invalid_prompt` with no error type, which can stand for a
  context overflow, raised `ConfigurationError` at a 400 and was
  `Unavailable` in a failed reply. It is `Rejected`.
- A 400 that names no reason, for a prompt that carries media, raised
  `ConfigurationError` on Anthropic, TypeSafe and OpenRouter Decisions, each
  of which answers an image it cannot use that way. It is
  `Rejected(status=400)`, so with media in the prompt a 400 for misuse, such
  as Anthropic's spend limit, fails closed too; for a text-only prompt it
  still raises.
- The OpenAI adapter's docstring said OpenAI retains none of the monitored
  agent's content. `store: false` stops the Responses API storing the
  response; OpenAI's abuse-monitoring logs, which can hold prompts, are
  kept for up to 30 days whatever `store` says.
- A `FormatError` from `Trajectory.load` inside a subagent named the subagent
  by its position, as in `$.subagents[1]`, which is not a JSONPath query:
  `subagents` is an object keyed by tool call id. The path now names the call
  id, as in `$.subagents['call_8']`, escaped as RFC 9535 escapes a member name
  in a Normalized Path. A key that is not a string is reported at
  `$.subagents`, and an unpaired surrogate in a call id, which no JSONPath can
  name, is escaped as `\ud800`. A `FormatError` quotes no value from the
  payload, but its path can hold a member name.
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
  reads the top-level `system`. An image or a document block in one raises
  `FormatError`, as Anthropic allows neither there.
- A model built with a client of the other family, such as
  `GenerativeModel.from_anthropic(AsyncAnthropic())`, built, and then every
  rating was `Unavailable`, or the first rating raised `AttributeError` or
  `TypeError`. Each Anthropic, OpenAI and TypeSafe constructor now raises
  `ConfigurationError` when the model is built, naming the constructor that
  takes that client.
- A model's function that returned a value outside its outcome union raised
  `AssertionError`, whose message quoted the value, which can hold a model's
  reply. It raises `ConfigurationError` naming the value's type only.
- A model built with a function that is not callable, a `perceives` that is
  not a `MediaTypes` or a name that is not a string failed only at the first
  rating, if at all, and a provider constructor given such a `perceives`
  raised `AttributeError`. Each model class and each provider constructor
  raises `ConfigurationError` when the model is built.
- A monitor given a model of the other kind named its own twin in the hint,
  as `LLMMonitor` given a `DecisionModel` said to use `AsyncLLMMonitor`. It
  names the monitors that take that model: `DecisionMonitor` for a decision
  model, `LLMMonitor` or `GuardMonitor` for a generative one.

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

[Unreleased]: https://github.com/omamori-lab/hyoshigi/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/omamori-lab/hyoshigi/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/omamori-lab/hyoshigi/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/omamori-lab/hyoshigi/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/omamori-lab/hyoshigi/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/omamori-lab/hyoshigi/releases/tag/v0.1.0
