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
- `Authorship`, `by_role` and `marked`: who wrote each user-role message.
- `OpaquePart`, for content recorded but not read, and media ranges such as
  `image/*` in `MediaPart`.
