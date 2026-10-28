# API

Every public name imports from the package root, as `from hyoshigi import Trajectory`.

## Trajectory

::: hyoshigi.Trajectory

::: hyoshigi.BlackBoxRun

::: hyoshigi.Step

::: hyoshigi.Rating

::: hyoshigi.Event

::: hyoshigi.Action

::: hyoshigi.Observation

::: hyoshigi.ToolCall

::: hyoshigi.ToolObservation

::: hyoshigi.Part

::: hyoshigi.TextPart

::: hyoshigi.MediaPart

::: hyoshigi.OpaquePart

::: hyoshigi.Description

::: hyoshigi.Author

::: hyoshigi.Authorship

::: hyoshigi.by_role

::: hyoshigi.marked

::: hyoshigi.JSON

::: hyoshigi.Redaction

## Models

::: hyoshigi.GenerativeModel

::: hyoshigi.AsyncGenerativeModel

::: hyoshigi.DecisionModel

::: hyoshigi.AsyncDecisionModel

::: hyoshigi.Prompt

::: hyoshigi.PromptPart

::: hyoshigi.MediaTypes

::: hyoshigi.Question

::: hyoshigi.GenerationOutcome

::: hyoshigi.DecisionOutcome

::: hyoshigi.Generation

::: hyoshigi.TokenLogProbability

::: hyoshigi.AlternativeToken

::: hyoshigi.Decision

::: hyoshigi.Truncated

::: hyoshigi.Refused

::: hyoshigi.Rejected

::: hyoshigi.Unavailable

## Protocols

::: hyoshigi.DeferToResample

::: hyoshigi.AsyncDeferToResample

::: hyoshigi.BlockAndFeedBack

::: hyoshigi.AsyncBlockAndFeedBack

::: hyoshigi.FailureResponse

::: hyoshigi.Resampling

::: hyoshigi.AuditResponse

::: hyoshigi.ProtocolRequest

::: hyoshigi.Resample

::: hyoshigi.TrustedStep

::: hyoshigi.ProtocolDecision

::: hyoshigi.Run

::: hyoshigi.Flag

::: hyoshigi.Replace

::: hyoshigi.Block

::: hyoshigi.Audit

::: hyoshigi.Halt

::: hyoshigi.Proposal

::: hyoshigi.Policy

## Monitors

::: hyoshigi.Monitor

::: hyoshigi.AsyncMonitor

::: hyoshigi.LLMMonitor

::: hyoshigi.AsyncLLMMonitor

::: hyoshigi.MonitorPrompt

::: hyoshigi.DEFAULT_PROMPT

::: hyoshigi.DecisionMonitor

::: hyoshigi.AsyncDecisionMonitor

::: hyoshigi.Aggregation

::: hyoshigi.DEFAULT_STATEMENT

::: hyoshigi.ATTACK_CRITERIA

::: hyoshigi.GuardMonitor

::: hyoshigi.AsyncGuardMonitor

::: hyoshigi.GuardLabels

::: hyoshigi.GPT_OSS_SAFEGUARD

::: hyoshigi.QWEN3GUARD

::: hyoshigi.HardLabelWarning

::: hyoshigi.View

::: hyoshigi.Channel

## Scores

::: hyoshigi.SuspicionScore

::: hyoshigi.Evidence

::: hyoshigi.Rationale

::: hyoshigi.Decisions

::: hyoshigi.StatementDecision

::: hyoshigi.Labels

::: hyoshigi.Composite

::: hyoshigi.FailedClosed

::: hyoshigi.Cause

::: hyoshigi.Unreadable

::: hyoshigi.Unperceivable

## Composition

::: hyoshigi.RepeatedMonitor

::: hyoshigi.AsyncRepeatedMonitor

::: hyoshigi.CalibratedMonitor

::: hyoshigi.AsyncCalibratedMonitor

::: hyoshigi.CascadeMonitor

::: hyoshigi.AsyncCascadeMonitor

## Reports

::: hyoshigi.ReportedMonitor

::: hyoshigi.AsyncReportedMonitor

::: hyoshigi.LangfuseScores

::: hyoshigi.LangSmithFeedback

::: hyoshigi.AsyncLangSmithFeedback

## Errors

::: hyoshigi.ConfigurationError

::: hyoshigi.FormatError
