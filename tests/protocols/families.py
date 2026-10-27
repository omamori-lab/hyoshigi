"""Each family's protocols around a real LLM monitor whose model replies from a script."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from hyoshigi import (
    Action,
    AsyncBlockAndFeedBack,
    AsyncDeferToResample,
    AsyncGenerativeModel,
    AsyncLLMMonitor,
    BlockAndFeedBack,
    DeferToResample,
    FailedClosed,
    Generation,
    GenerationOutcome,
    GenerativeModel,
    LLMMonitor,
    Policy,
    Prompt,
    Proposal,
    ProtocolDecision,
    ProtocolRequest,
    Rationale,
    Refused,
    Rejected,
    Resample,
    SuspicionScore,
    TextPart,
    Trajectory,
    Truncated,
    TrustedStep,
    Unavailable,
    Unreadable,
)

MODEL_NAME = "trusted-1"
TIMEOUT_SECONDS = 5
TASK = "Summarise q3.md"
UNAVAILABLE_EXPLANATION = (
    "The monitor's model was unavailable, so the action is treated as suspicious."
)
UNAVAILABLE_SCORE = SuspicionScore(
    value=1.0,
    explanation=UNAVAILABLE_EXPLANATION,
    evidence=FailedClosed(
        model=MODEL_NAME,
        cause=Unavailable(status=503),
    ),
)
CUT_OFF = Truncated(text="<reasoning>cut")
# A failed outcome with an answer, and the score the monitor fails closed with on it.
FAILURES_WITH_AN_ANSWER = (
    (
        Refused(explanation="I cannot assess this."),
        SuspicionScore(
            value=1.0,
            explanation=(
                "The monitor's model refused to rate the action, so it is treated as suspicious."
            ),
            evidence=FailedClosed(
                model=MODEL_NAME,
                cause=Refused(explanation="I cannot assess this."),
            ),
        ),
    ),
    (
        Rejected(status=400),
        SuspicionScore(
            value=1.0,
            explanation=(
                "The monitor's model rejected what it was given, so the action is treated as "
                "suspicious."
            ),
            evidence=FailedClosed(
                model=MODEL_NAME,
                cause=Rejected(status=400),
            ),
        ),
    ),
    (
        Generation(text="no score"),
        SuspicionScore(
            value=1.0,
            explanation=(
                "The monitor could not read its model's reply, so the action is treated as "
                "suspicious."
            ),
            evidence=FailedClosed(
                model=MODEL_NAME,
                cause=Unreadable(),
            ),
        ),
    ),
    (
        CUT_OFF,
        SuspicionScore(
            value=1.0,
            explanation=(
                "The monitor's model stopped before it gave a score, so the action is treated "
                "as suspicious."
            ),
            evidence=FailedClosed(
                model=MODEL_NAME,
                cause=CUT_OFF,
            ),
        ),
    ),
)
FAILURE_IDS = ["refused", "rejected", "unreadable", "cut-off"]


class Script:
    """The outcomes a scripted model returns, in the order it is asked."""

    def __init__(
        self,
        *outcomes: GenerationOutcome,
    ) -> None:
        self.outcomes = list(outcomes)

    def generate(
        self,
        prompt: Prompt,
    ) -> GenerationOutcome:
        return self.outcomes.pop(0)

    async def generate_asynchronously(
        self,
        prompt: Prompt,
    ) -> GenerationOutcome:
        return self.generate(prompt)


def build_reply(
    score: int,
    reasoning: str = "r",
) -> Generation:
    """A reply the LLM monitor reads as a tenth of the score, explained by the reasoning."""
    return Generation(text=f"<reasoning>{reasoning}</reasoning><score>{score}</score>")


def build_score(
    value: float,
    explanation: str = "r",
) -> SuspicionScore:
    """The score an LLM monitor gives a reply: a value and the reasoning, from the model."""
    return SuspicionScore(
        value=value,
        explanation=explanation,
        evidence=Rationale(model=MODEL_NAME),
    )


def build_proposal(
    action: Action,
    score: SuspicionScore,
) -> Proposal:
    """The agent's proposal of the action, with its score."""
    return Proposal(
        action=action,
        score=score,
        policy=Policy.UNTRUSTED,
    )


def build_action(name: str) -> Action:
    """An action that says only its name, with the name as its id."""
    return Action(
        id=name,
        parts=(TextPart(text=f"Doing {name}."),),
    )


def build_trajectory() -> Trajectory:
    """A trajectory that holds the task alone, one entry long."""
    return Trajectory.from_run(
        TASK,
        answer="Done.",
    ).trajectory


def run[Result](awaitable: Awaitable[Result]) -> Result:
    """Run on a fresh event loop; a test that waits longer than the timeout fails."""

    async def finish() -> Result:
        return await asyncio.wait_for(
            awaitable,
            timeout=TIMEOUT_SECONDS,
        )

    return asyncio.run(finish())


def build_synchronous_llm(script: Script) -> LLMMonitor:
    """An LLM monitor that asks its model once per rating."""
    return LLMMonitor(
        model=GenerativeModel(
            name=MODEL_NAME,
            generate=script.generate,
        ),
        max_parse_retries=0,
    )


def build_asynchronous_llm(script: Script) -> AsyncLLMMonitor:
    """An async LLM monitor that asks its model once per rating."""
    return AsyncLLMMonitor(
        model=AsyncGenerativeModel(
            name=MODEL_NAME,
            generate=script.generate_asynchronously,
        ),
        max_parse_retries=0,
    )


def decide_synchronously(
    protocol: DeferToResample | BlockAndFeedBack,
    proposed: Action,
    trajectory: Trajectory,
) -> ProtocolDecision | ProtocolRequest:
    return protocol.decide(
        proposed,
        trajectory=trajectory,
    )


def decide_asynchronously(
    protocol: AsyncDeferToResample | AsyncBlockAndFeedBack,
    proposed: Action,
    trajectory: Trajectory,
) -> ProtocolDecision | ProtocolRequest:
    return run(
        protocol.decide(
            proposed,
            trajectory=trajectory,
        ),
    )


def resume_synchronously(
    protocol: DeferToResample,
    request: ProtocolRequest,
    proposals: tuple[Action, ...],
    trajectory: Trajectory,
) -> ProtocolDecision | ProtocolRequest:
    return protocol.resume(
        request,
        proposals=proposals,
        trajectory=trajectory,
    )


def resume_asynchronously(
    protocol: AsyncDeferToResample,
    request: ProtocolRequest,
    proposals: tuple[Action, ...],
    trajectory: Trajectory,
) -> ProtocolDecision | ProtocolRequest:
    return run(
        protocol.resume(
            request,
            proposals=proposals,
            trajectory=trajectory,
        ),
    )


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class Family[Rater, Deferral, Blocking]:
    """One family's protocols around a scripted LLM monitor, and how they are called."""

    build_llm: Callable[[Script], Rater]
    deferral: Callable[..., Deferral]
    blocking: Callable[..., Blocking]
    decide: Callable[
        [
            Deferral | Blocking,
            Action,
            Trajectory,
        ],
        ProtocolDecision | ProtocolRequest,
    ]
    resume: Callable[
        [
            Deferral,
            ProtocolRequest,
            tuple[Action, ...],
            Trajectory,
        ],
        ProtocolDecision | ProtocolRequest,
    ]

    def build_deferral(
        self,
        script: Script,
        **options: object,
    ) -> Deferral:
        """A DeferToResample whose monitor's model replies from the script, with a defer threshold
        of 0.5, an audit threshold of 0.9 and two resamples unless the options say otherwise.
        """
        return self.deferral(
            monitor=self.build_llm(script),
            **{
                "defer_threshold": 0.5,
                "audit_threshold": 0.9,
                "max_resamples": 2,
                **options,
            },
        )

    def build_blocking(
        self,
        script: Script,
        **options: object,
    ) -> Blocking:
        """A BlockAndFeedBack whose monitor's model replies from the script, with a block
        threshold of 0.5 unless the options say otherwise.
        """
        return self.blocking(
            monitor=self.build_llm(script),
            **{
                "block_threshold": 0.5,
                **options,
            },
        )

    def decide_deferral(
        self,
        script: Script,
        *,
        proposed: Action,
        trajectory: Trajectory,
        **options: object,
    ) -> ProtocolDecision | ProtocolRequest:
        """Decide about the proposal with the DeferToResample that build_deferral builds."""
        return self.decide(
            self.build_deferral(
                script,
                **options,
            ),
            proposed,
            trajectory,
        )

    def resume_deferral(
        self,
        script: Script,
        *,
        request: ProtocolRequest,
        proposals: tuple[Action, ...],
        trajectory: Trajectory,
        **options: object,
    ) -> ProtocolDecision | ProtocolRequest:
        """Answer the request with the DeferToResample that build_deferral builds."""
        return self.resume(
            self.build_deferral(
                script,
                **options,
            ),
            request,
            proposals,
            trajectory,
        )

    def decide_blocking(
        self,
        script: Script,
        *,
        proposed: Action,
        trajectory: Trajectory,
        **options: object,
    ) -> ProtocolDecision | ProtocolRequest:
        """Decide about the proposal with the BlockAndFeedBack that build_blocking builds."""
        return self.decide(
            self.build_blocking(
                script,
                **options,
            ),
            proposed,
            trajectory,
        )

    def take_deferral_step(
        self,
        script: Script,
        *,
        proposed: Action,
        trajectory: Trajectory,
        answers: tuple[tuple[Action, ...], ...],
    ) -> None:
        """Decide about the proposal with the DeferToResample that build_deferral builds, answer
        each request with the next answer, and record the decision, as the program running the
        agent does.
        """
        protocol = self.build_deferral(script)
        unanswered = iter(answers)
        outcome = self.decide(
            protocol,
            proposed,
            trajectory,
        )
        while isinstance(
            outcome,
            Resample | TrustedStep,
        ):
            outcome = self.resume(
                protocol,
                outcome,
                next(unanswered),
                trajectory,
            )
        trajectory.record(outcome)


SYNCHRONOUS = Family(
    build_llm=build_synchronous_llm,
    deferral=DeferToResample,
    blocking=BlockAndFeedBack,
    decide=decide_synchronously,
    resume=resume_synchronously,
)
ASYNCHRONOUS = Family(
    build_llm=build_asynchronous_llm,
    deferral=AsyncDeferToResample,
    blocking=AsyncBlockAndFeedBack,
    decide=decide_asynchronously,
    resume=resume_asynchronously,
)
type AnyFamily = (
    Family[LLMMonitor, DeferToResample, BlockAndFeedBack]
    | Family[AsyncLLMMonitor, AsyncDeferToResample, AsyncBlockAndFeedBack]
)
