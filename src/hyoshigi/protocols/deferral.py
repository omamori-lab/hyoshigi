"""Defer to resample: a suspicious proposal is resampled, and the least suspicious sample decides
the step.
"""

from dataclasses import dataclass
from typing import NamedTuple, assert_never

from hyoshigi.concurrency import Plan, drive, drive_concurrently
from hyoshigi.errors import ConfigurationError
from hyoshigi.models.model import Unavailable
from hyoshigi.monitors.monitor import AsyncMonitor, Monitor
from hyoshigi.monitors.score import SuspicionScore, find_cause
from hyoshigi.options import check_instance, describe_type, read_count, read_threshold
from hyoshigi.protocols.decisions import (
    Audit,
    Policy,
    Proposal,
    ProtocolDecision,
    Replace,
    Run,
)
from hyoshigi.protocols.requests import (
    FailureResponse,
    ProtocolRequest,
    Resample,
    TrustedStep,
    decide_on_unavailable,
)
from hyoshigi.trajectory.history import Action
from hyoshigi.trajectory.trajectory import Trajectory


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class DeferToResample:
    """Runs the agent's proposal below `defer_threshold`; otherwise asks for up to `max_resamples`
    more proposals from the agent's model, and decides the step from all of them.

    `decide` rates the proposal. A score that failed closed is Audit, or, when the monitor got
    no answer, what `on_unavailable` says: Audit, Block with the score's explanation as the
    reason, a TrustedStep request, or Run. A score below `defer_threshold` is Run, and any other
    a Resample request, or with `max_resamples` at 0 the step's decision. `resume` rates the
    proposals that answer a Resample, and decides the step from the proposals so far and them.

    The step's decision, over the agent's proposals in order: the first whose score failed
    closed with a cause other than Unavailable is Audit. Otherwise the resamples whose monitor
    got no answer are set aside, and the step's score is the least suspicious remaining value.
    At or above `audit_threshold`, the most suspicious remaining proposal is Audit; otherwise
    the least suspicious below `defer_threshold` is Replace, the first of a tie each time;
    otherwise the step is a TrustedStep request. Bhatt et al. 2025, "Ctrl-Z", sec. 3.2.

    A TrustedStep answered with one action is Replace with it, unrated, and with none is Audit
    of the most suspicious remaining proposal, the first of a tie. Every decision holds the
    proposals weighed, in the order they were rated, and the length of the history.

    With `max_resamples` at 0 this is defer to trusted, and with equal thresholds as well,
    trusted monitoring (Greenblatt et al. 2023, "AI Control", secs. 3.3, 3.2).

    The thresholds are numbers from 0 to 1, `defer_threshold` at most `audit_threshold`, kept
    as floats; `max_resamples` is a whole number of at least 0, kept as an int; `monitor` is a
    Monitor and `on_unavailable` a FailureResponse. Anything else raises ConfigurationError
    when the protocol is built.

    <https://omamori-lab.github.io/hyoshigi/explanation/control-protocols/>
    """

    monitor: Monitor
    defer_threshold: float
    audit_threshold: float
    max_resamples: int
    on_unavailable: FailureResponse = FailureResponse.AUDIT

    def __post_init__(self) -> None:
        check_instance(
            self.monitor,
            expected=Monitor,
            name="monitor",
            hint="Use AsyncDeferToResample for an AsyncMonitor.",
        )
        _check_options(self)

    def decide(
        self,
        proposed: Action,
        *,
        trajectory: Trajectory,
    ) -> ProtocolDecision | ProtocolRequest:
        """Rate the proposed action, which is not yet in the trajectory, and decide about it or
        ask for more proposals.
        """
        return drive(
            plan_decision(
                self,
                proposed=proposed,
                trajectory=trajectory,
            ),
            ask=lambda action: self.monitor.rate(
                action,
                trajectory=trajectory,
            ),
        )

    def resume(
        self,
        request: ProtocolRequest,
        *,
        proposals: tuple[Action, ...],
        trajectory: Trajectory,
    ) -> ProtocolDecision | ProtocolRequest:
        """Rate each of the proposals that answer the request, in turn, and decide the step or ask
        for a trusted step.

        A request that is not a Resample or a TrustedStep, a trajectory whose history is no
        longer the request's position long, proposals that are not a tuple of Action, more
        proposals than the request asks for, and a TrustedStep answered with one of the agent's
        proposals raise ConfigurationError.
        """
        check_answer(
            request,
            proposals=proposals,
            trajectory=trajectory,
        )
        return drive(
            plan_resumption(
                self,
                request=request,
                proposals=proposals,
            ),
            ask=lambda action: self.monitor.rate(
                action,
                trajectory=trajectory,
            ),
        )


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class AsyncDeferToResample:
    """Decides as DeferToResample does, with the monitor awaited and the resamples rated at once.

    One failed rating cancels the others and its error is raised as it is.
    """

    monitor: AsyncMonitor
    defer_threshold: float
    audit_threshold: float
    max_resamples: int
    on_unavailable: FailureResponse = FailureResponse.AUDIT

    def __post_init__(self) -> None:
        check_instance(
            self.monitor,
            expected=AsyncMonitor,
            name="monitor",
            hint="Use DeferToResample for a Monitor.",
        )
        _check_options(self)

    async def decide(
        self,
        proposed: Action,
        *,
        trajectory: Trajectory,
    ) -> ProtocolDecision | ProtocolRequest:
        """Rate the proposed action, which is not yet in the trajectory, and decide about it or
        ask for more proposals.
        """
        return await drive_concurrently(
            plan_decision(
                self,
                proposed=proposed,
                trajectory=trajectory,
            ),
            ask=lambda action: self.monitor.rate(
                action,
                trajectory=trajectory,
            ),
        )

    async def resume(
        self,
        request: ProtocolRequest,
        *,
        proposals: tuple[Action, ...],
        trajectory: Trajectory,
    ) -> ProtocolDecision | ProtocolRequest:
        """Rate the proposals that answer the request, all at once, and decide the step or ask
        for a trusted step.

        A request that is not a Resample or a TrustedStep, a trajectory whose history is no
        longer the request's position long, proposals that are not a tuple of Action, more
        proposals than the request asks for, and a TrustedStep answered with one of the agent's
        proposals raise ConfigurationError.
        """
        check_answer(
            request,
            proposals=proposals,
            trajectory=trajectory,
        )
        return await drive_concurrently(
            plan_resumption(
                self,
                request=request,
                proposals=proposals,
            ),
            ask=lambda action: self.monitor.rate(
                action,
                trajectory=trajectory,
            ),
        )


def _check_options(protocol: DeferToResample | AsyncDeferToResample) -> None:
    """Keep the thresholds as floats and the resample count as an int, or raise
    ConfigurationError naming the option that cannot work.
    """
    defer_threshold = read_threshold(
        protocol.defer_threshold,
        name="defer_threshold",
    )
    audit_threshold = read_threshold(
        protocol.audit_threshold,
        name="audit_threshold",
    )
    if defer_threshold > audit_threshold:
        raise ConfigurationError(
            f"defer_threshold ({defer_threshold}) must be at most "
            f"audit_threshold ({audit_threshold})",
        )
    max_resamples = read_count(
        protocol.max_resamples,
        name="max_resamples",
        minimum=0,
    )
    check_instance(
        protocol.on_unavailable,
        expected=FailureResponse,
        name="on_unavailable",
    )
    # The dataclass is frozen, so the values are set past its guard.
    object.__setattr__(
        protocol,
        "defer_threshold",
        defer_threshold,
    )
    object.__setattr__(
        protocol,
        "audit_threshold",
        audit_threshold,
    )
    object.__setattr__(
        protocol,
        "max_resamples",
        max_resamples,
    )


def check_answer(
    request: ProtocolRequest,
    *,
    proposals: tuple[Action, ...],
    trajectory: Trajectory,
) -> None:
    """Raise ConfigurationError unless the request is a Resample or a TrustedStep at the length
    of the history, and the proposals a tuple of at most as many actions as it asks for, none of
    them, for a TrustedStep, one of the agent's proposals.
    """
    limit = _read_limit(request)
    if request.position != len(trajectory.history):
        raise ConfigurationError("the request's position is not the length of the history")
    check_instance(
        proposals,
        expected=tuple,
        name="proposals",
    )
    for index, action in enumerate(proposals):
        check_instance(
            action,
            expected=Action,
            name=f"proposals[{index}]",
        )
    if len(proposals) > limit:
        raise ConfigurationError(
            f"the request asks for {limit} or fewer proposals, got {len(proposals)}",
        )
    if isinstance(
        request,
        TrustedStep,
    ) and any(proposal.action in proposals for proposal in request.proposals):
        raise ConfigurationError(
            "a TrustedStep's answer must be the trusted model's action, not one of the agent's "
            "proposals",
        )


def _read_limit(request: ProtocolRequest) -> int:
    """The number of proposals the request asks for at most, or ConfigurationError for a value
    that is not a request.
    """
    match request:
        case Resample(count=count):
            return count
        case TrustedStep():
            return 1
        case _:
            raise ConfigurationError(
                f"request must be a Resample or TrustedStep, got {describe_type(request)}",
            )


def plan_decision(
    protocol: DeferToResample | AsyncDeferToResample,
    *,
    proposed: Action,
    trajectory: Trajectory,
) -> Plan[Action, SuspicionScore, ProtocolDecision | ProtocolRequest]:
    """Plan the decision about the agent's proposal: rate it, then answer a failure, run it,
    ask for resamples, or decide the step.
    """
    position = len(trajectory.history)
    (score,) = yield (proposed,)
    proposal = Proposal(
        action=proposed,
        score=score,
        policy=Policy.UNTRUSTED,
    )
    if score.failed_closed:
        return decide_on_failure(
            protocol,
            proposal=proposal,
            score=score,
            position=position,
        )
    if score.value < protocol.defer_threshold:
        return Run(
            action=proposed,
            proposals=(proposal,),
            position=position,
        )
    if protocol.max_resamples > 0:
        return Resample(
            count=protocol.max_resamples,
            proposals=(proposal,),
            position=position,
        )
    return weigh_step(
        (proposal,),
        defer_threshold=protocol.defer_threshold,
        audit_threshold=protocol.audit_threshold,
        position=position,
    )


def decide_on_failure(
    protocol: DeferToResample | AsyncDeferToResample,
    *,
    proposal: Proposal,
    score: SuspicionScore,
    position: int,
) -> ProtocolDecision | TrustedStep:
    """Decide about the agent's proposal when its score failed closed: what `on_unavailable` says
    when the monitor got no answer, and Audit for any other cause.
    """
    match find_cause(score), protocol.on_unavailable:
        case Unavailable(), FailureResponse.TRUSTED_STEP:
            return TrustedStep(
                proposals=(proposal,),
                position=position,
            )
        case (
            Unavailable(),
            FailureResponse.AUDIT | FailureResponse.BLOCK | FailureResponse.RUN as response,
        ):
            return decide_on_unavailable(
                response,
                proposal=proposal,
                reason=score.explanation,
                position=position,
            )
        case _:
            return Audit(
                action=proposal.action,
                proposals=(proposal,),
                position=position,
            )


def plan_resumption(
    protocol: DeferToResample | AsyncDeferToResample,
    *,
    request: ProtocolRequest,
    proposals: tuple[Action, ...],
) -> Plan[Action, SuspicionScore, ProtocolDecision | ProtocolRequest]:
    """Plan the answer to a request: rate a Resample's proposals and decide the step, or answer
    a TrustedStep.
    """
    match request:
        case Resample():
            scores = yield proposals
            weighed = (
                *request.proposals,
                *(
                    Proposal(
                        action=action,
                        score=score,
                        policy=Policy.UNTRUSTED,
                    )
                    for action, score in zip(
                        proposals,
                        scores,
                        strict=True,
                    )
                ),
            )
            return weigh_step(
                weighed,
                defer_threshold=protocol.defer_threshold,
                audit_threshold=protocol.audit_threshold,
                position=request.position,
            )
        case TrustedStep():
            return decide_on_trusted_step(
                request,
                proposals=proposals,
            )
        case unreachable:
            assert_never(unreachable)


class RatedProposal(NamedTuple):
    """One of the agent's proposals and its score."""

    proposal: Proposal
    score: SuspicionScore


def list_untrusted(proposals: tuple[Proposal, ...]) -> list[RatedProposal]:
    """The agent's proposals among the proposals, in order, each with its score."""
    return [
        RatedProposal(
            proposal=proposal,
            score=proposal.score,
        )
        for proposal in proposals
        if proposal.score is not None
    ]


def list_remaining(proposals: tuple[Proposal, ...]) -> list[RatedProposal]:
    """The agent's proposals the step's score is taken over, in order: the first, the proposed
    action, and each resample whose monitor got an answer.
    """
    proposed, *resamples = list_untrusted(proposals)
    return [
        proposed,
        *(
            rated
            for rated in resamples
            if not isinstance(
                find_cause(rated.score),
                Unavailable,
            )
        ),
    ]


def read_suspicion(rated: RatedProposal) -> float:
    """The suspicion of a rated proposal."""
    return rated.score.value


def weigh_step(
    proposals: tuple[Proposal, ...],
    *,
    defer_threshold: float,
    audit_threshold: float,
    position: int,
) -> Replace | Audit | TrustedStep:
    """Decide the step from the agent's proposals, the first of them the proposed action and the
    rest its resamples.

    The first proposal whose score failed closed with a cause other than Unavailable is Audit.
    Otherwise the resamples whose monitor got no answer are set aside, and the step's score is the
    least suspicious remaining value: at or above the audit threshold the most suspicious remaining
    proposal is Audit; otherwise the least suspicious below the defer threshold is Replace, the
    first of a tie each time; otherwise the step is a TrustedStep request. Bhatt et al. 2025,
    "Ctrl-Z", sec. 3.2.
    """
    failed = next(
        (rated for rated in list_untrusted(proposals) if has_failed_with_an_answer(rated.score)),
        None,
    )
    if failed is not None:
        return Audit(
            action=failed.proposal.action,
            proposals=proposals,
            position=position,
        )
    remaining = list_remaining(proposals)
    least = min(
        remaining,
        key=read_suspicion,
    )
    if least.score.value >= audit_threshold:
        most = max(
            remaining,
            key=read_suspicion,
        )
        return Audit(
            action=most.proposal.action,
            proposals=proposals,
            position=position,
        )
    if least.score.value < defer_threshold:
        return Replace(
            action=least.proposal.action,
            proposals=proposals,
            position=position,
        )
    return TrustedStep(
        proposals=proposals,
        position=position,
    )


def has_failed_with_an_answer(score: SuspicionScore) -> bool:
    """Whether the score failed closed with a cause other than Unavailable."""
    cause = find_cause(score)
    return cause is not None and not isinstance(
        cause,
        Unavailable,
    )


def decide_on_trusted_step(
    request: TrustedStep,
    *,
    proposals: tuple[Action, ...],
) -> Replace | Audit:
    """Replace the proposed action with the trusted model's, which is not rated; with none, audit
    the most suspicious remaining proposal, the first of a tie.
    """
    match proposals:
        case (action,):
            return Replace(
                action=action,
                proposals=(
                    *request.proposals,
                    Proposal(
                        action=action,
                        score=None,
                        policy=Policy.TRUSTED,
                    ),
                ),
                position=request.position,
            )
        case _:
            most = max(
                list_remaining(request.proposals),
                key=read_suspicion,
            )
            return Audit(
                action=most.proposal.action,
                proposals=request.proposals,
                position=request.position,
            )
