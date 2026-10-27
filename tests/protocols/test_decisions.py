"""A control protocol's decisions and proposals: the values that cannot work."""

import pytest

from hyoshigi import (
    Action,
    Audit,
    Block,
    ConfigurationError,
    Halt,
    Policy,
    Proposal,
    Rationale,
    Replace,
    Run,
    SuspicionScore,
)

READ = Action(id="a1")
WRITE = Action(id="a2")
READS_THE_FILE = SuspicionScore(
    value=0.1,
    explanation="It reads the file the task names.",
    evidence=Rationale(model="m"),
)
PROPOSED_READ = Proposal(
    action=READ,
    score=READS_THE_FILE,
    policy=Policy.UNTRUSTED,
)
REASON = "The write is outside the task."
NOT_PROPOSED = "the action must be the action of one of the proposals"
SCORED_BY_POLICY = "an untrusted proposal has a SuspicionScore, and a trusted one None"


@pytest.mark.parametrize(
    ("kind", "fields", "message"),
    [
        pytest.param(
            Run,
            {
                "action": READ,
                "proposals": [PROPOSED_READ],
                "position": 0,
            },
            "proposals must be a tuple, got an instance of list",
            id="proposals-a-list",
        ),
        pytest.param(
            Audit,
            {
                "action": READ,
                "proposals": (
                    PROPOSED_READ,
                    READ,
                ),
                "position": 0,
            },
            "proposals[1] must be a Proposal, got an instance of Action",
            id="proposal-an-action",
        ),
        pytest.param(
            Halt,
            {
                "reason": REASON,
                "proposals": (),
                "position": 0,
            },
            "proposals must hold at least one Proposal",
            id="no-proposals",
            marks=pytest.mark.xfail(strict=True),
        ),
        pytest.param(
            Run,
            {
                "action": READ,
                "proposals": (PROPOSED_READ,),
                "position": True,
            },
            "position must be a whole number of at least 0, got True",
            id="position-a-bool",
        ),
        pytest.param(
            Halt,
            {
                "reason": REASON,
                "proposals": (PROPOSED_READ,),
                "position": -1,
            },
            "position must be at least 0, got -1",
            id="position-below-zero",
        ),
        pytest.param(
            Run,
            {
                "action": WRITE,
                "proposals": (PROPOSED_READ,),
                "position": 0,
            },
            NOT_PROPOSED,
            id="run-of-an-action-not-proposed",
        ),
        pytest.param(
            Replace,
            {
                "action": WRITE,
                "proposals": (PROPOSED_READ,),
                "position": 0,
            },
            NOT_PROPOSED,
            id="replace-with-an-action-not-proposed",
        ),
        pytest.param(
            Block,
            {
                "action": WRITE,
                "reason": REASON,
                "proposals": (PROPOSED_READ,),
                "position": 0,
            },
            NOT_PROPOSED,
            id="block-of-an-action-not-proposed",
        ),
        pytest.param(
            Audit,
            {
                "action": WRITE,
                "proposals": (PROPOSED_READ,),
                "position": 0,
            },
            NOT_PROPOSED,
            id="audit-of-an-action-not-proposed",
        ),
        pytest.param(
            Halt,
            {
                "reason": 5,
                "proposals": (PROPOSED_READ,),
                "position": 0,
            },
            "reason must be a str, got an instance of int",
            id="halt-with-a-number-as-the-reason",
        ),
        pytest.param(
            Proposal,
            {
                "action": READ,
                "score": None,
                "policy": Policy.UNTRUSTED,
            },
            SCORED_BY_POLICY,
            id="untrusted-without-a-score",
        ),
        pytest.param(
            Proposal,
            {
                "action": "a1",
                "score": READS_THE_FILE,
                "policy": Policy.UNTRUSTED,
            },
            "action must be an Action, got an instance of str",
            id="action-an-id",
            marks=pytest.mark.xfail(strict=True),
        ),
        pytest.param(
            Proposal,
            {
                "action": READ,
                "score": None,
                "policy": "trusted",
            },
            "policy must be a Policy, got an instance of str",
            id="policy-a-string",
            marks=pytest.mark.xfail(strict=True),
        ),
        pytest.param(
            Proposal,
            {
                "action": READ,
                "score": 0.1,
                "policy": Policy.UNTRUSTED,
            },
            SCORED_BY_POLICY,
            id="untrusted-with-a-number-as-the-score",
        ),
        pytest.param(
            Proposal,
            {
                "action": READ,
                "score": READS_THE_FILE,
                "policy": Policy.TRUSTED,
            },
            SCORED_BY_POLICY,
            id="trusted-with-a-score",
        ),
    ],
)
def test_a_value_that_cannot_work_raises_naming_the_problem(
    kind: type,
    fields: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ConfigurationError) as raised:
        kind(**fields)

    assert str(raised.value) == message


@pytest.mark.xfail(strict=True)
@pytest.mark.parametrize(
    ("kind", "fields"),
    [
        pytest.param(
            Block,
            {
                "action": READ,
                "reason": "",
                "proposals": (PROPOSED_READ,),
                "position": 0,
            },
            id="block",
        ),
        pytest.param(
            Halt,
            {
                "reason": "",
                "proposals": (PROPOSED_READ,),
                "position": 0,
            },
            id="halt",
        ),
    ],
)
def test_a_decision_with_a_reason_may_give_an_empty_one(
    kind: type,
    fields: dict[str, object],
) -> None:
    decision = kind(**fields)

    assert decision.reason == ""
