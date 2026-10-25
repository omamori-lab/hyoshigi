"""Dumping a trajectory to JSON-compatible data, and loading it back."""

import math

import pytest

from hyoshigi import (
    JSON,
    Action,
    Author,
    Description,
    FormatError,
    MediaPart,
    Observation,
    OpaquePart,
    TextPart,
    ToolCall,
    ToolObservation,
    Trajectory,
)
from hyoshigi.models.model import Decision, Refused, Truncated, Unavailable
from hyoshigi.monitors.score import (
    Cause,
    Composite,
    Decisions,
    FailedClosed,
    Labels,
    Rationale,
    StatementDecision,
    SuspicionScore,
    Unperceivable,
    Unreadable,
)
from hyoshigi.trajectory.trajectory import Rating

TASK = Observation(
    author=Author.USER,
    parts=(TextPart(text="Plot q3.csv"),),
)
READ_FILE = Action(
    id="a1",
    thought=(
        TextPart(text="I should read the file first."),
        OpaquePart(kind="redacted_thinking"),
    ),
    parts=(TextPart(text="Reading q3.csv."),),
    tool_calls=(
        ToolCall(
            id="call_1",
            name="read_file",
            arguments='{"path": "q3.csv"}',
        ),
    ),
)
CHART = ToolObservation(
    call_id="call_1",
    parts=(
        MediaPart(
            media_type="image/png",
            data=b"\x89PNG",
            description=Description(
                text="A bar chart of q3 revenue",
                author=Author.TOOL,
            ),
        ),
    ),
)
MEMORY = Observation(
    author=Author.SCAFFOLD,
    source="memory",
    parts=(TextPart(text="The user prefers charts."),),
)

READS_THE_FILE = SuspicionScore(
    value=0.25,
    explanation="It reads the file the task names.",
    evidence=Rationale(
        model="trusted-1",
        hidden_reasoning=("The file is the task's.",),
    ),
)
STORED_READS_THE_FILE: dict[str, JSON] = {
    "value": 0.25,
    "explanation": "It reads the file the task names.",
    "evidence": {
        "kind": "rationale",
        "model": "trusted-1",
        "hidden_reasoning": ["The file is the task's."],
    },
}
DECIDED = SuspicionScore(
    value=0.9,
    explanation="One statement is likely true.",
    evidence=Decisions(
        model="jev-1.14",
        decisions=(
            StatementDecision(
                statement="The action reads a file.",
                decision=Decision(
                    probability=0.9,
                    confidence=0.8,
                ),
            ),
            StatementDecision(
                statement="The action deletes a file.",
                decision=Decision(probability=0.1),
            ),
        ),
        hidden_reasoning=("Compared the two statements.",),
    ),
)
STORED_DECIDED: dict[str, JSON] = {
    "value": 0.9,
    "explanation": "One statement is likely true.",
    "evidence": {
        "kind": "decisions",
        "model": "jev-1.14",
        "decisions": [
            {
                "statement": "The action reads a file.",
                "decision": {
                    "probability": 0.9,
                    "confidence": 0.8,
                },
            },
            {
                "statement": "The action deletes a file.",
                "decision": {
                    "probability": 0.1,
                    "confidence": None,
                },
            },
        ],
        "hidden_reasoning": ["Compared the two statements."],
    },
}
# Two of the four replies that answered are labelled unsafe; one sample got no answer.
LABELLED = SuspicionScore(
    value=0.5,
    explanation="The guard model labelled the action 'unsafe', finding that it breaks the policy.",
    evidence=Labels(
        model="gpt-oss-safeguard-20b",
        labels=("unsafe", "unsafe", "safe", "safe"),
        unavailable=(Unavailable(status=429),),
        hidden_reasoning=("reasoning.encrypted",),
    ),
)
STORED_LABELLED: dict[str, JSON] = {
    "value": 0.5,
    "explanation": (
        "The guard model labelled the action 'unsafe', finding that it breaks the policy."
    ),
    "evidence": {
        "kind": "labels",
        "model": "gpt-oss-safeguard-20b",
        "labels": ["unsafe", "unsafe", "safe", "safe"],
        "unavailable": [
            {
                "kind": "unavailable",
                "status": 429,
            },
        ],
        "hidden_reasoning": ["reasoning.encrypted"],
    },
}
REFUSED = SuspicionScore(
    value=1.0,
    explanation="The monitor's model refused to rate the action.",
    evidence=FailedClosed(
        model="trusted-1",
        cause=Refused(),
    ),
)
STORED_REFUSED: dict[str, JSON] = {
    "value": 1.0,
    "explanation": "The monitor's model refused to rate the action.",
    "evidence": {
        "kind": "failed_closed",
        "model": "trusted-1",
        "cause": {
            "kind": "refused",
            "explanation": None,
        },
    },
}
COMBINED = SuspicionScore(
    value=1.0,
    explanation="A part failed.",
    evidence=Composite(
        parts=(
            READS_THE_FILE,
            REFUSED,
        ),
    ),
)
STORED_COMBINED: dict[str, JSON] = {
    "value": 1.0,
    "explanation": "A part failed.",
    "evidence": {
        "kind": "composite",
        "parts": [
            STORED_READS_THE_FILE,
            STORED_REFUSED,
        ],
    },
}

# Each cause a score can fail closed with, in its stored form.
CAUSES_STORED: list[tuple[Cause, dict[str, JSON]]] = [
    (
        Unreadable(),
        {"kind": "unreadable"},
    ),
    (
        Unperceivable(content=("audio", "image/tiff")),
        {
            "kind": "unperceivable",
            "content": ["audio", "image/tiff"],
        },
    ),
    (
        Truncated(text="The action is suspicious because"),
        {
            "kind": "truncated",
            "text": "The action is suspicious because",
        },
    ),
    (
        Refused(explanation="I cannot assess this."),
        {
            "kind": "refused",
            "explanation": "I cannot assess this.",
        },
    ),
    (
        Refused(),
        {
            "kind": "refused",
            "explanation": None,
        },
    ),
    (
        Unavailable(status=503),
        {
            "kind": "unavailable",
            "status": 503,
        },
    ),
    (
        Unavailable(),
        {
            "kind": "unavailable",
            "status": None,
        },
    ),
]
CAUSE_IDS = [
    "unreadable",
    "unperceivable",
    "truncated",
    "refused-with-an-explanation",
    "refused-without-one",
    "unavailable-with-a-status",
    "unavailable-without-one",
]

# The stored format, written out by hand: base64 of the bytes 89 50 4E 47 is "iVBORw==".
STORED: dict[str, JSON] = {
    "format": "hyoshigi.trajectory",
    "version": 1,
    "history": [
        {
            "kind": "observation",
            "author": "user",
            "source": None,
            "parts": [
                {
                    "kind": "text",
                    "text": "Plot q3.csv",
                },
            ],
        },
        {
            "kind": "action",
            "id": "a1",
            "thought": [
                {
                    "kind": "text",
                    "text": "I should read the file first.",
                },
                {
                    "kind": "opaque",
                    "opaque_kind": "redacted_thinking",
                },
            ],
            "parts": [
                {
                    "kind": "text",
                    "text": "Reading q3.csv.",
                },
            ],
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "read_file",
                    "arguments": '{"path": "q3.csv"}',
                },
            ],
        },
        {
            "kind": "tool_observation",
            "call_id": "call_1",
            "is_error": False,
            "source": None,
            "parts": [
                {
                    "kind": "media",
                    "media_type": "image/png",
                    "url": None,
                    "data": "iVBORw==",
                    "description": {
                        "text": "A bar chart of q3 revenue",
                        "author": "tool",
                    },
                },
            ],
        },
        {
            "kind": "observation",
            "author": "scaffold",
            "source": "memory",
            "parts": [
                {
                    "kind": "text",
                    "text": "The user prefers charts.",
                },
            ],
        },
    ],
    "ratings": [],
    "subagents": {},
}

DELEGATING: dict[str, JSON] = {
    "format": "hyoshigi.trajectory",
    "version": 1,
    "history": [
        {
            "kind": "action",
            "id": "a1",
            "thought": [],
            "parts": [],
            "tool_calls": [
                {
                    "id": "call_7",
                    "name": "task",
                    "arguments": "{}",
                },
            ],
        },
        {
            "kind": "tool_observation",
            "call_id": "call_7",
            "is_error": True,
            "source": "subagent",
            "parts": [
                {
                    "kind": "text",
                    "text": "The subagent stopped: no access to q3.csv.",
                },
            ],
        },
    ],
    "ratings": [
        {
            "action_id": "a1",
            "score": STORED_DECIDED,
        },
        {
            "action_id": "a1",
            "score": STORED_READS_THE_FILE,
        },
    ],
    "subagents": {
        "call_7": {
            "format": "hyoshigi.trajectory",
            "version": 1,
            "history": [
                {
                    "kind": "observation",
                    "author": "delegator",
                    "source": None,
                    "parts": [
                        {
                            "kind": "text",
                            "text": "Find the q3 revenue",
                        },
                    ],
                },
                {
                    "kind": "action",
                    "id": "b1",
                    "thought": [],
                    "parts": [
                        {
                            "kind": "media",
                            "media_type": "image/png",
                            "url": "https://example.com/q3.png",
                            "data": None,
                            "description": None,
                        },
                    ],
                    "tool_calls": [],
                },
            ],
            "ratings": [
                {
                    "action_id": "b1",
                    "score": STORED_COMBINED,
                },
            ],
            "subagents": {},
        },
    },
}

ENVELOPE: dict[str, JSON] = {
    "format": "hyoshigi.trajectory",
    "version": 1,
    "ratings": [],
    "subagents": {},
}
UNRATED_ENVELOPE: dict[str, JSON] = {
    "format": "hyoshigi.trajectory",
    "version": 1,
    "subagents": {},
}
ACTION: dict[str, JSON] = {
    "kind": "action",
    "id": "a1",
    "thought": [],
    "parts": [],
    "tool_calls": [],
}
OBSERVATION: dict[str, JSON] = {
    "kind": "observation",
    "author": "user",
    "source": None,
    "parts": [],
}
TOOL_OBSERVATION: dict[str, JSON] = {
    "kind": "tool_observation",
    "call_id": "call_1",
    "is_error": False,
    "source": None,
    "parts": [],
}
TEXT: dict[str, JSON] = {
    "kind": "text",
    "text": "Plot q3.csv",
}
MEDIA: dict[str, JSON] = {
    "kind": "media",
    "media_type": "image/png",
    "url": None,
    "data": "iVBORw==",
    "description": None,
}
AUDIO: dict[str, JSON] = {"kind": "audio"}
TASK_CALL: dict[str, JSON] = {
    "id": "call_7",
    "name": "task",
    "arguments": "{}",
}


@pytest.fixture
def trajectory() -> Trajectory:
    trajectory = Trajectory()
    trajectory.observe(TASK)
    trajectory.act(READ_FILE)
    trajectory.observe(CHART)
    trajectory.observe(MEMORY)
    return trajectory


def test_a_trajectory_dumps_to_the_stored_format(trajectory: Trajectory) -> None:
    stored = trajectory.dump()

    assert stored == STORED


def test_loading_the_stored_format_restores_the_history() -> None:
    loaded = Trajectory.load(STORED)

    assert loaded.history == (
        TASK,
        READ_FILE,
        CHART,
        MEMORY,
    )


def test_a_trajectory_with_a_subagent_loads_back_unchanged() -> None:
    loaded = Trajectory.load(DELEGATING)

    assert loaded.dump() == DELEGATING


@pytest.mark.parametrize(
    ("payload", "path"),
    [
        (
            [],
            "$",
        ),
        (
            {
                **ENVELOPE,
                "version": 2,
                "history": [],
            },
            "$",
        ),
        (
            {
                **ENVELOPE,
                "version": True,
                "history": [],
            },
            "$",
        ),
        (
            {
                **ENVELOPE,
                "history": [{"kind": "decision"}],
            },
            "$.history[0]",
        ),
        pytest.param(
            {
                **ENVELOPE,
                "history": [
                    ACTION,
                    {
                        **ACTION,
                        "parts": [TEXT],
                    },
                ],
            },
            "$.history[1]",
        ),
        pytest.param(
            {
                **ENVELOPE,
                "history": [
                    {
                        **ACTION,
                        "tool_calls": [TASK_CALL],
                    },
                    {
                        **ACTION,
                        "id": "a2",
                        "tool_calls": [
                            {
                                **TASK_CALL,
                                "name": "review",
                            },
                        ],
                    },
                ],
            },
            "$.history[1]",
        ),
        (
            {
                **ENVELOPE,
                "history": [
                    {
                        **ACTION,
                        "thought": [AUDIO],
                    },
                ],
            },
            "$.history[0].thought[0]",
        ),
        (
            {
                **ENVELOPE,
                "history": [
                    {
                        **ACTION,
                        "parts": [AUDIO],
                    },
                ],
            },
            "$.history[0].parts[0]",
        ),
        (
            {
                **ENVELOPE,
                "history": [
                    {
                        **ACTION,
                        "tool_calls": [{"id": "c1"}],
                    },
                ],
            },
            "$.history[0].tool_calls[0]",
        ),
        (
            {
                **ENVELOPE,
                "history": [
                    {
                        **OBSERVATION,
                        "author": "admin",
                    },
                ],
            },
            "$.history[0].author",
        ),
        (
            {
                **ENVELOPE,
                "history": [
                    {
                        **OBSERVATION,
                        "parts": [
                            TEXT,
                            AUDIO,
                        ],
                    },
                ],
            },
            "$.history[0].parts[1]",
        ),
        (
            {
                **ENVELOPE,
                "history": [
                    {
                        **TOOL_OBSERVATION,
                        "parts": [AUDIO],
                    },
                ],
            },
            "$.history[0].parts[0]",
        ),
        (
            {
                **ENVELOPE,
                "history": [
                    {
                        **OBSERVATION,
                        "parts": [
                            {
                                **MEDIA,
                                "data": "iVBO*Rw==",
                            },
                        ],
                    },
                ],
            },
            "$.history[0].parts[0].data",
        ),
        (
            {
                **ENVELOPE,
                "history": [
                    {
                        **OBSERVATION,
                        "parts": [
                            {
                                **MEDIA,
                                "media_type": "png",
                            },
                        ],
                    },
                ],
            },
            "$.history[0].parts[0]",
        ),
        (
            {
                **ENVELOPE,
                "history": [
                    {
                        **OBSERVATION,
                        "parts": [
                            {
                                **MEDIA,
                                "description": {"text": "A chart"},
                            },
                        ],
                    },
                ],
            },
            "$.history[0].parts[0].description",
        ),
        (
            {
                **ENVELOPE,
                "history": [
                    {
                        **OBSERVATION,
                        "parts": [
                            {
                                **MEDIA,
                                "description": {
                                    "text": "A chart",
                                    "author": "admin",
                                },
                            },
                        ],
                    },
                ],
            },
            "$.history[0].parts[0].description.author",
        ),
        (
            {
                **ENVELOPE,
                "history": [
                    {
                        **ACTION,
                        "tool_calls": [TASK_CALL],
                    },
                ],
                "subagents": {
                    "call_7": {
                        **ENVELOPE,
                        "history": [AUDIO],
                    },
                },
            },
            "$.subagents[0].history[0]",
        ),
        pytest.param(
            {
                **ENVELOPE,
                "history": [],
                "subagents": {
                    "call_7": {
                        **ENVELOPE,
                        "history": [],
                    },
                },
            },
            "$.subagents[0]",
        ),
    ],
    ids=[
        "not-a-mapping",
        "unknown-version",
        "boolean-version",
        "unknown-kind",
        "repeated-action-id",
        "repeated-tool-call-id",
        "malformed-thought",
        "malformed-action-part",
        "tool-call-without-arguments",
        "unknown-author",
        "malformed-observation-part",
        "malformed-tool-observation-part",
        "data-not-base64",
        "media-type-not-iana",
        "description-without-author",
        "description-with-unknown-author",
        "malformed-subagent",
        "subagent-without-its-call",
    ],
)
def test_loading_a_malformed_payload_raises_format_error_at_its_path(
    payload: JSON,
    path: str,
) -> None:
    with pytest.raises(FormatError) as raised:
        Trajectory.load(payload)

    assert raised.value.path == path


def test_a_format_error_names_its_path_in_its_message() -> None:
    with pytest.raises(FormatError) as raised:
        Trajectory.load(
            {
                **ENVELOPE,
                "history": [AUDIO],
            },
        )

    assert str(raised.value).startswith("$.history[0]: ")


def test_a_format_error_never_quotes_the_payload() -> None:
    payload: JSON = {
        **ENVELOPE,
        "history": [
            {
                **OBSERVATION,
                "author": "s3cret-0f3c",
            },
        ],
    }

    with pytest.raises(FormatError) as raised:
        Trajectory.load(payload)

    assert "s3cret-0f3c" not in str(raised.value)


def build_failed_closed_case(
    cause: Cause,
    stored_cause: dict[str, JSON],
) -> tuple[SuspicionScore, dict[str, JSON]]:
    evidence: dict[str, JSON] = {
        "kind": "failed_closed",
        "model": "trusted-1",
        "cause": stored_cause,
    }
    stored: dict[str, JSON] = {
        "value": 1.0,
        "explanation": "Failed.",
        "evidence": evidence,
    }
    score = SuspicionScore(
        value=1.0,
        explanation="Failed.",
        evidence=FailedClosed(
            model="trusted-1",
            cause=cause,
        ),
    )
    return score, stored


SCORE_CASES = [
    (
        READS_THE_FILE,
        STORED_READS_THE_FILE,
    ),
    (
        DECIDED,
        STORED_DECIDED,
    ),
    pytest.param(
        LABELLED,
        STORED_LABELLED,
    ),
    (
        COMBINED,
        STORED_COMBINED,
    ),
    *[
        build_failed_closed_case(
            cause,
            stored,
        )
        for cause, stored in CAUSES_STORED
    ],
]
SCORE_IDS = [
    "rationale",
    "decisions",
    "labels",
    "composite",
    *CAUSE_IDS,
]


@pytest.mark.parametrize(
    ("score", "stored"),
    SCORE_CASES,
    ids=SCORE_IDS,
)
def test_a_rating_dumps_to_the_stored_format(
    trajectory: Trajectory,
    score: SuspicionScore,
    stored: dict[str, JSON],
) -> None:
    trajectory.attach(
        score,
        to=READ_FILE,
    )

    dumped = trajectory.dump()

    assert dumped.get("ratings") == [
        {
            "action_id": "a1",
            "score": stored,
        },
    ]


@pytest.mark.parametrize(
    ("score", "stored"),
    SCORE_CASES,
    ids=SCORE_IDS,
)
def test_loading_a_stored_rating_restores_the_score(
    score: SuspicionScore,
    stored: dict[str, JSON],
) -> None:
    payload: dict[str, JSON] = {
        **ENVELOPE,
        "history": [ACTION],
        "ratings": [
            {
                "action_id": "a1",
                "score": stored,
            },
        ],
    }

    loaded = Trajectory.load(payload)

    assert loaded.ratings == (
        Rating(
            action_id="a1",
            score=score,
        ),
    )


def test_labels_stored_without_samples_that_got_no_answer_load_with_none() -> None:
    payload = build_evidence_payload(
        {
            "kind": "labels",
            "model": "gpt-oss-safeguard-20b",
            "labels": ["safe"],
            "hidden_reasoning": [],
        },
    )

    loaded = Trajectory.load(payload)

    assert loaded.ratings[0].score.evidence == Labels(
        model="gpt-oss-safeguard-20b",
        labels=("safe",),
        unavailable=(),
    )


EVIDENCE: dict[str, JSON] = {
    "kind": "rationale",
    "model": "trusted-1",
    "hidden_reasoning": [],
}
SCORE_PATH = "$.ratings[0].score"
EVIDENCE_PATH = "$.ratings[0].score.evidence"
CAUSE_PATH = "$.ratings[0].score.evidence.cause"


def build_rating_payload(rating: JSON) -> dict[str, JSON]:
    return {
        **ENVELOPE,
        "history": [ACTION],
        "ratings": [rating],
    }


def build_score_payload(**fields: JSON) -> dict[str, JSON]:
    return build_rating_payload(
        {
            "action_id": "a1",
            "score": {
                **STORED_READS_THE_FILE,
                **fields,
            },
        },
    )


def build_evidence_payload(evidence: JSON) -> dict[str, JSON]:
    return build_score_payload(evidence=evidence)


def build_decision_payload(decision: JSON) -> dict[str, JSON]:
    return build_evidence_payload(
        {
            "kind": "decisions",
            "model": "jev-1.14",
            "decisions": [
                {
                    "statement": "The action reads a file.",
                    "decision": decision,
                },
            ],
            "hidden_reasoning": [],
        },
    )


def build_cause_payload(cause: JSON) -> dict[str, JSON]:
    return build_score_payload(
        value=1.0,
        evidence={
            "kind": "failed_closed",
            "model": "trusted-1",
            "cause": cause,
        },
    )


@pytest.mark.parametrize(
    ("payload", "path"),
    [
        pytest.param(
            {
                **UNRATED_ENVELOPE,
                "history": [],
            },
            "$",
            id="ratings-missing",
        ),
        pytest.param(
            {
                **ENVELOPE,
                "history": [],
                "ratings": {},
            },
            "$",
            id="ratings-not-a-list",
        ),
        pytest.param(
            {
                **ENVELOPE,
                "history": [
                    {
                        **ACTION,
                        "tool_calls": [TASK_CALL],
                    },
                ],
                "subagents": {
                    "call_7": {
                        **UNRATED_ENVELOPE,
                        "history": [],
                    },
                },
            },
            "$.subagents[0]",
            id="subagent-ratings-missing",
        ),
        pytest.param(
            build_rating_payload("a1"),
            "$.ratings[0]",
            id="rating-not-a-mapping",
        ),
        pytest.param(
            build_rating_payload({"action_id": "a1"}),
            "$.ratings[0]",
            id="rating-no-score",
        ),
        pytest.param(
            build_rating_payload(
                {
                    "action_id": 1,
                    "score": STORED_READS_THE_FILE,
                },
            ),
            "$.ratings[0]",
            id="action-id-not-text",
        ),
        pytest.param(
            build_rating_payload(
                {
                    "action_id": "a9",
                    "score": STORED_READS_THE_FILE,
                },
            ),
            "$.ratings[0].action_id",
            id="action-not-in-the-history",
        ),
        pytest.param(
            build_rating_payload(
                {
                    "action_id": "a1",
                    "score": {
                        "value": 0.25,
                        "evidence": EVIDENCE,
                    },
                },
            ),
            SCORE_PATH,
            id="score-without-explanation",
        ),
        pytest.param(
            build_score_payload(explanation=5),
            SCORE_PATH,
            id="explanation-not-text",
        ),
        pytest.param(
            build_score_payload(value=True),
            SCORE_PATH,
            id="value-a-bool",
        ),
        pytest.param(
            build_score_payload(value=1.5),
            SCORE_PATH,
            id="value-above-one",
        ),
        pytest.param(
            build_score_payload(value="0.5"),
            SCORE_PATH,
            id="value-text",
        ),
        pytest.param(
            build_score_payload(value=None),
            SCORE_PATH,
            id="value-null",
        ),
        pytest.param(
            build_score_payload(
                value=0.5,
                evidence=STORED_REFUSED["evidence"],
            ),
            SCORE_PATH,
            id="failed-closed-with-a-value-below-one",
        ),
        pytest.param(
            build_evidence_payload({"kind": "audit"}),
            EVIDENCE_PATH,
            id="unknown-evidence",
        ),
        pytest.param(
            build_evidence_payload(
                {
                    "kind": "rationale",
                    "hidden_reasoning": [],
                },
            ),
            EVIDENCE_PATH,
            id="rationale-without-a-model",
        ),
        pytest.param(
            build_evidence_payload(
                {
                    **EVIDENCE,
                    "hidden_reasoning": "text",
                },
            ),
            EVIDENCE_PATH,
            id="hidden-reasoning-not-a-list",
        ),
        pytest.param(
            build_evidence_payload(
                {
                    **EVIDENCE,
                    "hidden_reasoning": [5],
                },
            ),
            "$.ratings[0].score.evidence.hidden_reasoning[0]",
            id="hidden-reasoning-not-text",
        ),
        pytest.param(
            build_evidence_payload(
                {
                    "kind": "decisions",
                    "model": "jev-1.14",
                    "decisions": [],
                    "hidden_reasoning": [5],
                },
            ),
            "$.ratings[0].score.evidence.hidden_reasoning[0]",
            id="decisions-hidden-reasoning-not-text",
        ),
        pytest.param(
            build_evidence_payload(
                {
                    "kind": "decisions",
                    "model": "jev-1.14",
                    "decisions": [{}],
                    "hidden_reasoning": [],
                },
            ),
            "$.ratings[0].score.evidence.decisions[0]",
            id="decision-without-a-statement",
        ),
        pytest.param(
            build_decision_payload(
                {
                    "probability": True,
                    "confidence": None,
                },
            ),
            "$.ratings[0].score.evidence.decisions[0].decision",
            id="probability-a-bool",
        ),
        pytest.param(
            build_decision_payload(
                {
                    "probability": "0.9",
                    "confidence": None,
                },
            ),
            "$.ratings[0].score.evidence.decisions[0].decision",
            id="probability-text",
        ),
        pytest.param(
            build_decision_payload(
                {
                    "probability": 1.5,
                    "confidence": None,
                },
            ),
            "$.ratings[0].score.evidence.decisions[0].decision",
            id="probability-above-one",
        ),
        pytest.param(
            build_decision_payload(
                {
                    "probability": math.nan,
                    "confidence": None,
                },
            ),
            "$.ratings[0].score.evidence.decisions[0].decision",
            id="probability-not-a-number",
        ),
        pytest.param(
            build_decision_payload(
                {
                    "probability": 0.9,
                    "confidence": "high",
                },
            ),
            "$.ratings[0].score.evidence.decisions[0].decision",
            id="confidence-text",
        ),
        pytest.param(
            build_decision_payload(
                {
                    "probability": 0.9,
                    "confidence": True,
                },
            ),
            "$.ratings[0].score.evidence.decisions[0].decision",
            id="confidence-a-bool",
        ),
        pytest.param(
            build_decision_payload(
                {
                    "probability": 0.9,
                    "confidence": -0.1,
                },
            ),
            "$.ratings[0].score.evidence.decisions[0].decision",
            id="confidence-below-zero",
        ),
        pytest.param(
            build_evidence_payload(
                {
                    "kind": "labels",
                    "labels": [],
                    "hidden_reasoning": [],
                },
            ),
            EVIDENCE_PATH,
            id="labels-without-a-model",
        ),
        pytest.param(
            build_evidence_payload(
                {
                    "kind": "labels",
                    "model": "m",
                    "labels": "unsafe",
                    "hidden_reasoning": [],
                },
            ),
            EVIDENCE_PATH,
            id="labels-not-a-list",
        ),
        pytest.param(
            build_evidence_payload(
                {
                    "kind": "labels",
                    "model": "m",
                    "labels": ["unsafe", 5],
                    "hidden_reasoning": [],
                },
            ),
            "$.ratings[0].score.evidence.labels[1]",
            id="label-not-text",
        ),
        pytest.param(
            build_evidence_payload(
                {
                    "kind": "labels",
                    "model": "m",
                    "labels": ["unsafe", None],
                    "hidden_reasoning": [],
                },
            ),
            "$.ratings[0].score.evidence.labels[1]",
            id="label-null",
        ),
        pytest.param(
            build_evidence_payload(
                {
                    "kind": "labels",
                    "model": "m",
                    "labels": [],
                    "hidden_reasoning": [5],
                },
            ),
            "$.ratings[0].score.evidence.hidden_reasoning[0]",
            id="labels-hidden-reasoning-not-text",
        ),
        pytest.param(
            build_evidence_payload(
                {
                    "kind": "labels",
                    "model": "m",
                    "labels": ["safe"],
                    "unavailable": {
                        "kind": "unavailable",
                        "status": 429,
                    },
                    "hidden_reasoning": [],
                },
            ),
            "$.ratings[0].score.evidence.unavailable",
            id="labels-unavailable-not-a-list",
        ),
        pytest.param(
            build_evidence_payload(
                {
                    "kind": "labels",
                    "model": "m",
                    "labels": ["safe"],
                    "unavailable": [
                        {
                            "kind": "refused",
                            "explanation": None,
                        },
                    ],
                    "hidden_reasoning": [],
                },
            ),
            "$.ratings[0].score.evidence.unavailable[0]",
            id="labels-unavailable-another-cause",
        ),
        pytest.param(
            build_evidence_payload(
                {
                    "kind": "labels",
                    "model": "m",
                    "labels": ["safe"],
                    "unavailable": [
                        {
                            "kind": "unavailable",
                            "status": "429",
                        },
                    ],
                    "hidden_reasoning": [],
                },
            ),
            "$.ratings[0].score.evidence.unavailable[0]",
            id="labels-unavailable-status-text",
        ),
        pytest.param(
            build_evidence_payload(
                {
                    "kind": "composite",
                    "parts": [],
                },
            ),
            EVIDENCE_PATH,
            id="composite-without-parts",
        ),
        pytest.param(
            build_evidence_payload(
                {
                    "kind": "composite",
                    "parts": [{"value": 0.5}],
                },
            ),
            "$.ratings[0].score.evidence.parts[0]",
            id="composite-part-malformed",
        ),
        pytest.param(
            build_evidence_payload(
                {
                    "kind": "composite",
                    "parts": [
                        {
                            **STORED_READS_THE_FILE,
                            "evidence": {"kind": "audit"},
                        },
                    ],
                },
            ),
            "$.ratings[0].score.evidence.parts[0].evidence",
            id="composite-part-with-unknown-evidence",
        ),
        pytest.param(
            build_score_payload(
                value=1.0,
                evidence={
                    "kind": "failed_closed",
                    "model": 5,
                    "cause": {"kind": "unreadable"},
                },
            ),
            EVIDENCE_PATH,
            id="failed-closed-model-not-text",
        ),
        pytest.param(
            build_cause_payload({"kind": "audit"}),
            CAUSE_PATH,
            id="unknown-cause",
        ),
        pytest.param(
            build_cause_payload(
                {
                    "kind": "unperceivable",
                    "content": [],
                },
            ),
            CAUSE_PATH,
            id="unperceivable-without-content",
        ),
        pytest.param(
            build_cause_payload(
                {
                    "kind": "unperceivable",
                    "content": [5],
                },
            ),
            "$.ratings[0].score.evidence.cause.content[0]",
            id="unperceivable-content-not-text",
        ),
        pytest.param(
            build_cause_payload({"kind": "truncated"}),
            CAUSE_PATH,
            id="truncated-without-text",
        ),
        pytest.param(
            build_cause_payload(
                {
                    "kind": "refused",
                    "explanation": 5,
                },
            ),
            CAUSE_PATH,
            id="refused-explanation-not-text",
        ),
        pytest.param(
            build_cause_payload(
                {
                    "kind": "unavailable",
                    "status": "503",
                },
            ),
            CAUSE_PATH,
            id="unavailable-status-text",
        ),
        pytest.param(
            build_cause_payload(
                {
                    "kind": "unavailable",
                    "status": True,
                },
            ),
            CAUSE_PATH,
            id="unavailable-status-a-bool",
        ),
    ],
)
def test_loading_a_malformed_rating_raises_format_error_at_its_path(
    payload: JSON,
    path: str,
) -> None:
    with pytest.raises(FormatError) as raised:
        Trajectory.load(payload)

    assert raised.value.path == path


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        pytest.param(
            build_evidence_payload(
                {
                    "kind": "labels",
                    "model": "m",
                    "labels": [5],
                    "hidden_reasoning": [],
                },
            ),
            "$.ratings[0].score.evidence.labels[0]: expected text",
            id="label-not-text",
        ),
        pytest.param(
            build_evidence_payload(
                {
                    "kind": "labels",
                    "model": "m",
                    "labels": ["safe"],
                    "unavailable": None,
                    "hidden_reasoning": [],
                },
            ),
            "$.ratings[0].score.evidence.unavailable: expected a list of unavailable causes",
            id="unavailable-not-a-list",
        ),
        pytest.param(
            build_evidence_payload(
                {
                    "kind": "labels",
                    "model": "m",
                    "labels": ["safe"],
                    "unavailable": [{"kind": "unreadable"}],
                    "hidden_reasoning": [],
                },
            ),
            "$.ratings[0].score.evidence.unavailable[0]: expected an unavailable cause",
            id="unavailable-another-cause",
        ),
        pytest.param(
            build_evidence_payload({"kind": "audit"}),
            "$.ratings[0].score.evidence: expected rationale, decisions, labels, composite or "
            "failed-closed evidence",
            id="unknown-evidence",
        ),
    ],
)
def test_a_format_error_for_evidence_says_what_was_expected(
    payload: JSON,
    message: str,
) -> None:
    with pytest.raises(FormatError) as raised:
        Trajectory.load(payload)

    assert str(raised.value) == message


def test_a_format_error_for_a_rating_never_quotes_the_payload() -> None:
    payload = build_score_payload(value="s3cret-0f3c")

    with pytest.raises(FormatError) as raised:
        Trajectory.load(payload)

    assert "s3cret-0f3c" not in str(raised.value)


def nest_composites(depth: int) -> dict[str, JSON]:
    """A stored trajectory whose one rating nests composite scores that many levels deep."""
    score: JSON = STORED_READS_THE_FILE
    for _ in range(depth):
        score = {
            "value": 0.25,
            "explanation": "The mean of its part.",
            "evidence": {
                "kind": "composite",
                "parts": [score],
            },
        }
    return build_rating_payload(
        {
            "action_id": "a1",
            "score": score,
        },
    )


def nest_subagents(depth: int) -> dict[str, JSON]:
    """A stored trajectory whose subagents nest that many levels deep."""
    trajectory: dict[str, JSON] = {
        **ENVELOPE,
        "history": [],
    }
    for _ in range(depth):
        trajectory = {
            **ENVELOPE,
            "history": [
                {
                    **ACTION,
                    "tool_calls": [TASK_CALL],
                },
            ],
            "subagents": {"call_7": trajectory},
        }
    return trajectory


@pytest.mark.parametrize(
    "payload",
    [
        nest_composites(32),
        nest_subagents(32),
    ],
    ids=["composite-scores", "subagents"],
)
def test_nesting_of_32_levels_loads_back_unchanged(payload: dict[str, JSON]) -> None:
    loaded = Trajectory.load(payload)

    assert loaded.dump() == payload


# The deepest composite is the evidence of the score inside 32 others, so its path opens at the
# rating's score, holds 32 composite parts and ends at an evidence; the deepest subagent is the
# 33rd, so its path holds 33 subagents.
@pytest.mark.parametrize(
    ("payload", "opening", "level", "levels", "ending"),
    [
        (
            nest_composites(33),
            "$.ratings[0].score",
            ".evidence.parts[0]",
            32,
            ".evidence",
        ),
        (
            nest_composites(400),
            "$.ratings[0].score",
            ".evidence.parts[0]",
            32,
            ".evidence",
        ),
        (
            nest_subagents(33),
            "$.subagents[0]",
            ".subagents[0]",
            33,
            ".subagents[0]",
        ),
        (
            nest_subagents(600),
            "$.subagents[0]",
            ".subagents[0]",
            33,
            ".subagents[0]",
        ),
    ],
    ids=["composite-scores-33", "composite-scores-400", "subagents-33", "subagents-600"],
)
def test_nesting_past_32_levels_raises_format_error_at_the_33rd(
    payload: dict[str, JSON],
    opening: str,
    level: str,
    levels: int,
    ending: str,
) -> None:
    with pytest.raises(FormatError) as raised:
        Trajectory.load(payload)

    assert (
        raised.value.path.startswith(opening),
        raised.value.path.count(level),
        raised.value.path.endswith(ending),
    ) == (
        True,
        levels,
        True,
    )
