"""How a stored trajectory is read back from JSON-compatible data.

The reader builds a Trajectory's state, its entries, ratings, records with their sequence numbers
and subagents, through the trajectory's private attributes. It is the one module outside the
trajectory's own that reaches them.
"""

from hyoshigi.errors import FormatError
from hyoshigi.trajectory.json_data import JSON
from hyoshigi.trajectory.storage.events import load_event
from hyoshigi.trajectory.storage.paths import select_member
from hyoshigi.trajectory.storage.records import load_records, read_sequence_numbers
from hyoshigi.trajectory.storage.scores import NESTING_LIMIT, load_score
from hyoshigi.trajectory.trajectory import FORMAT, VERSION, Delegation, Rating, Trajectory


def load_trajectory[Kind: Trajectory](
    data: JSON,
    *,
    kind: type[Kind],
    path: str,
    depth: int,
    seen: set[int],
) -> Kind:
    """Read the stored trajectory at the path, `depth` levels below the root, as a `kind`, and
    add its records' sequence numbers to `seen`.

    Raises FormatError, naming the JSONPath of the first problem, on data that is not what
    Trajectory.dump writes.
    """
    match data:
        case {
            "format": "hyoshigi.trajectory",
            "version": int(version),
            "history": list(history),
            "ratings": list(ratings),
            "records": list(records),
            "subagents": dict(subagents),
        } if version == VERSION and not isinstance(
            version,
            bool,
        ):
            trajectory = kind()
            _load_history(
                history,
                path=f"{path}.history",
                trajectory=trajectory,
            )
            _load_ratings(
                ratings,
                path=f"{path}.ratings",
                trajectory=trajectory,
            )
            trajectory._records = load_records(
                records,
                path=f"{path}.records",
                history=trajectory._history,
            )
            trajectory._sequence_numbers = read_sequence_numbers(
                records,
                path=f"{path}.records",
                seen=seen,
            )
            _load_subagents(
                subagents,
                path=f"{path}.subagents",
                depth=depth,
                seen=seen,
                trajectory=trajectory,
            )
            return trajectory
        case _:
            raise FormatError(
                path=path,
                expected=f"a {FORMAT}, format version {VERSION}",
            )


def _load_history(
    history: list[JSON],
    *,
    path: str,
    trajectory: Trajectory,
) -> None:
    for index, data in enumerate(history):
        event_path = f"{path}[{index}]"
        trajectory._add_event(
            load_event(
                data,
                path=event_path,
            ),
            path=event_path,
        )


def _load_ratings(
    ratings: list[JSON],
    *,
    path: str,
    trajectory: Trajectory,
) -> None:
    for index, data in enumerate(ratings):
        rating_path = f"{path}[{index}]"
        match data:
            case {
                "action_id": str(action_id),
                "score": dict() as score,
            }:
                if action_id not in trajectory._action_ids:
                    raise FormatError(
                        path=f"{rating_path}.action_id",
                        expected="the id of an action in the history",
                    )
                loaded = load_score(
                    score,
                    path=f"{rating_path}.score",
                    depth=0,
                )
                trajectory._ratings.append(
                    Rating(
                        action_id=action_id,
                        score=loaded,
                    ),
                )
            case _:
                raise FormatError(
                    path=rating_path,
                    expected="a rating with an action id and a score",
                )


def _load_subagents(
    subagents: dict[str, JSON],
    *,
    path: str,
    depth: int,
    seen: set[int],
    trajectory: Trajectory,
) -> None:
    """Load the subagents of a trajectory `depth` levels deep, adding their records'
    sequence numbers to `seen`. A subagent deeper than NESTING_LIMIT raises FormatError.
    """
    for call_id, data in subagents.items():
        match call_id:
            case str():
                subagent_path = select_member(
                    call_id,
                    path=path,
                )
            case _:
                raise FormatError(
                    path=path,
                    expected="subagents keyed by tool call id",
                )
        if call_id not in trajectory._call_ids:
            raise FormatError(
                path=subagent_path,
                expected="a subagent under a tool call in the history",
            )
        if depth == NESTING_LIMIT:
            raise FormatError(
                path=subagent_path,
                expected=f"subagents nested at most {NESTING_LIMIT} deep",
            )
        subagent = load_trajectory(
            data,
            kind=Trajectory,
            path=subagent_path,
            depth=depth + 1,
            seen=seen,
        )
        subagent._delegation = Delegation(
            parent=trajectory,
            call_id=call_id,
        )
        trajectory._subagents[call_id] = subagent
