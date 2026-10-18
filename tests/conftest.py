"""The OpenTelemetry tracer provider that records the spans of a test run."""

import itertools
from dataclasses import dataclass
from typing import cast

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.id_generator import IdGenerator
from opentelemetry.sdk.trace.sampling import ALWAYS_ON, ParentBased

# A trace id and a first span id with leading zeros and hex letters, so an id written without
# its padding or in upper case differs from the literal a test expects.
TRACE_ID = 0x00ABCDEF0123456789ABCDEF01234567
FIRST_SPAN_ID = 0x00F1E2D3C4B5A697


class CountedIds(IdGenerator):
    """Gives every trace TRACE_ID, and spans FIRST_SPAN_ID and the numbers after it in the order
    they start, counted again from FIRST_SPAN_ID after each restart.
    """

    def __init__(self) -> None:
        self.span_ids = itertools.count(FIRST_SPAN_ID)

    def restart(self) -> None:
        self.span_ids = itertools.count(FIRST_SPAN_ID)

    def generate_span_id(self) -> int:
        return next(self.span_ids)

    def generate_trace_id(self) -> int:
        return TRACE_ID


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class Tracing:
    """The global tracer provider's ids and the exporter that keeps its finished spans."""

    ids: CountedIds
    exporter: InMemorySpanExporter


@pytest.fixture(scope="session")
def tracing() -> Tracing:
    # OpenTelemetry sets the global tracer provider once per process: a second
    # set_tracer_provider is ignored with a warning (opentelemetry/trace/__init__.py,
    # _set_tracer_provider), and hyoshigi takes its tracer from the global provider. So the
    # first test run in a process installs one provider, a later run in the same process (as
    # mutmut makes) records with it too, and each test clears its spans and restarts its ids.
    installed = trace.get_tracer_provider()
    if isinstance(
        installed,
        TracerProvider,
    ):
        provider = installed
    else:
        provider = TracerProvider(
            sampler=ParentBased(ALWAYS_ON),
            id_generator=CountedIds(),
        )
        trace.set_tracer_provider(provider)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return Tracing(
        # The provider's own generator, which a later run's module did not define.
        ids=cast(
            CountedIds,
            provider.id_generator,
        ),
        exporter=exporter,
    )


@pytest.fixture
def spans(tracing: Tracing) -> InMemorySpanExporter:
    """The exporter of the global tracer provider, with no spans, its ids counted afresh."""
    tracing.exporter.clear()
    tracing.ids.restart()
    return tracing.exporter
