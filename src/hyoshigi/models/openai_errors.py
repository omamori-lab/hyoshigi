"""The outcome of an HTTP status and an error code as OpenAI's API answers them, which the OpenAI
and LiteLLM models share.

The module imports no SDK, so a model on another SDK, or on an older version of OpenAI's, reads
errors by the same rules.
"""

from hyoshigi.errors import ConfigurationError
from hyoshigi.models import responses
from hyoshigi.models.model import Refused, Rejected, Unavailable

# Statuses that are no answer rather than a mistake: request timeout, conflict and too many
# requests.
_UNAVAILABLE_STATUSES = frozenset({408, 409, 429})
# The error codes that blame what the prompt holds, which an agent can shape, not the request:
# an input beyond the context window (https://github.com/openai/openai-openapi, openapi.yaml,
# schema SessionTurnErrorCodeResource), an invalid prompt and an image that cannot be used (the
# same file, schema ResponseErrorCode, less image_content_policy_violation, a policy refusal,
# and image_file_not_found, which no request built here can cause).
UNUSABLE_INPUT_CODES = frozenset(
    {
        "context_length_exceeded",
        "invalid_prompt",
        "invalid_image",
        "invalid_image_format",
        "invalid_base64_image",
        "invalid_image_url",
        "image_too_large",
        "image_too_small",
        "image_parse_error",
        "invalid_image_mode",
        "image_file_too_large",
        "unsupported_image_media_type",
        "empty_image_file",
        "failed_to_download_image",
    },
)


def is_rejection(
    status: int,
    *,
    code: object,
) -> bool:
    """Whether a status and its error code say the prompt cannot be used: a 413, or a code that
    blames what the prompt holds.
    """
    return status == 413 or code in UNUSABLE_INPUT_CODES


def read_status(
    status: int,
    *,
    code: object,
    provider: str,
) -> Refused | Rejected | Unavailable:
    """The outcome of an error with this HTTP status and error code.

    A code that is a refusal or names a policy is a refusal. A 413, or a code that says the prompt
    cannot be used, is rejected with the status. Any other 4xx status but 408, 409 and 429 raises
    ConfigurationError naming the provider and only the status. Any other status is unavailable
    with it.

    <https://omamori-lab.github.io/hyoshigi/reference/providers/>
    """
    match code:
        case str() if responses.names_refusal(code):
            return Refused()
    if is_rejection(
        status,
        code=code,
    ):
        return Rejected(status=status)
    if 400 <= status < 500 and status not in _UNAVAILABLE_STATUSES:
        raise ConfigurationError(f"{provider} rejected the request with HTTP status {status}")
    return Unavailable(status=status)
