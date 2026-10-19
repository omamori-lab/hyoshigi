"""A payload with one node replaced is read, or raises a FormatError that quotes none of it."""

import json
from collections.abc import Callable

import pytest
from damage import draw_edit, reads_cleanly, replace_each_node
from hypothesis import given
from hypothesis import strategies as st

from hyoshigi import JSON, Action, Trajectory

# Every free value in a payload below holds the marker; only the schema tokens do not.

ANTHROPIC_REQUEST: JSON = json.loads("""{
  "system": [{"type": "text", "text": "SECRET system"}],
  "messages": [
    {"role": "user", "content": "SECRET task"},
    {"role": "assistant", "content": [
      {"type": "thinking", "thinking": "SECRET thinking"},
      {"type": "redacted_thinking"},
      {"type": "text", "text": "SECRET said"},
      {"type": "tool_use", "id": "SECRET_toolu_1", "name": "SECRET_read",
       "input": {"path": "SECRET path"}},
      {"type": "server_tool_use", "id": "SECRET_srvtoolu_1", "name": "SECRET_search",
       "input": {"q": "SECRET"}},
      {"type": "web_search_tool_result", "tool_use_id": "SECRET_srvtoolu_1", "content": [
        {"type": "web_search_result", "url": "https://example.com/SECRET"}
      ]},
      {"type": "text", "text": "SECRET more"}
    ]},
    {"role": "user", "content": [
      {"type": "text", "text": "SECRET note"},
      {"type": "image", "source": {
        "type": "base64", "media_type": "image/png", "data": "SECRETSECRET"}},
      {"type": "image", "source": {"type": "url", "url": "https://example.com/SECRET.png"}},
      {"type": "document", "source": {
        "type": "base64", "media_type": "application/pdf", "data": "SECRETSECRET"}},
      {"type": "document", "source": {"type": "url", "url": "https://example.com/SECRET.pdf"}},
      {"type": "document", "source": {"type": "text", "data": "SECRET document"}},
      {"type": "tool_result", "tool_use_id": "SECRET_toolu_1", "content": "SECRET result"},
      {"type": "tool_result", "tool_use_id": "SECRET_toolu_1", "is_error": true, "content": [
        {"type": "text", "text": "SECRET error"}
      ]}
    ]}
  ]
}""")
ANTHROPIC_RESPONSE: JSON = json.loads("""{
  "id": "SECRET_msg_1",
  "role": "assistant",
  "content": [
    {"type": "thinking", "thinking": "SECRET thinking"},
    {"type": "text", "text": "SECRET said"},
    {"type": "tool_use", "id": "SECRET_toolu_1", "name": "SECRET_read",
     "input": {"path": "SECRET path"}}
  ]
}""")
OPENAI_CHAT_ASSISTANT: JSON = json.loads("""{
  "role": "assistant",
  "reasoning": "SECRET reasoning",
  "reasoning_details": [
    {"type": "reasoning.text", "text": "SECRET thinking"},
    {"type": "reasoning.summary", "summary": "SECRET summary"},
    {"type": "reasoning.encrypted", "data": "SECRET abc"}
  ],
  "content": [
    {"type": "text", "text": "SECRET said"},
    {"type": "refusal", "refusal": "SECRET refusal"},
    {"type": "image_url"}
  ],
  "refusal": "SECRET refused",
  "audio": {"id": "SECRET_audio_1"},
  "tool_calls": [
    {"id": "SECRET_call_1", "type": "function",
     "function": {"name": "SECRET_read", "arguments": "SECRET"}},
    {"id": "SECRET_call_2", "type": "custom",
     "custom": {"name": "SECRET_patch", "input": "SECRET"}},
    {"id": "SECRET_call_3", "type": "other"}
  ],
  "function_call": {"name": "SECRET_read", "arguments": "SECRET"}
}""")
OPENAI_CHAT_REQUEST: JSON = {
    "messages": [
        *json.loads("""[
          {"role": "system", "content": "SECRET system"},
          {"role": "developer", "content": [{"type": "text", "text": "SECRET developer"}]},
          {"role": "user", "content": "SECRET task"},
          {"role": "user", "content": [
            {"type": "text", "text": "SECRET note"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,SECRETSECRET"}},
            {"type": "image_url", "image_url": {"url": "https://example.com/SECRET.png"}},
            {"type": "input_audio", "input_audio": {"data": "SECRETSECRET", "format": "wav"}},
            {"type": "file", "file": {"file_data": "data:application/pdf;base64,SECRETSECRET"}},
            {"type": "file", "file": {"file_id": "SECRET_file_1"}},
            {"type": "video"}
          ]}
        ]"""),
        OPENAI_CHAT_ASSISTANT,
        *json.loads("""[
          {"role": "tool", "tool_call_id": "SECRET_call_1", "content": "SECRET result"},
          {"role": "function", "name": "SECRET_read", "content": "SECRET result"}
        ]"""),
    ],
}
OPEN_RESPONSES_OUTPUT: list[JSON] = json.loads("""[
  {"type": "reasoning", "encrypted_content": "SECRET abc",
   "summary": [{"type": "summary_text", "text": "SECRET summary"}],
   "content": [{"type": "reasoning_text", "text": "SECRET thinking"}]},
  {"type": "message", "role": "assistant", "content": [
    {"type": "output_text", "text": "SECRET said"},
    {"type": "refusal", "refusal": "SECRET refusal"}
  ]},
  {"type": "function_call", "call_id": "SECRET_call_1", "name": "SECRET_read",
   "arguments": "SECRET"},
  {"type": "web_search_call", "id": "SECRET_ws_1"}
]""")
OPEN_RESPONSES_REQUEST: JSON = {
    "instructions": "SECRET instructions",
    "previous_response_id": "SECRET_resp_1",
    "input": [
        *json.loads("""[
          {"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "SECRET task"},
            {"type": "input_image", "image_url": "https://example.com/SECRET.png"},
            {"type": "input_image", "image_url": "data:image/png;base64,SECRETSECRET"},
            {"type": "input_video", "video_url": "https://example.com/SECRET.mp4"},
            {"type": "input_file", "file_data": "data:application/pdf;base64,SECRETSECRET"},
            {"type": "input_file", "file_id": "SECRET_file_1"},
            {"type": "input_audio"}
          ]},
          {"role": "developer", "content": "SECRET developer"}
        ]"""),
        *OPEN_RESPONSES_OUTPUT,
        *json.loads("""[
          {"type": "function_call_output", "call_id": "SECRET_call_1", "output": "SECRET result"},
          {"type": "function_call_output", "call_id": "SECRET_call_1", "output": [
            {"type": "input_text", "text": "SECRET result"}
          ]},
          {"type": "compaction"},
          {"id": "SECRET_item_1"}
        ]"""),
    ],
}


READERS = pytest.mark.parametrize(
    ("read", "payload"),
    [
        pytest.param(
            Trajectory.from_anthropic,
            ANTHROPIC_REQUEST,
            id="Trajectory.from_anthropic",
        ),
        pytest.param(
            Action.from_anthropic,
            ANTHROPIC_RESPONSE,
            id="Action.from_anthropic",
        ),
        pytest.param(
            Trajectory.from_openai_chat,
            OPENAI_CHAT_REQUEST,
            id="Trajectory.from_openai_chat",
        ),
        pytest.param(
            Action.from_openai_chat,
            OPENAI_CHAT_ASSISTANT,
            id="Action.from_openai_chat",
        ),
        pytest.param(
            Trajectory.from_open_responses,
            OPEN_RESPONSES_REQUEST,
            id="Trajectory.from_open_responses",
        ),
        pytest.param(
            Action.from_open_responses,
            OPEN_RESPONSES_OUTPUT,
            id="Action.from_open_responses",
        ),
    ],
)


@READERS
@given(data=st.data())
def test_a_payload_with_one_node_replaced_is_read_or_raises_a_format_error_that_quotes_none_of_it(
    read: Callable[[JSON], object],
    payload: JSON,
    data: st.DataObject,
) -> None:
    edit = data.draw(draw_edit(payload))

    unclean = [
        damaged
        for damaged in replace_each_node(
            payload,
            edit,
        )
        if not reads_cleanly(
            read,
            damaged,
        )
    ]

    assert unclean == []
