"""Tests for AIGenerator."""

from unittest.mock import patch, MagicMock
import pytest

from ai_generator import AIGenerator


# ---------------------------------------------------------------------------
# Helpers for building mock Anthropic responses
# ---------------------------------------------------------------------------

def make_text_block(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def make_tool_use_block(name: str, tool_id: str, inputs: dict):
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.id = tool_id
    block.input = inputs
    return block


def make_response(stop_reason: str, content_blocks: list):
    resp = MagicMock()
    resp.stop_reason = stop_reason
    resp.content = content_blocks
    return resp


@pytest.fixture
def generator():
    """Create an AIGenerator with a mocked Anthropic client."""
    with patch("ai_generator.anthropic.Anthropic") as MockClient:
        gen = AIGenerator(api_key="fake-key", model="claude-test")
        # Expose the mock so tests can configure messages.create
        gen._mock_client = MockClient.return_value
        yield gen


# ===================================================================
# Direct (no-tool) responses
# ===================================================================

class TestDirectResponse:

    def test_direct_answer_no_tools(self, generator):
        generator._mock_client.messages.create.return_value = make_response(
            "end_turn", [make_text_block("Hello!")]
        )
        result = generator.generate_response(query="Hi")
        assert result == "Hello!"

    def test_system_prompt_included(self, generator):
        generator._mock_client.messages.create.return_value = make_response(
            "end_turn", [make_text_block("ok")]
        )
        generator.generate_response(query="Hi")
        call_kwargs = generator._mock_client.messages.create.call_args[1]
        assert "system" in call_kwargs
        assert AIGenerator.SYSTEM_PROMPT in call_kwargs["system"]

    def test_conversation_history_appended(self, generator):
        generator._mock_client.messages.create.return_value = make_response(
            "end_turn", [make_text_block("ok")]
        )
        generator.generate_response(query="Hi", conversation_history="prev msg")
        call_kwargs = generator._mock_client.messages.create.call_args[1]
        assert "Previous conversation:" in call_kwargs["system"]
        assert "prev msg" in call_kwargs["system"]

    def test_no_history_clean_system(self, generator):
        generator._mock_client.messages.create.return_value = make_response(
            "end_turn", [make_text_block("ok")]
        )
        generator.generate_response(query="Hi")
        call_kwargs = generator._mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == AIGenerator.SYSTEM_PROMPT

    def test_tools_passed_to_api(self, generator):
        generator._mock_client.messages.create.return_value = make_response(
            "end_turn", [make_text_block("ok")]
        )
        tools = [{"name": "t1"}]
        generator.generate_response(query="Hi", tools=tools)
        call_kwargs = generator._mock_client.messages.create.call_args[1]
        assert call_kwargs["tools"] == tools
        assert call_kwargs["tool_choice"] == {"type": "auto"}

    def test_no_tools_excluded_from_api(self, generator):
        generator._mock_client.messages.create.return_value = make_response(
            "end_turn", [make_text_block("ok")]
        )
        generator.generate_response(query="Hi")
        call_kwargs = generator._mock_client.messages.create.call_args[1]
        assert "tools" not in call_kwargs
        assert "tool_choice" not in call_kwargs


# ===================================================================
# Tool-use flow
# ===================================================================

class TestToolUseFlow:

    def test_tool_use_flow_end_to_end(self, generator):
        tool_block = make_tool_use_block("search_course_content", "call_1", {"query": "q"})
        first_resp = make_response("tool_use", [tool_block])
        final_resp = make_response("end_turn", [make_text_block("Final answer")])

        generator._mock_client.messages.create.side_effect = [first_resp, final_resp]

        mgr = MagicMock()
        mgr.execute_tool.return_value = "tool output"

        result = generator.generate_response(
            query="question", tools=[{"name": "search_course_content"}], tool_manager=mgr
        )
        assert result == "Final answer"
        mgr.execute_tool.assert_called_once_with("search_course_content", query="q")

    def test_tool_use_final_call_excludes_tools(self, generator):
        tool_block = make_tool_use_block("t", "id1", {})
        first_resp = make_response("tool_use", [tool_block])
        final_resp = make_response("end_turn", [make_text_block("done")])

        generator._mock_client.messages.create.side_effect = [first_resp, final_resp]

        mgr = MagicMock()
        mgr.execute_tool.return_value = "out"

        generator.generate_response(query="q", tools=[{"name": "t"}], tool_manager=mgr)

        # Second call should NOT contain tools
        second_call_kwargs = generator._mock_client.messages.create.call_args_list[1][1]
        assert "tools" not in second_call_kwargs
        assert "tool_choice" not in second_call_kwargs

    def test_tool_use_message_structure(self, generator):
        tool_block = make_tool_use_block("t", "id1", {"a": 1})
        first_resp = make_response("tool_use", [tool_block])
        final_resp = make_response("end_turn", [make_text_block("done")])

        generator._mock_client.messages.create.side_effect = [first_resp, final_resp]

        mgr = MagicMock()
        mgr.execute_tool.return_value = "result"

        generator.generate_response(query="q", tools=[{"name": "t"}], tool_manager=mgr)

        second_call_kwargs = generator._mock_client.messages.create.call_args_list[1][1]
        messages = second_call_kwargs["messages"]

        # [user, assistant+tool_use, user+tool_result]
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        # Tool result content
        assert messages[2]["content"][0]["type"] == "tool_result"
        assert messages[2]["content"][0]["tool_use_id"] == "id1"

    def test_tool_use_without_manager_returns_text(self, generator):
        tool_block = make_tool_use_block("t", "id1", {})
        text_block = make_text_block("fallback text")
        resp = make_response("tool_use", [text_block, tool_block])

        generator._mock_client.messages.create.return_value = resp

        # tool_manager=None → should fall through to content[0].text
        result = generator.generate_response(query="q", tools=[{"name": "t"}], tool_manager=None)
        assert result == "fallback text"
