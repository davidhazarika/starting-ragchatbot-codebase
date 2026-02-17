"""Tests for RAGSystem.query()."""

from unittest.mock import patch, MagicMock, call
import pytest


# Patch heavy dependencies at module level so importing rag_system never
# instantiates VectorStore, DocumentProcessor, etc.
@pytest.fixture(autouse=True)
def _patch_deps():
    with patch("rag_system.VectorStore") as MockVS, \
         patch("rag_system.AIGenerator") as MockAI, \
         patch("rag_system.SessionManager") as MockSM, \
         patch("rag_system.DocumentProcessor") as MockDP:
        # Store on the module so individual tests can access them
        _patch_deps.MockVS = MockVS
        _patch_deps.MockAI = MockAI
        _patch_deps.MockSM = MockSM
        _patch_deps.MockDP = MockDP
        yield


def _make_rag():
    """Create a RAGSystem with fake config."""
    from rag_system import RAGSystem

    config = MagicMock()
    config.CHUNK_SIZE = 500
    config.CHUNK_OVERLAP = 50
    config.CHROMA_PATH = "/tmp/chroma"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.MAX_RESULTS = 5
    config.ANTHROPIC_API_KEY = "fake"
    config.ANTHROPIC_MODEL = "claude-test"
    config.MAX_HISTORY = 10

    rag = RAGSystem(config)
    return rag


# ===================================================================
# RAGSystem.query() tests
# ===================================================================

class TestRAGQuery:

    def test_query_wraps_prompt(self):
        rag = _make_rag()
        rag.ai_generator.generate_response.return_value = "resp"
        rag.query("What is Python?")
        call_kwargs = rag.ai_generator.generate_response.call_args[1]
        assert call_kwargs["query"] == "Answer this question about course materials: What is Python?"

    def test_query_passes_tools_and_manager(self):
        rag = _make_rag()
        rag.ai_generator.generate_response.return_value = "resp"
        rag.query("q")
        call_kwargs = rag.ai_generator.generate_response.call_args[1]
        assert "tools" in call_kwargs
        assert "tool_manager" in call_kwargs
        assert call_kwargs["tool_manager"] is rag.tool_manager

    def test_query_returns_response_and_sources(self):
        rag = _make_rag()
        rag.ai_generator.generate_response.return_value = "answer"
        rag.tool_manager.get_last_sources = MagicMock(return_value=[{"text": "s"}])
        rag.tool_manager.reset_sources = MagicMock()
        response, sources = rag.query("q")
        assert response == "answer"
        assert sources == [{"text": "s"}]

    def test_query_resets_sources(self):
        rag = _make_rag()
        rag.ai_generator.generate_response.return_value = "resp"
        rag.tool_manager.reset_sources = MagicMock()
        rag.query("q")
        rag.tool_manager.reset_sources.assert_called_once()

    def test_query_with_session_gets_history(self):
        rag = _make_rag()
        rag.ai_generator.generate_response.return_value = "resp"
        rag.session_manager.get_conversation_history.return_value = "hist"
        rag.query("q", session_id="s1")
        call_kwargs = rag.ai_generator.generate_response.call_args[1]
        assert call_kwargs["conversation_history"] == "hist"

    def test_query_without_session_no_history(self):
        rag = _make_rag()
        rag.ai_generator.generate_response.return_value = "resp"
        rag.query("q")
        call_kwargs = rag.ai_generator.generate_response.call_args[1]
        assert call_kwargs["conversation_history"] is None

    def test_query_updates_session_history(self):
        rag = _make_rag()
        rag.ai_generator.generate_response.return_value = "the answer"
        rag.query("the question", session_id="s1")
        rag.session_manager.add_exchange.assert_called_once_with("s1", "the question", "the answer")

    def test_query_no_session_skips_update(self):
        rag = _make_rag()
        rag.ai_generator.generate_response.return_value = "resp"
        rag.query("q")
        rag.session_manager.add_exchange.assert_not_called()

    def test_init_registers_search_tool(self):
        rag = _make_rag()
        # The tool_manager should have the search tool registered
        defs = rag.tool_manager.get_tool_definitions()
        names = [d["name"] for d in defs]
        assert "search_course_content" in names
