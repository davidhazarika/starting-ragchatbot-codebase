"""Tests for CourseSearchTool and ToolManager."""

from unittest.mock import MagicMock
import pytest

from vector_store import SearchResults
from search_tools import CourseSearchTool, ToolManager


# ===================================================================
# CourseSearchTool.execute()
# ===================================================================

class TestCourseSearchToolExecute:

    def _make_tool(self, search_return):
        store = MagicMock()
        store.search.return_value = search_return
        return CourseSearchTool(store), store

    # -- error path --------------------------------------------------

    def test_execute_returns_error_when_search_errors(self, mock_error_results):
        tool, _ = self._make_tool(mock_error_results)
        result = tool.execute(query="anything")
        assert result == "Search error: connection failed"

    # -- empty-result paths ------------------------------------------

    def test_execute_returns_empty_message_no_filters(self, mock_empty_results):
        tool, _ = self._make_tool(mock_empty_results)
        result = tool.execute(query="anything")
        assert result == "No relevant content found."

    def test_execute_returns_empty_message_with_course_filter(self, mock_empty_results):
        tool, _ = self._make_tool(mock_empty_results)
        result = tool.execute(query="anything", course_name="Python")
        assert "in course 'Python'" in result

    def test_execute_returns_empty_message_with_lesson_filter(self, mock_empty_results):
        tool, _ = self._make_tool(mock_empty_results)
        result = tool.execute(query="anything", lesson_number=3)
        assert "in lesson 3" in result

    def test_execute_returns_empty_message_with_both_filters(self, mock_empty_results):
        tool, _ = self._make_tool(mock_empty_results)
        result = tool.execute(query="anything", course_name="Python", lesson_number=3)
        assert "in course 'Python'" in result
        assert "in lesson 3" in result

    # -- success path ------------------------------------------------

    def test_execute_returns_formatted_results(self, mock_search_results):
        tool, store = self._make_tool(mock_search_results)
        store.get_lesson_link.return_value = None
        store.get_course_link.return_value = None
        result = tool.execute(query="basics")
        assert "[Intro to Python - Lesson 1]" in result
        assert "Content about Python basics" in result

    def test_execute_passes_params_to_store_search(self, mock_search_results):
        tool, store = self._make_tool(mock_search_results)
        store.get_lesson_link.return_value = None
        store.get_course_link.return_value = None
        tool.execute(query="q", course_name="C", lesson_number=5)
        store.search.assert_called_once_with(query="q", course_name="C", lesson_number=5)


# ===================================================================
# CourseSearchTool._format_results()
# ===================================================================

class TestFormatResults:

    def _make_tool(self):
        store = MagicMock()
        store.get_lesson_link.return_value = None
        store.get_course_link.return_value = None
        return CourseSearchTool(store), store

    def test_format_results_header_with_lesson(self):
        tool, _ = self._make_tool()
        results = SearchResults(
            documents=["doc"],
            metadata=[{"course_title": "ML", "lesson_number": 2}],
            distances=[0.1],
        )
        out = tool._format_results(results)
        assert "[ML - Lesson 2]" in out

    def test_format_results_header_without_lesson(self):
        tool, _ = self._make_tool()
        results = SearchResults(
            documents=["doc"],
            metadata=[{"course_title": "ML"}],
            distances=[0.1],
        )
        out = tool._format_results(results)
        assert "[ML]" in out
        assert "Lesson" not in out

    def test_format_results_populates_last_sources(self):
        tool, _ = self._make_tool()
        results = SearchResults(
            documents=["doc"],
            metadata=[{"course_title": "ML", "lesson_number": 1}],
            distances=[0.1],
        )
        tool._format_results(results)
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["text"] == "ML - Lesson 1"
        assert "link" in tool.last_sources[0]

    def test_format_results_lesson_link_preferred(self):
        tool, store = self._make_tool()
        store.get_lesson_link.return_value = "http://lesson"
        store.get_course_link.return_value = "http://course"
        results = SearchResults(
            documents=["doc"],
            metadata=[{"course_title": "ML", "lesson_number": 1}],
            distances=[0.1],
        )
        tool._format_results(results)
        assert tool.last_sources[0]["link"] == "http://lesson"

    def test_format_results_falls_back_to_course_link(self):
        tool, store = self._make_tool()
        store.get_lesson_link.return_value = None
        store.get_course_link.return_value = "http://course"
        results = SearchResults(
            documents=["doc"],
            metadata=[{"course_title": "ML", "lesson_number": 1}],
            distances=[0.1],
        )
        tool._format_results(results)
        assert tool.last_sources[0]["link"] == "http://course"

    def test_format_results_multiple_documents(self):
        tool, _ = self._make_tool()
        results = SearchResults(
            documents=["d1", "d2", "d3"],
            metadata=[
                {"course_title": "A", "lesson_number": 1},
                {"course_title": "B", "lesson_number": 2},
                {"course_title": "C", "lesson_number": 3},
            ],
            distances=[0.1, 0.2, 0.3],
        )
        out = tool._format_results(results)
        assert out.count("\n\n") == 2  # 3 blocks separated by \n\n
        assert len(tool.last_sources) == 3


# ===================================================================
# ToolManager
# ===================================================================

class TestToolManager:

    def _make_mock_tool(self, name="my_tool"):
        tool = MagicMock()
        tool.get_tool_definition.return_value = {"name": name}
        tool.execute.return_value = "result"
        tool.last_sources = []
        return tool

    def test_register_tool(self):
        mgr = ToolManager()
        tool = self._make_mock_tool("abc")
        mgr.register_tool(tool)
        assert "abc" in mgr.tools

    def test_execute_unknown_tool(self):
        mgr = ToolManager()
        result = mgr.execute_tool("nonexistent")
        assert "not found" in result.lower()

    def test_execute_tool_delegates(self):
        mgr = ToolManager()
        tool = self._make_mock_tool("abc")
        mgr.register_tool(tool)
        mgr.execute_tool("abc", query="q")
        tool.execute.assert_called_once_with(query="q")

    def test_get_last_sources(self):
        mgr = ToolManager()
        tool = self._make_mock_tool("abc")
        tool.last_sources = [{"text": "src", "link": "http://x"}]
        mgr.register_tool(tool)
        assert mgr.get_last_sources() == [{"text": "src", "link": "http://x"}]

    def test_reset_sources(self):
        mgr = ToolManager()
        tool = self._make_mock_tool("abc")
        tool.last_sources = [{"text": "src"}]
        mgr.register_tool(tool)
        mgr.reset_sources()
        assert tool.last_sources == []

    def test_get_tool_definitions(self):
        mgr = ToolManager()
        mgr.register_tool(self._make_mock_tool("a"))
        mgr.register_tool(self._make_mock_tool("b"))
        defs = mgr.get_tool_definitions()
        assert len(defs) == 2
        assert all(isinstance(d, dict) for d in defs)
