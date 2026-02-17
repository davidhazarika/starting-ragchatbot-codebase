"""Shared fixtures and path setup for backend tests."""

import sys
import os
from unittest.mock import MagicMock
from dataclasses import dataclass

import pytest

# Ensure bare imports like `from vector_store import ...` resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vector_store import SearchResults


# ---------------------------------------------------------------------------
# Search-result fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_search_results():
    """Non-empty search results with two documents."""
    return SearchResults(
        documents=["Content about Python basics", "Content about variables"],
        metadata=[
            {"course_title": "Intro to Python", "lesson_number": 1},
            {"course_title": "Intro to Python", "lesson_number": 2},
        ],
        distances=[0.1, 0.2],
    )


@pytest.fixture
def mock_empty_results():
    """Empty (but successful) search results."""
    return SearchResults(documents=[], metadata=[], distances=[])


@pytest.fixture
def mock_error_results():
    """Search results representing an error."""
    return SearchResults.empty("Search error: connection failed")


# ---------------------------------------------------------------------------
# Anthropic response helpers
# ---------------------------------------------------------------------------

def make_text_block(text: str):
    """Create a mock content block with type='text'."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def make_tool_use_block(name: str, tool_id: str, inputs: dict):
    """Create a mock content block with type='tool_use'."""
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.id = tool_id
    block.input = inputs
    return block


def make_response(stop_reason: str, content_blocks: list):
    """Create a mock Anthropic response object."""
    resp = MagicMock()
    resp.stop_reason = stop_reason
    resp.content = content_blocks
    return resp
