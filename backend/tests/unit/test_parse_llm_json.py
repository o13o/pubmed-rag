"""Tests for parse_llm_json utility."""

import pytest

from src.agents import parse_llm_json


def test_plain_json():
    raw = '{"summary": "test", "score": 5}'
    result = parse_llm_json(raw)
    assert result["summary"] == "test"
    assert result["score"] == 5


def test_json_with_json_fence():
    raw = '```json\n{"summary": "test", "score": 5}\n```'
    result = parse_llm_json(raw)
    assert result["summary"] == "test"


def test_json_with_plain_fence():
    raw = '```\n{"summary": "test"}\n```'
    result = parse_llm_json(raw)
    assert result["summary"] == "test"


def test_json_with_whitespace():
    raw = '  \n  {"summary": "test"}  \n  '
    result = parse_llm_json(raw)
    assert result["summary"] == "test"


def test_json_fence_with_surrounding_text():
    raw = 'Here is the result:\n```json\n{"summary": "test"}\n```\nDone.'
    result = parse_llm_json(raw)
    assert result["summary"] == "test"


def test_invalid_json_raises():
    with pytest.raises(Exception):
        parse_llm_json("This is not JSON at all")


def test_empty_string_raises():
    with pytest.raises(Exception):
        parse_llm_json("")
