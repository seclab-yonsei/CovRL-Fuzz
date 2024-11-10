import pytest
from covrl.utils.map_target_error import *


def test_v8_error_mapping():
    result = map_target_error("v8")
    assert result == V8_ERROR, "V8_ERROR mapping did not match the expected result."


def test_jsc_error_mapping():
    result = map_target_error("jsc")
    assert result == JSC_ERROR, "JSC_ERROR mapping did not match the expected result."


def test_chakra_error_mapping():
    result = map_target_error("chakra")
    assert (
        result == CHAKRA_ERROR
    ), "CHAKRA_ERROR mapping did not match the expected result."


def test_jerry_error_mapping():
    result = map_target_error("jerry")
    assert (
        result == JERRY_ERROR
    ), "JERRY_ERROR mapping did not match the expected result."


def test_case_insensitivity():
    result = map_target_error("V8")
    assert result == V8_ERROR, "Mapping should be case-insensitive but failed for 'V8'."


def test_invalid_target():
    result = map_target_error("invalid_target")
    assert (
        result is None
    ), "Expected None for an invalid target but got a different result."
