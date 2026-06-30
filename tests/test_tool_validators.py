"""Tests for pre-execution tool argument validators (args_validator integration)."""

from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import BaseModel, Field
from pydantic_ai.exceptions import ModelRetry

from pydantic_ai_guardrails.tool_validators import (
    args_allowlist_validator,
    args_custom_validator,
    args_schema_validator,
)

# --- Mock RunContext ---


@dataclass
class MockRunContext:
    """Minimal mock for pydantic_ai.tools.RunContext."""

    deps: Any = None


# --- Test schemas ---


class WeatherParams(BaseModel):
    location: str = Field(max_length=50)
    units: str = Field(pattern="^(celsius|fahrenheit)$", default="celsius")


class FileReadParams(BaseModel):
    path: str = Field(pattern="^(/tmp/|\\./public/)")
    encoding: str = Field(default="utf-8")


# --- args_schema_validator tests ---


class TestArgsSchemaValidator:
    def test_valid_args_pass(self):
        validator = args_schema_validator(WeatherParams)
        validator(MockRunContext(), location="Paris", units="celsius")

    def test_valid_args_with_defaults(self):
        validator = args_schema_validator(WeatherParams)
        validator(MockRunContext(), location="London")

    def test_invalid_pattern_raises_model_retry(self):
        validator = args_schema_validator(WeatherParams)
        with pytest.raises(ModelRetry, match="Invalid tool arguments"):
            validator(MockRunContext(), location="Paris", units="kelvin")

    def test_missing_required_field_raises_model_retry(self):
        validator = args_schema_validator(WeatherParams)
        with pytest.raises(ModelRetry, match="Invalid tool arguments"):
            validator(MockRunContext())

    def test_max_length_violation_raises_model_retry(self):
        validator = args_schema_validator(WeatherParams)
        with pytest.raises(ModelRetry, match="Invalid tool arguments"):
            validator(MockRunContext(), location="x" * 100, units="celsius")

    def test_multiple_errors_combined(self):
        validator = args_schema_validator(FileReadParams)
        with pytest.raises(ModelRetry, match="Invalid tool arguments"):
            validator(MockRunContext(), path="../../etc/passwd")

    def test_error_message_includes_field_details(self):
        validator = args_schema_validator(WeatherParams)
        with pytest.raises(ModelRetry, match="units"):
            validator(MockRunContext(), location="Paris", units="kelvin")


# --- args_custom_validator tests ---


class TestArgsCustomValidator:
    def test_none_return_passes(self):
        def check(args: dict[str, Any]) -> str | None:
            return None

        validator = args_custom_validator(check)
        validator(MockRunContext(), path="/tmp/safe.txt")

    def test_error_string_raises_model_retry(self):
        def check(args: dict[str, Any]) -> str | None:
            if ".." in args.get("path", ""):
                return "Path traversal detected"
            return None

        validator = args_custom_validator(check)
        with pytest.raises(ModelRetry, match="Path traversal detected"):
            validator(MockRunContext(), path="../../etc/passwd")

    def test_receives_kwargs_as_dict(self):
        received = {}

        def check(args: dict[str, Any]) -> str | None:
            received.update(args)
            return None

        validator = args_custom_validator(check)
        validator(MockRunContext(), name="test", value=42)
        assert received == {"name": "test", "value": 42}

    def test_validator_exception_propagates(self):
        def check(args: dict[str, Any]) -> str | None:
            raise ValueError("Unexpected error")

        validator = args_custom_validator(check)
        with pytest.raises(ValueError, match="Unexpected error"):
            validator(MockRunContext(), path="/tmp/test")


# --- args_allowlist_validator tests ---


class TestArgsAllowlistValidator:
    def test_allowed_values_pass(self):
        validator = args_allowlist_validator(
            {"units": ["celsius", "fahrenheit"]}
        )
        validator(MockRunContext(), location="Paris", units="celsius")

    def test_disallowed_value_raises_model_retry(self):
        validator = args_allowlist_validator(
            {"units": ["celsius", "fahrenheit"]}
        )
        with pytest.raises(ModelRetry, match="not allowed"):
            validator(MockRunContext(), location="Paris", units="kelvin")

    def test_unchecked_params_pass_through(self):
        validator = args_allowlist_validator(
            {"units": ["celsius", "fahrenheit"]}
        )
        # 'location' is not in the allowlist dict, so it passes through
        validator(MockRunContext(), location="anywhere", units="celsius")

    def test_multiple_violations_combined(self):
        validator = args_allowlist_validator(
            {
                "units": ["celsius", "fahrenheit"],
                "language": ["en", "es", "fr"],
            }
        )
        with pytest.raises(ModelRetry, match="units.*not allowed"):
            validator(
                MockRunContext(), units="kelvin", language="en"
            )

    def test_empty_allowlist_blocks_all_values(self):
        validator = args_allowlist_validator({"units": []})
        with pytest.raises(ModelRetry, match="not allowed"):
            validator(MockRunContext(), units="celsius")

    def test_error_includes_allowed_values(self):
        validator = args_allowlist_validator(
            {"units": ["celsius", "fahrenheit"]}
        )
        with pytest.raises(ModelRetry, match="celsius.*fahrenheit"):
            validator(MockRunContext(), units="kelvin")

    def test_param_not_provided_is_ignored(self):
        validator = args_allowlist_validator(
            {"units": ["celsius", "fahrenheit"]}
        )
        # 'units' not provided at all -- no error
        validator(MockRunContext(), location="Paris")


# --- Import tests ---


class TestImports:
    def test_importable_from_package(self):
        from pydantic_ai_guardrails import (
            args_allowlist_validator,
            args_custom_validator,
            args_schema_validator,
        )

        assert callable(args_schema_validator)
        assert callable(args_custom_validator)
        assert callable(args_allowlist_validator)
