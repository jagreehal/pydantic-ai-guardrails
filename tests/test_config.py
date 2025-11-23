"""Tests for configuration system."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from pydantic_ai_guardrails import (
    GuardrailConfig,
    create_guarded_agent_from_config,
    load_config,
    load_guardrails_from_config,
)


class TestGuardrailConfig:
    """Test GuardrailConfig class."""

    def test_create_config(self):
        """Test creating configuration."""
        config = GuardrailConfig(
            version=1,
            settings={"parallel": True},
            input_guardrails=[{"name": "length_limit", "config": {"max_chars": 100}}],
            output_guardrails=[{"name": "min_length", "config": {"min_chars": 10}}],
        )

        assert config.version == 1
        assert config.settings == {"parallel": True}
        assert len(config.input_guardrails) == 1
        assert len(config.output_guardrails) == 1

    def test_from_dict(self):
        """Test creating configuration from dictionary."""
        data = {
            "version": 1,
            "settings": {"parallel": True, "on_block": "raise"},
            "input": {
                "version": 1,
                "guardrails": [
                    {"name": "length_limit", "config": {"max_chars": 100}}
                ]
            },
            "output": {
                "version": 1,
                "guardrails": [
                    {"name": "min_length", "config": {"min_chars": 10}}
                ]
            },
        }

        config = GuardrailConfig.from_dict(data)

        assert config.version == 1
        assert config.settings["parallel"] is True
        assert len(config.input_guardrails) == 1

    def test_to_dict(self):
        """Test converting configuration to dictionary."""
        config = GuardrailConfig(
            version=1,
            settings={"parallel": True},
            input_guardrails=[{"name": "length_limit", "config": {"max_chars": 100}}],
        )

        data = config.to_dict()

        assert data["version"] == 1
        assert data["settings"]["parallel"] is True
        assert len(data["input"]["guardrails"]) == 1

    def test_unsupported_version(self):
        """Test that unsupported version raises error."""
        data = {"version": 999}

        with pytest.raises(ValueError, match="Unsupported configuration version"):
            GuardrailConfig.from_dict(data)


class TestLoadConfig:
    """Test configuration loading."""

    def test_load_json_config(self):
        """Test loading JSON configuration."""
        # Create temporary config file
        config_data = {
            "version": 1,
            "settings": {"parallel": True},
            "input": {
                "version": 1,
                "guardrails": [
                    {"name": "length_limit", "config": {"max_chars": 100}}
                ]
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            # Load configuration
            config = load_config(config_path)

            assert config.version == 1
            assert config.settings["parallel"] is True
            assert len(config.input_guardrails) == 1

        finally:
            Path(config_path).unlink()

    def test_load_nonexistent_file(self):
        """Test loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.json")

    def test_unsupported_format(self):
        """Test loading unsupported format raises error."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            config_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported configuration file format"):
                load_config(config_path)
        finally:
            Path(config_path).unlink()


class TestLoadGuardrailsFromConfig:
    """Test loading guardrails from configuration."""

    def test_load_input_guardrails(self):
        """Test loading input guardrails."""
        config = GuardrailConfig(
            version=1,
            input_guardrails=[
                {"name": "length_limit", "config": {"max_chars": 100}},
                {"name": "pii_detector", "config": {}},
            ],
        )

        input_guardrails, output_guardrails, settings = load_guardrails_from_config(config)

        assert len(input_guardrails) == 2
        assert len(output_guardrails) == 0

    def test_load_output_guardrails(self):
        """Test loading output guardrails."""
        config = GuardrailConfig(
            version=1,
            output_guardrails=[
                {"name": "min_length", "config": {"min_chars": 10}},
                {"name": "secret_redaction", "config": {}},
            ],
        )

        input_guardrails, output_guardrails, settings = load_guardrails_from_config(config)

        assert len(input_guardrails) == 0
        assert len(output_guardrails) == 2

    def test_load_settings(self):
        """Test loading settings."""
        config = GuardrailConfig(
            version=1,
            settings={
                "parallel": True,
                "on_block": "log",
            },
        )

        input_guardrails, output_guardrails, settings = load_guardrails_from_config(config)

        assert settings["parallel"] is True
        assert settings["on_block"] == "log"

    def test_unknown_guardrail_type(self):
        """Test that unknown guardrail type prints warning and continues."""
        config = GuardrailConfig(
            version=1,
            input_guardrails=[{"name": "unknown_guardrail", "config": {}}],
        )

        # Should not raise, just warn
        input_guardrails, _, _ = load_guardrails_from_config(config)
        assert len(input_guardrails) == 0

    def test_invalid_on_block_value(self):
        """Test that invalid on_block value raises error."""
        config = GuardrailConfig(
            version=1,
            settings={"on_block": "invalid"},
        )

        with pytest.raises(ValueError, match="Invalid on_block value"):
            load_guardrails_from_config(config)

    def test_missing_type_field(self):
        """Test that missing name field raises error."""
        config = GuardrailConfig(
            version=1,
            input_guardrails=[{"config": {}}],  # Missing 'name'
        )

        with pytest.raises(ValueError, match="missing 'name' field"):
            load_guardrails_from_config(config)

    def test_invalid_guardrail_config(self):
        """Test that invalid guardrail config raises error."""
        config = GuardrailConfig(
            version=1,
            input_guardrails=[
                {
                    "name": "length_limit",
                    "config": {
                        "invalid_param": 123
                    },  # Invalid parameter
                }
            ],
        )

        with pytest.raises(ValueError, match="Invalid configuration"):
            load_guardrails_from_config(config)


class TestCreateGuardedAgentFromConfig:
    """Test creating guarded agent from configuration."""

    async def test_create_from_json_file(self):
        """Test creating guarded agent from JSON file."""
        from pydantic_ai import Agent

        # Create temporary config file
        config_data = {
            "version": 1,
            "settings": {"parallel": False, "on_block": "raise"},
            "input": {
                "version": 1,
                "guardrails": [
                    {"name": "length_limit", "config": {"max_chars": 100}}
                ]
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            # Create agent
            agent = Agent("test")

            # Create guarded agent from config
            guarded_agent = create_guarded_agent_from_config(agent, config_path)

            # Test it works
            result = await guarded_agent.run("Short prompt")
            assert result is not None

        finally:
            Path(config_path).unlink()

    async def test_create_blocks_long_prompt(self):
        """Test that guarded agent blocks long prompts."""
        from pydantic_ai import Agent

        from pydantic_ai_guardrails.exceptions import InputGuardrailViolation

        # Create temporary config file
        config_data = {
            "version": 1,
            "input": {
                "version": 1,
                "guardrails": [
                    {"name": "length_limit", "config": {"max_chars": 10}}
                ]
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            # Create guarded agent
            agent = Agent("test")
            guarded_agent = create_guarded_agent_from_config(agent, config_path)

            # Test blocks long prompt
            with pytest.raises(InputGuardrailViolation):
                await guarded_agent.run("This is a very long prompt")

        finally:
            Path(config_path).unlink()


class TestAllBuiltInGuardrails:
    """Test that all built-in guardrails can be loaded from config."""

    def test_all_input_guardrails_loadable(self):
        """Test all input guardrails can be loaded."""
        config = GuardrailConfig(
            version=1,
            input_guardrails=[
                {"name": "length_limit", "config": {"max_chars": 100}},
                {"name": "pii_detector", "config": {}},
                {"name": "prompt_injection", "config": {}},
                {"name": "toxicity_detector", "config": {}},
                {"name": "rate_limiter", "config": {"max_requests": 10, "window_seconds": 60}},
            ],
        )

        input_guardrails, _, _ = load_guardrails_from_config(config)

        assert len(input_guardrails) == 5

    def test_all_output_guardrails_loadable(self):
        """Test all output guardrails can be loaded."""
        config = GuardrailConfig(
            version=1,
            output_guardrails=[
                {"name": "min_length", "config": {"min_chars": 10}},
                {"name": "secret_redaction", "config": {}},
                {"name": "json_validator", "config": {}},
                {"name": "toxicity_filter", "config": {}},
                {"name": "hallucination_detector", "config": {}},
            ],
        )

        _, output_guardrails, _ = load_guardrails_from_config(config)

        assert len(output_guardrails) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
