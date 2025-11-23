"""Configuration system for loading guardrails from JSON/YAML files.

This module provides a configuration-based approach to guardrail setup,
allowing teams to manage guardrails through configuration files rather
than code.

Example JSON configuration:
    ```json
    {
      "version": 1,
      "settings": {
        "parallel": true,
        "on_block": "raise",
        "telemetry": true
      },
      "input_guardrails": [
        {
          "type": "length_limit",
          "name": "input_length_check",
          "config": {
            "max_chars": 500,
            "max_tokens": 100
          }
        },
        {
          "type": "pii_detector",
          "config": {
            "detect_types": ["email", "phone", "ssn"]
          }
        }
      ],
      "output_guardrails": [
        {
          "type": "min_length",
          "config": {
            "min_chars": 20
          }
        }
      ]
    }
    ```
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ._guardrails import InputGuardrail, OutputGuardrail

# Import built-in guardrails
from .guardrails.input import (
    length_limit,
    pii_detector,
    prompt_injection,
    rate_limiter,
    toxicity_detector,
)
from .guardrails.output import (
    hallucination_detector,
    json_validator,
    min_length,
    secret_redaction,
    toxicity_filter,
)

__all__ = (
    "GuardrailConfig",
    "load_config",
    "load_guardrails_from_config",
)

# Optional YAML support
try:
    import yaml  # type: ignore[import-untyped]

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# ============================================================================
# OpenAI Guardrails Config Mapping
# ============================================================================

def _map_openai_config_to_ours(
    guardrail_name: str, openai_config: dict[str, Any], is_output: bool = False
) -> dict[str, Any]:
    """Map OpenAI Guardrails config parameters to our parameters.

    Args:
        guardrail_name: The guardrail name from OpenAI config.
        openai_config: The config dict from OpenAI Guardrails.
        is_output: Whether this is for an output guardrail (affects mapping).

    Returns:
        Config dict compatible with our guardrails.
    """
    # Map specific guardrails
    if guardrail_name == "Contains PII" and is_output:
        # For OUTPUT: secret_redaction doesn't take detect_types
        # Just pass action parameter
        return {
            "action": "block" if openai_config.get("block") else "log",
        }

    elif guardrail_name == "Contains PII":
        # For INPUT: pii_detector takes detect_types
        # OpenAI uses "entities", we use "detect_types"
        # Map their entity names to ours
        entities = openai_config.get("entities", [])
        detect_types = []

        # Map OpenAI's extensive entity list to our supported types
        # We support: email, phone, ssn, credit_card, ip_address
        # Map as many as possible, ignore unsupported ones
        entity_mapping = {
            "EMAIL_ADDRESS": "email",
            "PHONE_NUMBER": "phone",
            "US_SSN": "ssn",
            "CREDIT_CARD": "credit_card",
            "IP_ADDRESS": "ip_address",
            "CVV": "credit_card",  # Related to credit cards
            # Many other types not yet supported - will be ignored
        }

        for entity in entities:
            if entity in entity_mapping:
                detected_type = entity_mapping[entity]
                if detected_type not in detect_types:  # Avoid duplicates
                    detect_types.append(detected_type)

        return {
            "detect_types": detect_types if detect_types else None,
            "action": "block" if openai_config.get("block") else "log",
        }

    elif guardrail_name == "Moderation":
        # OpenAI uses "categories", we use "categories" too
        categories = openai_config.get("categories", [])

        # Map their category names to ours
        category_mapping = {
            "sexual": "profanity",
            "sexual/minors": "profanity",
            "hate": "hate_speech",
            "hate/threatening": "threats",
            "harassment": "personal_attacks",
            "harassment/threatening": "threats",
            "self-harm": None,  # Not supported
            "self-harm/intent": None,
            "self-harm/instructions": None,
            "violence": "threats",
            "violence/graphic": "threats",
            "illicit": None,
            "illicit/violent": "threats",
        }

        mapped_categories = []
        for cat in categories:
            if (
                cat in category_mapping
                and category_mapping[cat]
                and category_mapping[cat] not in mapped_categories
            ):
                mapped_categories.append(category_mapping[cat])

        return {"categories": mapped_categories if mapped_categories else None}

    elif guardrail_name in ("Prompt Injection Detection", "Jailbreak"):
        # OpenAI uses "confidence_threshold", we use "sensitivity"
        confidence = openai_config.get("confidence_threshold", 0.7)

        # Map confidence to sensitivity
        if confidence >= 0.8:
            sensitivity = "high"
        elif confidence >= 0.6:
            sensitivity = "medium"
        else:
            sensitivity = "low"

        return {"sensitivity": sensitivity, "action": "block"}

    elif guardrail_name == "Hallucination Detection":
        # Simple mapping - we have similar defaults
        return {
            "check_uncertainty": True,
            "check_suspicious_data": True,
        }

    elif guardrail_name == "NSFW Text":
        # Map to toxicity_filter
        return {"categories": ["profanity", "offensive", "hate_speech"]}

    # Default: return config as-is
    return openai_config


# ============================================================================
# Guardrail Registry
# ============================================================================

# Map of guardrail type names to factory functions
INPUT_GUARDRAIL_REGISTRY = {
    # pydantic-ai-guardrails names
    "length_limit": length_limit,
    "pii_detector": pii_detector,
    "prompt_injection": prompt_injection,
    "toxicity_detector": toxicity_detector,
    "rate_limiter": rate_limiter,
    # OpenAI Guardrails compatible names (mapped to our implementations)
    "Contains PII": pii_detector,  # Maps to pii_detector
    "Moderation": toxicity_detector,  # Maps to toxicity_detector
    "Prompt Injection Detection": prompt_injection,  # Maps to prompt_injection
    "Jailbreak": prompt_injection,  # Maps to prompt_injection (jailbreak is a type of injection)
    "Off Topic Prompts": None,  # Not implemented yet - requires LLM
    "Custom Prompt Check": None,  # Not implemented yet - requires LLM
}

OUTPUT_GUARDRAIL_REGISTRY = {
    # pydantic-ai-guardrails names
    "min_length": min_length,
    "secret_redaction": secret_redaction,
    "json_validator": json_validator,
    "toxicity_filter": toxicity_filter,
    "hallucination_detector": hallucination_detector,
    # OpenAI Guardrails compatible names (mapped to our implementations)
    "URL Filter": None,  # Not implemented yet
    "Contains PII": secret_redaction,  # Maps to secret_redaction for output
    "Hallucination Detection": hallucination_detector,  # Maps to hallucination_detector
    "NSFW Text": toxicity_filter,  # Maps to toxicity_filter
    "Prompt Injection Detection": None,  # Typically input-only, but could block outputs too
}


# ============================================================================
# Configuration Types
# ============================================================================

class GuardrailConfig:
    """Configuration for guardrails loaded from JSON/YAML.

    Uses OpenAI Guardrails-compatible format for maximum compatibility.
    Configurations generated from OpenAI Guardrails UI can be used directly!

    Format:
        {
          "version": 1,
          "input": {
            "version": 1,
            "guardrails": [
              {"name": "Contains PII", "config": {...}},
              {"name": "Prompt Injection Detection", "config": {...}}
            ]
          },
          "output": {
            "version": 1,
            "guardrails": [
              {"name": "Hallucination Detection", "config": {...}},
              {"name": "NSFW Text", "config": {...}}
            ]
          }
        }

    Attributes:
        version: Configuration schema version (currently only 1 supported).
        settings: Global settings for guardrail execution.
        input_guardrails: List of input guardrail configurations.
        output_guardrails: List of output guardrail configurations.
    """

    def __init__(
        self,
        version: int = 1,
        settings: dict[str, Any] | None = None,
        input_guardrails: list[dict[str, Any]] | None = None,
        output_guardrails: list[dict[str, Any]] | None = None,
    ):
        """Initialize guardrail configuration.

        Args:
            version: Configuration schema version.
            settings: Global settings (parallel, on_block, telemetry).
            input_guardrails: List of input guardrail configurations.
            output_guardrails: List of output guardrail configurations.
        """
        self.version = version
        self.settings = settings or {}
        self.input_guardrails = input_guardrails or []
        self.output_guardrails = output_guardrails or []

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GuardrailConfig:
        """Create configuration from dictionary (OpenAI Guardrails format).

        Args:
            data: Configuration dictionary in OpenAI Guardrails format.

        Returns:
            GuardrailConfig instance.

        Raises:
            ValueError: If configuration is invalid.
        """
        version = data.get("version", 1)
        if version != 1:
            raise ValueError(f"Unsupported configuration version: {version}")

        # OpenAI Guardrails format uses "input"/"output" sections
        # "pre_flight" section contains guardrails that run before input validation
        # We treat pre_flight as additional input guardrails
        pre_flight_guardrails = data.get("pre_flight", {}).get("guardrails", [])
        input_guardrails = data.get("input", {}).get("guardrails", [])
        output_guardrails = data.get("output", {}).get("guardrails", [])

        # Combine pre_flight and input guardrails
        all_input_guardrails = pre_flight_guardrails + input_guardrails

        settings = data.get("settings", {})

        return cls(
            version=version,
            settings=settings,
            input_guardrails=all_input_guardrails,
            output_guardrails=output_guardrails,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary (OpenAI Guardrails format).

        Returns:
            Configuration dictionary in OpenAI Guardrails format.
        """
        return {
            "version": self.version,
            "settings": self.settings,
            "input": {
                "version": 1,
                "guardrails": self.input_guardrails,
            },
            "output": {
                "version": 1,
                "guardrails": self.output_guardrails,
            },
        }


# ============================================================================
# Configuration Loading
# ============================================================================

def load_config(path: str | Path) -> GuardrailConfig:
    """Load guardrail configuration from JSON or YAML file.

    Args:
        path: Path to configuration file (.json or .yaml/.yml).

    Returns:
        GuardrailConfig instance.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file format is unsupported or invalid.

    Example:
        ```python
        from pathlib import Path
        from pydantic_ai_guardrails import load_config

        # Load configuration
        config = load_config(Path("guardrails.json"))

        # Use with agent
        from pydantic_ai_guardrails import load_guardrails_from_config
        input_guardrails, output_guardrails, settings = load_guardrails_from_config(config)
        ```
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    # Load based on file extension
    suffix = path.suffix.lower()

    if suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    elif suffix in (".yaml", ".yml"):
        if not YAML_AVAILABLE:
            raise ValueError(
                "YAML support requires PyYAML. Install with: pip install pyyaml"
            )
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    else:
        raise ValueError(
            f"Unsupported configuration file format: {suffix}. "
            "Supported formats: .json, .yaml, .yml"
        )

    return GuardrailConfig.from_dict(data)


def load_guardrails_from_config(
    config: GuardrailConfig,
) -> tuple[
    list[InputGuardrail[Any, Any]],
    list[OutputGuardrail[Any, Any, Any]],
    dict[str, Any],
]:
    """Load guardrails from configuration.

    Args:
        config: Guardrail configuration.

    Returns:
        Tuple of (input_guardrails, output_guardrails, settings).

    Raises:
        ValueError: If guardrail type is unknown or configuration is invalid.

    Example:
        ```python
        from pydantic_ai_guardrails import load_config, load_guardrails_from_config

        config = load_config("guardrails.json")
        input_guardrails, output_guardrails, settings = load_guardrails_from_config(config)

        # Use with agent
        from pydantic_ai_guardrails import with_guardrails
        guarded_agent = with_guardrails(
            agent,
            input_guardrails=input_guardrails,
            output_guardrails=output_guardrails,
            **settings,
        )
        ```
    """
    input_guardrails = []
    output_guardrails = []

    # Load input guardrails
    for item in config.input_guardrails:
        # OpenAI Guardrails format uses "name" field
        guardrail_name = item.get("name")
        if not guardrail_name:
            raise ValueError("Guardrail configuration missing 'name' field")

        # Get factory function - check if it exists first
        factory = INPUT_GUARDRAIL_REGISTRY.get(guardrail_name)

        if factory is None:
            # Guardrail not implemented yet or unknown
            if guardrail_name in INPUT_GUARDRAIL_REGISTRY:
                print(f"Warning: Guardrail '{guardrail_name}' is not implemented yet, skipping")
            else:
                print(f"Warning: Unknown guardrail '{guardrail_name}', skipping")
            continue

        # Get configuration parameters and map to our format
        openai_config = item.get("config", {})
        guardrail_config = _map_openai_config_to_ours(guardrail_name, openai_config, is_output=False)

        # Remove None values
        guardrail_config = {k: v for k, v in guardrail_config.items() if v is not None}

        # Create guardrail
        # Type ignore: factory is from registry dict, mypy can't infer callable type from dict.get()
        try:
            guardrail = factory(**guardrail_config)  # type: ignore[operator]
            input_guardrails.append(guardrail)
        except TypeError as e:
            raise ValueError(
                f"Invalid configuration for guardrail '{guardrail_name}': {e}\n"
                f"Config: {guardrail_config}"
            ) from e

    # Load output guardrails
    for item in config.output_guardrails:
        # OpenAI Guardrails format uses "name" field
        guardrail_name = item.get("name")
        if not guardrail_name:
            raise ValueError("Guardrail configuration missing 'name' field")

        # Get factory function - check if it exists first
        factory = OUTPUT_GUARDRAIL_REGISTRY.get(guardrail_name)

        if factory is None:
            # Guardrail not implemented yet or unknown
            if guardrail_name in OUTPUT_GUARDRAIL_REGISTRY:
                print(f"Warning: Guardrail '{guardrail_name}' is not implemented yet, skipping")
            else:
                print(f"Warning: Unknown guardrail '{guardrail_name}', skipping")
            continue

        # Get configuration parameters and map to our format
        openai_config = item.get("config", {})
        guardrail_config = _map_openai_config_to_ours(guardrail_name, openai_config, is_output=True)

        # Remove None values
        guardrail_config = {k: v for k, v in guardrail_config.items() if v is not None}

        # Create guardrail
        # Type ignore: factory is from registry dict, mypy can't infer callable type from dict.get()
        try:
            guardrail = factory(**guardrail_config)  # type: ignore[operator]
            output_guardrails.append(guardrail)
        except TypeError as e:
            raise ValueError(
                f"Invalid configuration for guardrail '{guardrail_name}': {e}\n"
                f"Config: {guardrail_config}"
            ) from e

    # Extract settings
    settings = {}
    if "parallel" in config.settings:
        settings["parallel"] = config.settings["parallel"]
    if "on_block" in config.settings:
        on_block_value = config.settings["on_block"]
        if on_block_value not in ("raise", "log", "silent"):
            raise ValueError(
                f"Invalid on_block value: {on_block_value}. "
                "Must be 'raise', 'log', or 'silent'"
            )
        settings["on_block"] = on_block_value

    return input_guardrails, output_guardrails, settings


# ============================================================================
# Convenience Functions
# ============================================================================

def create_guarded_agent_from_config(
    agent: Any,
    config_path: str | Path,
) -> Any:
    """Create guarded agent from configuration file.

    This is a convenience function that combines load_config,
    load_guardrails_from_config, and with_guardrails.

    Args:
        agent: The Pydantic AI agent to wrap.
        config_path: Path to configuration file (.json or .yaml/.yml).

    Returns:
        Guarded agent with configured guardrails.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_guardrails import create_guarded_agent_from_config

        agent = Agent('openai:gpt-4o')
        guarded_agent = create_guarded_agent_from_config(
            agent,
            "guardrails.json"
        )

        # Use normally
        result = await guarded_agent.run("Your prompt")
        ```
    """
    from ._integration import with_guardrails

    # Load configuration
    config = load_config(config_path)

    # Load guardrails
    input_guardrails, output_guardrails, settings = load_guardrails_from_config(config)

    # Apply telemetry setting if specified
    if "telemetry" in config.settings:
        from ._telemetry import configure_telemetry

        configure_telemetry(enabled=config.settings["telemetry"])

    # Create guarded agent
    return with_guardrails(
        agent,
        input_guardrails=input_guardrails,
        output_guardrails=output_guardrails,
        **settings,
    )
