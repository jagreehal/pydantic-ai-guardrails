"""Shared utilities for examples.

This module provides common configuration and setup functions
used across all examples.
"""

import os


def setup_api_config():
    """Configure API settings based on OPENAI_API_KEY.

    If OPENAI_API_KEY is "ollama", configures for local Ollama.
    Otherwise, uses real OpenAI (assumes key is already set).

    This allows examples to work locally with Ollama by simply setting:
        export OPENAI_API_KEY=ollama

    Or use real OpenAI by setting a real API key:
        export OPENAI_API_KEY=sk-...
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")

    if api_key.lower() == "ollama":
        # Configure for local Ollama
        os.environ["OPENAI_API_KEY"] = "ollama"  # Dummy key for local Ollama
        os.environ["OPENAI_BASE_URL"] = "http://localhost:11434/v1"
        return "ollama"
    else:
        # Use real OpenAI (key should already be set)
        # Remove base URL if it was set for Ollama
        if os.environ.get("OPENAI_BASE_URL") == "http://localhost:11434/v1":
            os.environ.pop("OPENAI_BASE_URL", None)
        return "openai"


def get_model_name(default_ollama: str = "llama3.2", default_openai: str = "gpt-4o") -> str:
    """Get the appropriate model name based on API configuration.

    Args:
        default_ollama: Model name to use with Ollama (default: "llama3.2")
        default_openai: Model name to use with OpenAI (default: "gpt-4o")

    Returns:
        Model string in format "openai:model_name"
    """
    api_type = setup_api_config()

    if api_type == "ollama":
        return f"openai:{default_ollama}"
    else:
        return f"openai:{default_openai}"

