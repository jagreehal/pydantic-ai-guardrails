"""Build guardrail capabilities from a JSON or YAML config file.

```yaml
# guardrails.yaml
version: 1
guardrails:
  - type: DetectPII
    config: {types: [ssn, credit_card]}
  - type: LimitOutputLength
    config: {min_chars: 10, on_fail: retry}
  - type: LimitCost
    config: {max_total_tokens: 50000}
```

```python
from pydantic_ai import Agent
from pydantic_ai_guardrails import build_guardrails, load_config

agent = Agent('openai:gpt-4o', capabilities=build_guardrails(load_config('guardrails.yaml')))
```
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic_ai.capabilities import AbstractCapability

from pydantic_ai_guardrails._cost import LimitCost
from pydantic_ai_guardrails._input import (
    BlockKeywords,
    DetectPII,
    DetectPromptInjection,
    DetectToxicity,
    LimitInputLength,
    RateLimit,
)
from pydantic_ai_guardrails._output import (
    BlockRefusals,
    DetectHallucination,
    FilterToxicity,
    LimitOutputLength,
    LlmJudge,
    MatchRegex,
    RedactSecrets,
    RequireToolUse,
    ValidateJson,
)
from pydantic_ai_guardrails._tool import RestrictTools

__all__ = ('GuardrailConfig', 'load_config', 'build_guardrails', 'REGISTRY')

_GUARDRAILS: tuple[type[AbstractCapability[Any]], ...] = (
    # Input
    DetectPII,
    DetectPromptInjection,
    DetectToxicity,
    BlockKeywords,
    LimitInputLength,
    RateLimit,
    # Output
    LimitOutputLength,
    RedactSecrets,
    ValidateJson,
    FilterToxicity,
    DetectHallucination,
    MatchRegex,
    BlockRefusals,
    RequireToolUse,
    LlmJudge,
    # Tool and cost
    RestrictTools,
    LimitCost,
)

#: Maps a config ``type`` to the capability class that implements it.
REGISTRY: dict[str, type[AbstractCapability[Any]]] = {cls.__name__: cls for cls in _GUARDRAILS}


@dataclass
class GuardrailConfig:
    """Parsed guardrail configuration."""

    version: int = 1
    guardrails: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GuardrailConfig:
        version = data.get('version', 1)
        if version != 1:
            raise ValueError(f'unsupported config version: {version}')
        guardrails = data.get('guardrails', [])
        if not isinstance(guardrails, list):
            raise ValueError("'guardrails' must be a list")
        return cls(version=version, guardrails=guardrails)


def load_config(path: str | Path) -> GuardrailConfig:
    """Load a guardrail config from a ``.json``, ``.yaml`` or ``.yml`` file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'config file not found: {path}')
    text = path.read_text(encoding='utf-8')
    if path.suffix.lower() in {'.yaml', '.yml'}:
        try:
            import yaml
        except ImportError as exc:
            raise ValueError('YAML config requires PyYAML (pip install pyyaml)') from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError('config root must be a mapping')
    return GuardrailConfig.from_dict(data)


def build_guardrails(config: GuardrailConfig | dict[str, Any]) -> list[AbstractCapability[Any]]:
    """Instantiate the guardrail capabilities described by *config*."""
    parsed = config if isinstance(config, GuardrailConfig) else GuardrailConfig.from_dict(config)
    capabilities: list[AbstractCapability[Any]] = []
    for item in parsed.guardrails:
        type_name = item.get('type')
        if not type_name:
            raise ValueError("each guardrail needs a 'type'")
        cls = REGISTRY.get(type_name)
        if cls is None:
            raise ValueError(f'unknown guardrail type: {type_name!r} (known: {", ".join(sorted(REGISTRY))})')
        capabilities.append(cls(**item.get('config', {})))
    return capabilities
