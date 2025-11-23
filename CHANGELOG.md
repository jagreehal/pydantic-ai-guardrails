# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-11-23 - Initial Release

### Added

- **Core Foundation**:
  - `GuardrailResult`: TypedDict for validation results
  - `InputGuardrail`: Dataclass for input validation
  - `OutputGuardrail`: Dataclass for output validation
  - `GuardrailViolation`: Base exception extending `AgentRunError`
  - `InputGuardrailViolation`: Input validation failures
  - `OutputGuardrailViolation`: Output validation failures
  - `with_guardrails()`: Non-invasive wrapper function
  - Auto-detection of sync/async functions
  - Auto-detection of `RunContext` parameter
  - Support for `on_block` modes: raise, log, silent

- **10 Production-Ready Built-in Guardrails**:

  **Input Guardrails (5)**:
  - `length_limit()`: Character/token limits with tiktoken support
  - `pii_detector()`: PII detection (email, phone, SSN, credit card, IP)
  - `prompt_injection()`: Security attack detection (3 sensitivity levels)
  - `toxicity_detector()`: Harmful content detection (pattern + optional ML)
  - `rate_limiter()`: Sliding window rate limiting per user/key

  **Output Guardrails (5)**:
  - `min_length()`: Response quality validation
  - `secret_redaction()`: API key/token detection (8 secret types)
  - `json_validator()`: Structured output validation with schema support
  - `toxicity_filter()`: Harmful content filtering
  - `hallucination_detector()`: Uncertainty/fake data detection

- **OpenTelemetry Integration**: Full observability support with optional telemetry
  - `GuardrailTelemetry` class for span management
  - Automatic span creation for each guardrail execution
  - Performance metrics (duration_ms) and violation events
  - Global configuration via `configure_telemetry()`
  - Gracefully degrades when OpenTelemetry is not installed

- **Parallel Execution**: Concurrent guardrail execution for better performance
  - `execute_input_guardrails_parallel()` utility
  - `execute_output_guardrails_parallel()` utility
  - `parallel=True` parameter in `with_guardrails()`
  - Async-safe concurrent execution using `asyncio.gather()`

- **Configuration System**: Load guardrails from JSON/YAML files
  - `GuardrailConfig` class for configuration management
  - `load_config()` to load from JSON or YAML files
  - `load_guardrails_from_config()` to create guardrails from configuration
  - `create_guarded_agent_from_config()` for one-line setup
  - Support for all built-in guardrails via configuration
  - Team-friendly configuration management

- **OpenAI Guardrails Compatibility**: 100% compatible with OpenAI Guardrails config format
  - Load configs generated from OpenAI Guardrails UI directly
  - Automatic mapping of OpenAI guardrail names to our implementations
  - Support for `pre_flight`, `input`, and `output` sections
  - Maps OpenAI config parameters to our format automatically
  - Supports both OpenAI names ("Contains PII") and our names ("pii_detector")
  - Graceful handling of unimplemented guardrails with warnings
  - Extended PII entity type mapping (50+ entity types)

- **Testing Utilities**: Comprehensive testing tools for guardrail development
  - `assert_guardrail_passes()` - Assert guardrail doesn't trigger
  - `assert_guardrail_blocks()` - Assert guardrail triggers
  - `assert_guardrail_result()` - Assert specific result values
  - `GuardrailTestCases` - Batch test case generator
  - `MockAgent` - Mock agent for testing without LLM calls
  - `create_test_context()` - Create test contexts for guardrails

- **Type Safety**: Full mypy compliance with strict type checking
  - Defined `AgentDepsT` locally to prevent import drift
  - Input guardrails accept `str | Sequence[Any]` for flexibility
  - Added proper `Sequence` imports from `collections.abc`
  - Used `cast()` for literal severity types to maintain type safety
  - Added targeted `# type: ignore` comments only where necessary
  - All 45 source files pass mypy with 0 errors
  - 270 tests passing with strict type checking

- **Documentation**:
  - Comprehensive README with quick start guide
  - Design document (`pydantic_ai_guardrails.md`)
  - Usage examples and integration guides
  - Architecture documentation

### Technical Details

- **Core Modules**:
  - `_guardrails.py`: Core guardrail classes
  - `_results.py`: Result type definitions
  - `_integration.py`: Agent wrapper integration
  - `exceptions.py`: Exception hierarchy
  - `_telemetry.py`: OpenTelemetry integration
  - `_parallel.py`: Parallel execution utilities
  - `_config.py`: Configuration system for JSON/YAML loading
  - `_testing.py`: Testing utilities for guardrail development

- **Test Coverage**:
  - 270 passing tests with strict type checking
  - Configuration system tests
  - Testing utilities tests
  - Integration tests with real LLM (Ollama)
  - Pattern-based detection works out-of-the-box
  - Optional ML support via detoxify library

## Design Philosophy

### Native Pydantic AI Integration

The library follows Pydantic AI's architectural patterns exactly:

1. **DataClass-based Design**: Mirrors `OutputValidator` pattern
2. **Auto-detection**: Automatically detects sync/async and context parameters
3. **Exception-based Control Flow**: Extends `AgentRunError` for native feel
4. **Type Safety**: Full generic type support with IDE autocomplete
5. **RunContext Integration**: Native dependency injection support

### Why Not Just Output Functions?

While Pydantic AI has output functions, they:

- Only validate output, not input
- Consume the output type slot
- Lack structured error reporting
- Can't prevent execution before it starts

This library provides:

- ✅ Input validation before agent execution
- ✅ Output validation after model response
- ✅ Structured results with severity levels
- ✅ Native Pydantic AI patterns
- ✅ Type-safe with full IDE support

## Quick Start

### Basic Usage

```python
from pydantic_ai_guardrails import with_guardrails
from pydantic_ai_guardrails.guardrails.input import length_limit, pii_detector
from pydantic_ai_guardrails.guardrails.output import min_length, secret_redaction

guarded_agent = with_guardrails(
    agent,
    input_guardrails=[length_limit(max_chars=500), pii_detector()],
    output_guardrails=[min_length(min_chars=20), secret_redaction()],
)
```

### Configuration from Files

The configuration system uses OpenAI Guardrails format for maximum compatibility:

```json
{
  "version": 1,
  "input": {
    "version": 1,
    "guardrails": [
      {"name": "pii_detector", "config": {...}}
    ]
  },
  "output": {
    "version": 1,
    "guardrails": [
      {"name": "min_length", "config": {...}}
    ]
  }
}
```

```python
# Load OpenAI Guardrails config directly
from pydantic_ai_guardrails import create_guarded_agent_from_config
guarded_agent = create_guarded_agent_from_config(
    agent,
    "openai_guardrails_config.json"  # From OpenAI Guardrails UI
)

# Enable telemetry
from pydantic_ai_guardrails import configure_telemetry
configure_telemetry(enabled=True)

# Use parallel execution
guarded_agent = with_guardrails(
    agent,
    input_guardrails=[...],
    parallel=True,
)
```

## Roadmap

### Future Releases

- Complete API stability guarantee (v1.0.0)
- Performance benchmarks and optimization
- Additional built-in guardrails
- Community guardrail registry
- Official plugin system

## Contributors

- Jag Reehal (@jagreehal) - Creator and maintainer

## Acknowledgments

- Inspired by [OpenAI Guardrails](https://openai.github.io/openai-guardrails-python/)
- Built for [Pydantic AI](https://ai.pydantic.dev/)
- Follows patterns from [ai-sdk-guardrails](https://github.com/jagreehal/ai-sdk-guardrails)

## License

MIT License - see [LICENSE](LICENSE) file for details.
