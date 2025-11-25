"""Tool Parameter Validation Example

Demonstrates the validate_tool_parameters guardrail that validates tool
arguments against Pydantic schemas and custom validators. This provides
a security layer to prevent invalid, malicious, or dangerous parameters
from reaching tool execution.

This example shows:
1. Schema-based validation with Pydantic models
2. Custom validators for complex security rules
3. Combining schemas and validators
4. Strict vs permissive modes
5. Security best practices for file operations, SQL, etc.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from examples._utils import get_model_name
from pydantic_ai import Agent, RunContext

from pydantic_ai_guardrails import OutputGuardrailViolation, with_guardrails
from pydantic_ai_guardrails.guardrails.output import validate_tool_parameters

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


# Define Pydantic schemas for tool parameters
class WeatherParams(BaseModel):
    """Parameters for weather lookup tool."""

    location: str = Field(max_length=50, description="City or location name")
    units: str = Field(
        pattern="^(celsius|fahrenheit)$",
        default="celsius",
        description="Temperature units",
    )


class FileReadParams(BaseModel):
    """Secure parameters for file reading tool."""

    path: str = Field(
        pattern="^(/tmp/|\\./public/)",
        description="File path (restricted to /tmp/ or ./public/)",
    )


class DatabaseQueryParams(BaseModel):
    """Parameters for database queries."""

    table: str = Field(pattern="^[a-zA-Z_][a-zA-Z0-9_]*$", max_length=50)
    columns: list[str] = Field(max_items=10)
    limit: int = Field(ge=1, le=1000, default=100)


# Custom validation functions for additional security
def validate_file_path(args: dict[str, Any]) -> str | None:
    """Custom validator to prevent path traversal and restricted access."""
    path = args.get("path", "")

    # Check for path traversal
    if ".." in path:
        return "Path traversal detected (..)"""

    # Block access to sensitive directories
    forbidden = ["/etc/", "/root/", "/home/", "~"]
    for forbidden_path in forbidden:
        if path.startswith(forbidden_path):
            return f"Access to {forbidden_path} is forbidden"

    return None


def validate_sql_query(args: dict[str, Any]) -> str | None:
    """Custom validator to prevent SQL injection."""
    table = args.get("table", "")
    columns = args.get("columns", [])

    # Check for SQL injection patterns in table name
    dangerous_patterns = ["--", ";", "/*", "*/", "union", "drop", "delete", "insert"]
    table_lower = table.lower()
    for pattern in dangerous_patterns:
        if pattern in table_lower:
            return f"Dangerous pattern detected in table name: {pattern}"

    # Check columns
    for column in columns:
        column_lower = column.lower()
        for pattern in dangerous_patterns:
            if pattern in column_lower:
                return f"Dangerous pattern detected in column: {pattern}"

    return None


# Define mock tools for demonstration
def get_weather(ctx: RunContext[None], location: str, units: str = "celsius") -> str:
    """Get weather for a location."""
    print(f"  🌤️  get_weather called: location={location}, units={units}")
    return f"Weather in {location}: Sunny, 72°{units[0].upper()}"


def read_file(ctx: RunContext[None], path: str) -> str:
    """Read a file (restricted paths)."""
    print(f"  📁 read_file called: path={path}")
    return f"Contents of {path}"


def query_database(
    ctx: RunContext[None], table: str, columns: list[str], limit: int = 100
) -> str:
    """Query database (safe parameters only)."""
    print(f"  🗄️  query_database called: table={table}, columns={columns}, limit={limit}")
    return f"Query results from {table}"


# Create agent with tools
agent = Agent(
    get_model_name(),
    system_prompt=(
        "You are a helpful assistant with access to tools. "
        "Use the available tools to answer questions."
    ),
    tools=[get_weather, read_file, query_database],
)


async def example_schema_validation():
    """Example 1: Schema-based parameter validation."""
    print("\n" + "=" * 70)
    print("Example 1: Schema-Based Parameter Validation")
    print("=" * 70)
    print("Validates tool arguments against Pydantic schemas.\n")

    guarded_agent = with_guardrails(
        agent,
        output_guardrails=[
            validate_tool_parameters(
                schemas={
                    "get_weather": WeatherParams,
                    "read_file": FileReadParams,
                }
            )
        ],
        max_retries=0,
        on_block="raise",
    )

    # Test valid parameters
    try:
        print("Query: What's the weather in Paris?\n")
        result = await guarded_agent.run("What's the weather in Paris?")
        print("\n✅ Success! Parameters valid.")
        print(f"Response: {result.output}")
    except OutputGuardrailViolation as e:
        print(f"\n❌ Blocked: {e.result.get('message')}")


async def example_custom_validators():
    """Example 2: Custom validation functions."""
    print("\n" + "=" * 70)
    print("Example 2: Custom Validation Functions")
    print("=" * 70)
    print("Use custom validators for complex security rules.\n")

    guarded_agent = with_guardrails(
        agent,
        output_guardrails=[
            validate_tool_parameters(
                validators={
                    "read_file": validate_file_path,
                    "query_database": validate_sql_query,
                }
            )
        ],
        max_retries=0,
        on_block="raise",
    )

    # Test path traversal attempt
    try:
        print("Query: Read the file at ../../etc/passwd\n")
        await guarded_agent.run("Read the file at ../../etc/passwd")
        print("\n❓ Unexpected success")
    except OutputGuardrailViolation as e:
        print(f"\n✅ Correctly blocked: {e.result.get('message')}")


async def example_combined_validation():
    """Example 3: Combine schemas and custom validators."""
    print("\n" + "=" * 70)
    print("Example 3: Combined Schema + Custom Validation")
    print("=" * 70)
    print("Use both Pydantic schemas AND custom validators together.\n")

    guarded_agent = with_guardrails(
        agent,
        output_guardrails=[
            validate_tool_parameters(
                schemas={"read_file": FileReadParams},
                validators={"read_file": validate_file_path},
            )
        ],
        max_retries=0,
        on_block="raise",
    )

    # Valid path
    try:
        print("Query: Read the file at /tmp/data.txt\n")
        await guarded_agent.run("Read the file at /tmp/data.txt")
        print("\n✅ Success! Both validations passed.")
    except OutputGuardrailViolation as e:
        print(f"\n❌ Blocked: {e.result.get('message')}")


async def example_strict_mode():
    """Example 4: Strict mode - all tools must have validation."""
    print("\n" + "=" * 70)
    print("Example 4: Strict Mode (Production Security)")
    print("=" * 70)
    print("Require validation for ALL tools - block undefined ones.\n")

    guarded_agent = with_guardrails(
        agent,
        output_guardrails=[
            validate_tool_parameters(
                schemas={
                    "get_weather": WeatherParams,
                    # Note: read_file and query_database not defined
                },
                allow_undefined_tools=False,  # Strict mode!
            )
        ],
        max_retries=0,
        on_block="raise",
    )

    # Tool without validation in strict mode
    try:
        print("Query: Query the users table\n")
        await guarded_agent.run("Query the users table")
        print("\n❓ Unexpected success")
    except OutputGuardrailViolation as e:
        print(f"\n✅ Correctly blocked: {e.result.get('message')}")
        print("    (Tool not in validation list + strict mode)")


async def example_sql_injection_prevention():
    """Example 5: SQL injection prevention."""
    print("\n" + "=" * 70)
    print("Example 5: SQL Injection Prevention")
    print("=" * 70)
    print("Detect dangerous SQL patterns in parameters.\n")

    guarded_agent = with_guardrails(
        agent,
        output_guardrails=[
            validate_tool_parameters(
                schemas={"query_database": DatabaseQueryParams},
                validators={"query_database": validate_sql_query},
            )
        ],
        max_retries=0,
        on_block="raise",
    )

    # SQL injection attempt
    try:
        print("Query: Query table 'users; DROP TABLE users--'\n")
        await guarded_agent.run("Query table 'users; DROP TABLE users--'")
        print("\n❓ Unexpected success")
    except OutputGuardrailViolation as e:
        print(f"\n✅ SQL injection blocked: {e.result.get('message')}")


async def example_parameter_limits():
    """Example 6: Enforce parameter limits."""
    print("\n" + "=" * 70)
    print("Example 6: Parameter Limits and Constraints")
    print("=" * 70)
    print("Enforce limits on string length, array size, numeric ranges.\n")

    guarded_agent = with_guardrails(
        agent,
        output_guardrails=[
            validate_tool_parameters(
                schemas={
                    "get_weather": WeatherParams,  # max_length=50
                    "query_database": DatabaseQueryParams,  # limit: 1-1000
                }
            )
        ],
        max_retries=0,
        on_block="raise",
    )

    # Test limit violation
    try:
        print("Query: Get weather for a location with a very long name...\n")
        long_location = "A" * 100
        await guarded_agent.run(
            f"Get weather for {long_location}"
        )
        print("\n❓ Unexpected success")
    except OutputGuardrailViolation as e:
        print(f"\n✅ Correctly blocked: {e.result.get('message')}")
        print("    (Location exceeds 50 character limit)")


async def main():
    """Run all examples."""
    print("\n")
    print("🛡️  Pydantic AI Guardrails - Tool Parameter Validation Examples")
    print("=" * 70)
    print("\nThese examples demonstrate validate_tool_parameters which validates")
    print("tool arguments for security, preventing injection attacks, path traversal,")
    print("and other malicious parameters from reaching tool execution.")
    print("\n📝 NOTE: These examples intentionally test security violations.")
    print("   Watch for blocked calls - that's the guardrail working!")
    print("\n" + "=" * 70)

    # Run examples
    await example_schema_validation()
    await example_custom_validators()
    await example_combined_validation()
    await example_strict_mode()
    await example_sql_injection_prevention()
    await example_parameter_limits()

    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. Pydantic schemas provide rich validation (types, lengths, patterns)")
    print("2. Custom validators handle complex security rules")
    print("3. Combine both for comprehensive parameter validation")
    print("4. Strict mode ensures all tools have validation defined")
    print("5. Prevents path traversal, SQL injection, and other attacks")
    print("6. Use Field() constraints: max_length, pattern, ge/le, etc.")
    print("\n💡 Security tip: Always use strict mode in production!")
    print("   Set allow_undefined_tools=False to require validation for all tools.")


if __name__ == "__main__":
    asyncio.run(main())
