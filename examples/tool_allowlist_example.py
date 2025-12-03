"""Tool Allowlist Example

Demonstrates the tool_allowlist guardrail that restricts which tools can
be called by the agent. This provides security through whitelisting only
approved/safe tools, preventing unauthorized operations.

This example shows:
1. Basic tool allowlisting (single and multiple tools)
2. Role-based access control (admin, user, guest)
3. Security sandboxing (safe vs dangerous tools)
4. Blocking unauthorized tool attempts
5. Combining with other tool guardrails
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from examples._utils import get_model_name
from pydantic_ai import Agent, RunContext

from pydantic_ai_guardrails import GuardedAgent, OutputGuardrailViolation
from pydantic_ai_guardrails.guardrails.output import (
    require_tool_use,
    tool_allowlist,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


# Define mock tools for demonstration (safe and dangerous)
def search_web(ctx: RunContext[None], query: str) -> str:
    """Search the web (SAFE - read-only)."""
    print(f"  🔍 search_web called: query={query}")
    return f"Search results for: {query}"


def get_weather(ctx: RunContext[None], location: str) -> str:
    """Get weather (SAFE - read-only)."""
    print(f"  🌤️  get_weather called: location={location}")
    return f"Weather in {location}: Sunny"


def read_file(ctx: RunContext[None], path: str) -> str:
    """Read file (MODERATE RISK - read filesystem)."""
    print(f"  📁 read_file called: path={path}")
    return f"Contents of {path}"


def write_file(ctx: RunContext[None], path: str, content: str) -> str:
    """Write file (HIGH RISK - modifies filesystem)."""
    print(f"  ✍️  write_file called: path={path}")
    return f"Wrote to {path}"


def execute_code(ctx: RunContext[None], code: str) -> str:
    """Execute code (DANGEROUS - arbitrary code execution)."""
    print(f"  ⚠️  execute_code called: code={code}")
    return "Code executed"


def send_email(ctx: RunContext[None], to: str, subject: str) -> str:
    """Send email (HIGH RISK - external communication)."""
    print(f"  📧 send_email called: to={to}, subject={subject}")
    return f"Email sent to {to}"


def delete_data(ctx: RunContext[None], table: str) -> str:
    """Delete data (DANGEROUS - data loss)."""
    print(f"  🗑️  delete_data called: table={table}")
    return f"Deleted from {table}"


# Create agent with ALL tools (we'll restrict via guardrails)
agent = Agent(
    get_model_name(),
    system_prompt=(
        "You are a helpful assistant with access to various tools. "
        "Use the appropriate tools to answer questions."
    ),
    tools=[
        search_web,
        get_weather,
        read_file,
        write_file,
        execute_code,
        send_email,
        delete_data,
    ],
)


async def example_basic_allowlist():
    """Example 1: Basic allowlisting - single tool."""
    print("\n" + "=" * 70)
    print("Example 1: Basic Allowlist (Single Tool)")
    print("=" * 70)
    print("Only allow search_web tool.\n")

    guarded_agent = GuardedAgent(
        agent,
        output_guardrails=[tool_allowlist(allowed_tools="search_web")],
        max_retries=0,
        on_block="raise",
    )

    # Test allowed tool
    try:
        print("Query: Search for Python tutorials\n")
        await guarded_agent.run("Search for Python tutorials")
        print("\n✅ Success! search_web is allowed.")
    except OutputGuardrailViolation as e:
        print(f"\n❌ Blocked: {e.result.get('message')}")


async def example_multiple_allowed_tools():
    """Example 2: Multiple allowed tools."""
    print("\n" + "=" * 70)
    print("Example 2: Multiple Allowed Tools")
    print("=" * 70)
    print("Allow search_web AND get_weather (read-only tools).\n")

    guarded_agent = GuardedAgent(
        agent,
        output_guardrails=[
            tool_allowlist(allowed_tools=["search_web", "get_weather"])
        ],
        max_retries=0,
        on_block="raise",
    )

    try:
        print("Query: Search for weather APIs and check weather in Paris\n")
        await guarded_agent.run(
            "Search for weather APIs and check weather in Paris"
        )
        print("\n✅ Success! Both tools are allowed.")
    except OutputGuardrailViolation as e:
        print(f"\n❌ Blocked: {e.result.get('message')}")


async def example_block_dangerous_tools():
    """Example 3: Block dangerous tools."""
    print("\n" + "=" * 70)
    print("Example 3: Block Dangerous Tools")
    print("=" * 70)
    print("Attempt to use execute_code (not in allowlist).\n")

    guarded_agent = GuardedAgent(
        agent,
        output_guardrails=[
            tool_allowlist(allowed_tools=["search_web", "get_weather"])
        ],
        max_retries=0,
        on_block="raise",
    )

    try:
        print("Query: Execute Python code to calculate 2+2\n")
        await guarded_agent.run("Execute Python code to calculate 2+2")
        print("\n❓ Unexpected success")
    except OutputGuardrailViolation as e:
        print(f"\n✅ Correctly blocked: {e.result.get('message')}")
        print(f"    Unauthorized tools: {e.result.get('metadata', {}).get('unauthorized_tools', [])}")


async def example_role_based_access_admin():
    """Example 4a: Role-based access - Admin role."""
    print("\n" + "=" * 70)
    print("Example 4a: Role-Based Access - Admin Role")
    print("=" * 70)
    print("Admin has full access to all tools.\n")

    admin_allowlist = [
        "search_web",
        "get_weather",
        "read_file",
        "write_file",
        "execute_code",
        "send_email",
        "delete_data",
    ]

    guarded_agent = GuardedAgent(
        agent,
        output_guardrails=[tool_allowlist(allowed_tools=admin_allowlist)],
        max_retries=0,
        on_block="raise",
    )

    try:
        print("Query: Write a file with test data\n")
        await guarded_agent.run("Write a file with test data")
        print("\n✅ Admin can use write_file (full access)")
    except OutputGuardrailViolation as e:
        print(f"\n❌ Blocked: {e.result.get('message')}")


async def example_role_based_access_user():
    """Example 4b: Role-based access - User role."""
    print("\n" + "=" * 70)
    print("Example 4b: Role-Based Access - User Role")
    print("=" * 70)
    print("User has read-only access (no write/execute).\n")

    user_allowlist = ["search_web", "get_weather", "read_file"]

    guarded_agent = GuardedAgent(
        agent,
        output_guardrails=[tool_allowlist(allowed_tools=user_allowlist)],
        max_retries=0,
        on_block="raise",
    )

    try:
        print("Query: Delete old user data\n")
        await guarded_agent.run("Delete old user data")
        print("\n❓ Unexpected success")
    except OutputGuardrailViolation as e:
        print(f"\n✅ Correctly blocked: {e.result.get('message')}")
        print("    Users cannot delete data (permission denied)")


async def example_role_based_access_guest():
    """Example 4c: Role-based access - Guest role."""
    print("\n" + "=" * 70)
    print("Example 4c: Role-Based Access - Guest Role")
    print("=" * 70)
    print("Guest has minimal access (search only).\n")

    guest_allowlist = ["search_web"]

    guarded_agent = GuardedAgent(
        agent,
        output_guardrails=[tool_allowlist(allowed_tools=guest_allowlist)],
        max_retries=0,
        on_block="raise",
    )

    try:
        print("Query: Read the configuration file\n")
        await guarded_agent.run("Read the configuration file")
        print("\n❓ Unexpected success")
    except OutputGuardrailViolation as e:
        print(f"\n✅ Correctly blocked: {e.result.get('message')}")
        print("    Guests cannot read files (minimal access)")


async def example_security_sandbox():
    """Example 5: Security sandbox - development environment."""
    print("\n" + "=" * 70)
    print("Example 5: Security Sandbox (Development/Testing)")
    print("=" * 70)
    print("Restrict to safe tools during testing.\n")

    # In dev/test, only allow safe, non-destructive tools
    safe_tools = ["search_web", "get_weather", "read_file"]

    guarded_agent = GuardedAgent(
        agent,
        output_guardrails=[tool_allowlist(allowed_tools=safe_tools)],
        max_retries=0,
        on_block="raise",
    )

    try:
        print("Query: Send email notification\n")
        await guarded_agent.run("Send email notification")
        print("\n❓ Unexpected success")
    except OutputGuardrailViolation as e:
        print(f"\n✅ Correctly blocked: {e.result.get('message')}")
        print("    Email sending disabled in test environment")


async def example_combined_with_require_tool_use():
    """Example 6: Combine allowlist with require_tool_use."""
    print("\n" + "=" * 70)
    print("Example 6: Combined Guardrails (Allowlist + Require Use)")
    print("=" * 70)
    print("Ensure tool IS used AND it's an allowed tool.\n")

    guarded_agent = GuardedAgent(
        agent,
        output_guardrails=[
            tool_allowlist(allowed_tools=["search_web", "get_weather"]),
            require_tool_use(),  # Must use at least one tool
        ],
        max_retries=0,
        on_block="raise",
    )

    try:
        print("Query: What's the capital of France?\n")
        await guarded_agent.run("What's the capital of France?")
        print("\n✅ Success! Tool used and allowed.")
    except OutputGuardrailViolation as e:
        print(f"\n❌ Blocked: {e.result.get('message')}")
        print("    (Either no tool used OR unauthorized tool)")


async def example_security_violation_details():
    """Example 7: Detailed security violation information."""
    print("\n" + "=" * 70)
    print("Example 7: Security Violation Details")
    print("=" * 70)
    print("Examine detailed information when tools are blocked.\n")

    guarded_agent = GuardedAgent(
        agent,
        output_guardrails=[
            tool_allowlist(allowed_tools=["search_web", "get_weather"])
        ],
        max_retries=0,
        on_block="raise",
    )

    try:
        print("Query: Execute code to delete user_data table\n")
        await guarded_agent.run("Execute code to delete user_data table")
        print("\n❓ Unexpected success")
    except OutputGuardrailViolation as e:
        print("\n🛡️  Security Violation Detected!")
        print(f"    Message: {e.result.get('message')}")
        print(f"    Severity: {e.result.get('severity')}")
        print(f"    Allowed tools: {e.result.get('metadata', {}).get('allowed_tools', [])}")
        print(f"    Tools attempted: {e.result.get('metadata', {}).get('tools_called', [])}")
        print(f"    Unauthorized: {e.result.get('metadata', {}).get('unauthorized_tools', [])}")
        print(f"    Suggestion: {e.result.get('suggestion')}")


async def main():
    """Run all examples."""
    print("\n")
    print("🛡️  Pydantic AI Guardrails - Tool Allowlist Examples")
    print("=" * 70)
    print("\nThese examples demonstrate tool_allowlist which restricts agent")
    print("capabilities to a whitelist of approved tools, providing security")
    print("through least-privilege access control.")
    print("\n📝 NOTE: These examples intentionally test unauthorized access.")
    print("   Watch for blocked calls - that's the security working!")
    print("\n" + "=" * 70)

    # Run examples
    await example_basic_allowlist()
    await example_multiple_allowed_tools()
    await example_block_dangerous_tools()
    await example_role_based_access_admin()
    await example_role_based_access_user()
    await example_role_based_access_guest()
    await example_security_sandbox()
    await example_combined_with_require_tool_use()
    await example_security_violation_details()

    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. Allowlists enforce least-privilege access (whitelist only safe tools)")
    print("2. Role-based access: Different allowlists for admin/user/guest")
    print("3. Security sandboxing: Restrict tools in dev/test environments")
    print("4. Combine with require_tool_use for complete control")
    print("5. Detailed violation info helps with security auditing")
    print("6. Prevents unauthorized operations (code execution, data deletion)")
    print("\n💡 Security tip: Always use the minimum set of tools needed!")
    print("   Start with restrictive allowlists and expand only when necessary.")
    print("\n🔐 Tool Security Suite:")
    print("   • tool_allowlist() - Restrict WHICH tools can be called")
    print("   • validate_tool_parameters() - Validate tool ARGUMENTS")
    print("   • require_tool_use() - Ensure tools ARE used")


if __name__ == "__main__":
    asyncio.run(main())
