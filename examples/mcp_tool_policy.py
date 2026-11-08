"""Tool policy for MCP toolsets.

MCP servers can expose dozens of tools. Prefix each server's toolset
(``MCPToolset(...).prefixed('github')``) and use RestrictTools globs to allow a server's read-only tools, block destructive
ones, and gate the rest behind approval.

Run with: python examples/mcp_tool_policy.py
(needs an MCP server; swap in your own command and model)
"""

from __future__ import annotations

import asyncio

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset

from pydantic_ai_guardrails import RestrictTools, ValidateToolArgs


def ask_user(tool_name: str, args: dict) -> bool:
    answer = input(f'Allow {tool_name}({args})? [y/N] ')
    return answer.strip().lower() == 'y'


def no_force_push(tool_name: str, args: dict) -> str | None:
    if args.get('force'):
        return 'force operations are not allowed; retry without force=True'
    return None


# Point these at your own MCP servers (URL, script path, or FastMCP client).
github = MCPToolset('http://localhost:3001/mcp').prefixed('github')
fs = MCPToolset('http://localhost:3002/mcp').prefixed('fs')

agent = Agent(
    'openai:gpt-4o',
    toolsets=[github, fs],
    capabilities=[
        RestrictTools(
            # Only these servers' tools are visible at all.
            allowed=['github_*', 'fs_read_*', 'fs_list_*', 'fs_search_*'],
            # Destructive operations are hidden even if allowed above.
            blocked=['github_delete_*', 'github_merge_*'],
            # Writes need a human.
            require_approval=['github_create_*', 'github_push_*', 'github_update_*'],
            approval=ask_user,
        ),
        ValidateToolArgs(check=no_force_push, tools=['github_*']),
    ],
)


async def main() -> None:
    async with agent:
        result = await agent.run('List the open issues in jagreehal/pydantic-ai-guardrails.')
        print(result.output)


if __name__ == '__main__':
    asyncio.run(main())
