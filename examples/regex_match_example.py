"""Regex Match Example

Demonstrates MatchRegex guardrail that validates outputs against regex patterns.
"""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from examples._utils import agent_with_guardrails, get_model_name, guard_allows_output, setup_api_config
from pydantic_ai import Agent

from pydantic_ai_guardrails import MatchRegex, OutputBlocked

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

api_type = setup_api_config()
agent = Agent(
    get_model_name(),
    system_prompt='You are a helpful assistant that provides structured information.',
)


async def example_live_agent() -> None:
    print('\n' + '=' * 70)
    print('Example 0: Live Agent with Regex Validation')
    print('=' * 70)

    guarded_agent = agent_with_guardrails(
        agent,
        output_guardrails=[
            MatchRegex(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                on_fail='raise',
            )
        ],
    )

    try:
        result = await guarded_agent.run('Please provide a contact email for customer support.')
        print('✅ Response contains email:')
        print(f'   {result.output}\n')
    except OutputBlocked as e:
        print(f'❌ Response blocked - no email found: {e}\n')


async def example_single_pattern() -> None:
    print('\n' + '=' * 70)
    print('Example 1: Single Pattern Validation')
    print('=' * 70)

    guardrail = MatchRegex(r'\d{3}-\d{3}-\d{4}')
    for output, should_pass in [
        ('My phone is 555-123-4567', True),
        ('Call me at (555) 123-4567', False),
        ('No contact info here', False),
    ]:
        passed = await guard_allows_output(guardrail, output)
        status = '✅ Passed' if passed else '❌ Blocked'
        expected = ' (as expected)' if passed == should_pass else ' (UNEXPECTED)'
        print(f'Output: {output}')
        print(f'  {status}{expected}\n')


async def example_email_validation() -> None:
    print('\n' + '=' * 70)
    print('Example 2: Email Format Validation')
    print('=' * 70)

    guardrail = MatchRegex(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    for output in [
        'Contact us at support@example.com',
        'Reach me at user.name+tag@company.co.uk',
        'Email me at invalid@',
        'No email address here',
    ]:
        passed = await guard_allows_output(guardrail, output)
        print(f'Output: {output}')
        print(f'  {"✅ Valid email found" if passed else "❌ No valid email"}\n')


async def example_multiple_patterns_or() -> None:
    print('\n' + '=' * 70)
    print('Example 3: Multiple Patterns (OR Logic)')
    print('=' * 70)

    guardrail = MatchRegex(
        patterns=[
            r'\d{3}-\d{3}-\d{4}',
            r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}',
        ],
        require_all=False,
    )
    for output, _should_pass, description in [
        ('Phone: 555-123-4567', True, 'Has phone'),
        ('Email: user@example.com', True, 'Has email'),
        ('No contact info', False, 'Has neither'),
    ]:
        passed = await guard_allows_output(guardrail, output)
        print(f'Case: {description}')
        print(f'  Output: {output}')
        print(f'  {"✅ Passed" if passed else "❌ Blocked"}\n')


async def example_multiple_patterns_and() -> None:
    print('\n' + '=' * 70)
    print('Example 4: Multiple Patterns (AND Logic)')
    print('=' * 70)

    guardrail = MatchRegex(
        patterns=[
            r'\d{3}-\d{3}-\d{4}',
            r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}',
        ],
        require_all=True,
    )
    for output, description in [
        ('Phone: 555-123-4567', 'Only phone'),
        ('Phone: 555-123-4567, Email: user@example.com', 'Has both'),
    ]:
        passed = await guard_allows_output(guardrail, output)
        print(f'Case: {description}')
        print(f'  {"✅ Passed" if passed else "❌ Blocked"}\n')


async def example_named_patterns() -> None:
    print('\n' + '=' * 70)
    print('Example 5: Named Patterns')
    print('=' * 70)

    guardrail = MatchRegex(
        patterns={
            'order_id': r'ORD-\d{6}',
            'status': r'(pending|completed|cancelled)',
            'amount': r'\$\d+\.\d{2}',
        },
        require_all=True,
    )
    for output in [
        'Order ORD-123456 is pending for $99.99',
        'Order ORD-123456 is pending',
    ]:
        passed = await guard_allows_output(guardrail, output)
        print(f'Output: {output}')
        print(f'  {"✅ All patterns found" if passed else "❌ Validation failed"}\n')


async def example_full_match_vs_search() -> None:
    print('\n' + '=' * 70)
    print('Example 6: Full Match vs Search Mode')
    print('=' * 70)

    full_match = MatchRegex(r'PROD-\d{5}', full_match=True)
    search_match = MatchRegex(r'PROD-\d{5}', full_match=False)

    for output in ['PROD-12345', 'Product: PROD-12345']:
        full_passed = await guard_allows_output(full_match, output)
        search_passed = await guard_allows_output(search_match, output)
        print(f"Output: '{output}'")
        print(f"  Full match: {'✅ Pass' if full_passed else '❌ Fail'}")
        print(f"  Search mode: {'✅ Pass' if search_passed else '❌ Fail'}\n")


async def main() -> None:
    print('\n🛡️  Pydantic AI Guardrails - MatchRegex Examples\n')
    print(f'Using {api_type} for live agent example.\n')
    await example_live_agent()
    await example_single_pattern()
    await example_email_validation()
    await example_multiple_patterns_or()
    await example_multiple_patterns_and()
    await example_named_patterns()
    await example_full_match_vs_search()
    print('=' * 70)
    print('Examples complete!')
    print('=' * 70)


if __name__ == '__main__':
    asyncio.run(main())
