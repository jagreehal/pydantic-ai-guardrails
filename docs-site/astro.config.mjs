import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import tailwindcss from '@tailwindcss/vite';

const base = process.env.BASE || '/pydantic-ai-guardrails';
const isLocalDev = base === '/';

export default defineConfig({
  // Always set site URL - Astro will automatically prepend base path to URLs
  // For local dev with BASE=/, links will be relative (not absolute)
  site: isLocalDev ? undefined : 'https://jagreehal.github.io',
  // Use base path for GitHub Pages deployment
  // For local development, you can override with: BASE=/ pnpm dev
  base,
  integrations: [
    starlight({
      title: 'Pydantic AI Guardrails',
      description: 'Guardrails for Pydantic AI agents, built as native capabilities',
      social: [
        {
          icon: 'github',
          label: 'GitHub',
          href: 'https://github.com/jagreehal/pydantic-ai-guardrails',
        },
      ],
      editLink: {
        baseUrl: 'https://github.com/jagreehal/pydantic-ai-guardrails/edit/main/docs-site/',
      },
      head: [
        {
          tag: 'meta',
          attrs: {
            property: 'og:image',
            content: '/og-image.png',
          },
        },
      ],
      sidebar: [
        {
          label: 'Getting Started',
          items: [
            { label: 'Installation', slug: 'getting-started/installation' },
            { label: 'Quick Start', slug: 'getting-started/quick-start' },
          ],
        },
        {
          label: 'Guides',
          items: [
            { label: 'Input Guardrails', slug: 'guides/input-guardrails' },
            { label: 'Output Guardrails', slug: 'guides/output-guardrails' },
            { label: 'Custom Guardrails', slug: 'guides/custom-guardrails' },
            { label: 'Auto-Retry', slug: 'guides/auto-retry' },
            { label: 'Async Guardrails', slug: 'guides/async-guardrails' },
            { label: 'Streaming', slug: 'guides/streaming' },
            { label: 'Multiple Agents', slug: 'guides/multi-agent' },
            { label: 'Error Handling', slug: 'guides/error-handling' },
            { label: 'Human-in-the-Loop', slug: 'guides/human-in-the-loop' },
          ],
        },
        {
          label: 'Built-in Guardrails',
          items: [
            { label: 'Overview', slug: 'guardrails/overview' },
            {
              label: 'Input',
              collapsed: true,
              items: [
                { label: 'DetectPII', slug: 'guardrails/input/detect-pii' },
                { label: 'DetectPromptInjection', slug: 'guardrails/input/detect-prompt-injection' },
                { label: 'DetectToxicity', slug: 'guardrails/input/detect-toxicity' },
                { label: 'BlockKeywords', slug: 'guardrails/input/block-keywords' },
                { label: 'LimitInputLength', slug: 'guardrails/input/limit-input-length' },
                { label: 'RateLimit', slug: 'guardrails/input/rate-limit' },
              ],
            },
            {
              label: 'Output',
              collapsed: true,
              items: [
                { label: 'LimitOutputLength', slug: 'guardrails/output/limit-output-length' },
                { label: 'RedactSecrets', slug: 'guardrails/output/redact-secrets' },
                { label: 'ValidateJson', slug: 'guardrails/output/validate-json' },
                { label: 'FilterToxicity', slug: 'guardrails/output/filter-toxicity' },
                { label: 'DetectHallucination', slug: 'guardrails/output/detect-hallucination' },
                { label: 'MatchRegex', slug: 'guardrails/output/match-regex' },
                { label: 'BlockRefusals', slug: 'guardrails/output/block-refusals' },
                { label: 'RequireToolUse', slug: 'guardrails/output/require-tool-use' },
                { label: 'LlmJudge', slug: 'guardrails/output/llm-judge' },
                { label: 'RequireGrounding', slug: 'guardrails/output/require-grounding' },
              ],
            },
            {
              label: 'Tools & Cost',
              collapsed: true,
              items: [
                { label: 'RestrictTools', slug: 'guardrails/tools/restrict-tools' },
                { label: 'ValidateToolArgs', slug: 'guardrails/tools/validate-tool-args' },
                { label: 'LimitCost', slug: 'guardrails/cost/limit-cost' },
              ],
            },
          ],
        },
        {
          label: 'Integrations',
          items: [
            { label: 'OpenTelemetry & Logfire', slug: 'integrations/opentelemetry' },
            { label: 'Audit Log', slug: 'integrations/audit-log' },
            { label: 'llm-guard', slug: 'integrations/llm-guard' },
            { label: 'OpenAI Moderation', slug: 'integrations/openai-moderation' },
            { label: 'autoevals', slug: 'integrations/autoevals' },
            { label: 'Pydantic Evals', slug: 'integrations/pydantic-evals' },
          ],
        },
        {
          label: 'Configuration',
          items: [{ label: 'JSON & YAML Config', slug: 'configuration/json-yaml' }],
        },
        {
          label: 'Testing',
          slug: 'testing',
        },
        {
          label: 'Reference',
          items: [
            { label: 'API Reference', slug: 'reference/api' },
            { label: 'Exceptions', slug: 'reference/exceptions' },
          ],
        },
      ],
      customCss: ['./src/styles/custom.css'],
    }),
  ],
  vite: {
    plugins: [tailwindcss()],
  },
});
