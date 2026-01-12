import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  site: 'https://jagreehal.github.io',
  // Use base path for GitHub Pages deployment
  // For local development, you can override with: BASE=/ pnpm dev
  base: process.env.BASE || '/pydantic-ai-guardrails',
  integrations: [
    starlight({
      title: 'Pydantic AI Guardrails',
      description: 'Production-ready guardrails for Pydantic AI agents',
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
            { label: 'Parallel Execution', slug: 'guides/parallel-execution' },
            { label: 'Error Handling', slug: 'guides/error-handling' },
            { label: 'Human-in-the-Loop', slug: 'guides/human-in-the-loop' },
          ],
        },
        {
          label: 'Built-in Guardrails',
          items: [
            { label: 'Overview', slug: 'guardrails/overview' },
            {
              label: 'Input Guardrails',
              collapsed: true,
              items: [
                { label: 'Length Limit', slug: 'guardrails/input/length-limit' },
                { label: 'PII Detector', slug: 'guardrails/input/pii-detector' },
                { label: 'Prompt Injection', slug: 'guardrails/input/prompt-injection' },
                { label: 'Toxicity', slug: 'guardrails/input/toxicity' },
                { label: 'Blocked Keywords', slug: 'guardrails/input/blocked-keywords' },
                { label: 'Rate Limit', slug: 'guardrails/input/rate-limit' },
              ],
            },
            {
              label: 'Output Guardrails',
              collapsed: true,
              items: [
                { label: 'Secret Redaction', slug: 'guardrails/output/secret-redaction' },
                { label: 'LLM Judge', slug: 'guardrails/output/llm-judge' },
                { label: 'JSON Validator', slug: 'guardrails/output/json-validator' },
                { label: 'Regex Match', slug: 'guardrails/output/regex-match' },
                { label: 'No Refusals', slug: 'guardrails/output/no-refusals' },
                { label: 'Tool Validation', slug: 'guardrails/output/tool-validation' },
              ],
            },
          ],
        },
        {
          label: 'Integrations',
          items: [
            { label: 'Logfire', slug: 'integrations/logfire' },
            { label: 'llm-guard', slug: 'integrations/llm-guard' },
            { label: 'autoevals', slug: 'integrations/autoevals' },
            { label: 'Pydantic Evals', slug: 'integrations/pydantic-evals' },
          ],
        },
        {
          label: 'Configuration',
          items: [
            { label: 'JSON/YAML Config', slug: 'configuration/json-yaml' },
            { label: 'OpenAI Format', slug: 'configuration/openai-format' },
          ],
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
