---
paths:
  - "py_igesio/**"
---

## Mandatory Rules for Claude Code

- Do NOT modify any files unless explicitly instructed.
- Do NOT refactor existing code unless clearly requested.
- Prefer minimal, localized changes over large improvements.
- Stability and existing behavior are more important than code cleanliness.

## Change Proposal Requirement

Before making any code changes:
- Explain what will be changed and why it is necessary
- Describe potential risks or side effects

**Wait for explicit approval before proceeding**.

### File Modification Policy

- IMPORTANT: Only modify files explicitly listed by the user (do NOT touch files not mentioned).
- Ask for confirmation before expanding the scope.
- Do NOT perform any Git operations that affect branches other than the feature/fix branch.

## Security Rules

- Never request or output (log, print ...) secrets, credentials, or personal data.
- Assume production-like constraints even in development.

## Cost Awareness

- Keep responses concise.
- Avoid repeating large code blocks unless necessary.
- Prefer explanation over full implementation when possible.
- Use the Default (recommended) model for all tasks.
  - The Default model is currently Sonnet 4.6.
  - Do NOT switch to Opus unless explicitly instructed by the user.
