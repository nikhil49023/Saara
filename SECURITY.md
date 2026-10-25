# Security Policy

## Supported versions
Security fixes are released for the latest minor version of `saara-ai` on PyPI.

## Reporting a vulnerability
Please **do not** open a public issue. Report it privately through
[GitHub Security Advisories](https://github.com/nikhil49023/Saara/security/advisories/new) with steps to
reproduce, the affected version (`saara --version`) and the impact. You will get an acknowledgement within
72 hours, and a fix or mitigation plan as soon as the issue is confirmed.

## Scope notes
- Saara treats retrieved web and document content as data, not instructions; prompt-injection bypasses are in scope.
- Never put API keys in workflow files that you commit; pass them with `--api-key` or environment variables.
