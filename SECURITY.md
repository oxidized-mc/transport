# Security Policy

## Supported Versions

| Version | Supported |
|---------|----------|
| `main` (latest) | ✅ |
| older releases | ❌ |

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Report security issues by using
[GitHub's private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing/privately-reporting-a-security-vulnerability)
on this repository.

Include:
- A description of the vulnerability and its impact
- Steps to reproduce (proof-of-concept if possible)
- Affected versions/commits

We will acknowledge your report within **72 hours** and aim to release a fix
within **7 days** for critical issues.

## Scope

This project is a **library crate**. Relevant vulnerability classes:

- Connection state machine vulnerabilities (invalid state transitions)
- Parsing vulnerabilities (malformed frames causing panics or crashes)
- Memory safety issues (should be none — the crate enforces `#![deny(unsafe_code)]`)
- Denial of service via connection flooding or oversized frames

Out of scope: game logic bugs, upstream dependency issues (report those to the
dependency maintainer directly).
