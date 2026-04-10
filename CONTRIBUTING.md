# Contributing to oxidized-transport

Thank you for your interest in contributing! This document explains the process.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Development Lifecycle](#development-lifecycle)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Design Principles](#design-principles)
- [Commit Style](#commit-style)
- [Pull Request Process](#pull-request-process)
- [Testing](#testing)
- [Release Process](#release-process)
- [Continuous Improvement](#continuous-improvement)

---

## Code of Conduct

This project follows the [Contributor Covenant](./CODE_OF_CONDUCT.md).
Be respectful and constructive.

---

## Development Lifecycle

Every change follows a structured lifecycle:

```
Identify → Research → Decide → Plan → Test First → Implement → Review → Integrate → Retrospect
```

The lifecycle ensures that we:

- Understand the problem before writing code
- Write tests before implementation (TDD)
- Actively identify improvements during review

For trivial changes (typo fixes, dependency bumps), an abbreviated lifecycle applies.

---

## How to Contribute

| Type | How |
|---|---|
| 🐛 Bug report | Open a [bug report issue](.github/ISSUE_TEMPLATE/bug_report.yml) |
| 💡 Feature request | Open a [feature request issue](.github/ISSUE_TEMPLATE/feature_request.yml) |
| 📖 Docs | Edit any `.md` file and open a PR |
| 🧩 Implementation | Pick an open issue and open a PR |
| 🔍 Review | Review open PRs and leave constructive feedback |
| 💡 Improvement | Found a better approach? Open an issue or PR |

---

## Development Setup

```bash
# 1. Fork and clone
git clone https://github.com/oxidized-mc/transport.git
cd transport

# 2. Rust stable (toolchain pinned via rust-toolchain.toml)
rustup update stable

# 3. Build
cargo build

# 4. Run tests
cargo test

# 5. Check formatting and lints
cargo fmt --check
cargo clippy --all-targets -- -D warnings
```

### Useful tools (optional)

```bash
cargo install cargo-deny    # licence + advisory checks
cargo install cargo-nextest # faster test runner
cargo install cargo-watch   # auto-rebuild on save
```

---

## Design Principles

`oxidized-transport` is a library crate in the
[Oxidized MC](https://github.com/oxidized-mc) ecosystem. Key principles:

- **API stability matters** — public types may be consumed by downstream
  crates. Think carefully before changing public signatures.
- **Wire compatibility** — types that touch the Minecraft protocol must produce
  byte-identical output to vanilla Java Edition 26.1.
- **Documentation required** — all public items must have `///` doc comments.
  The crate enforces `#![warn(missing_docs)]`.
- **No unsafe code** — the crate enforces `#![deny(unsafe_code)]`.
- **Idiomatic Rust** — when implementing logic from the vanilla Java source,
  rewrite idiomatically rather than transliterating Java line-by-line.

---

## Commit Style

All commits **must** follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short description>
```

### Types

| Type | When | Version bump |
|---|---|---|
| `feat` | New user-visible feature | Minor |
| `fix` | Bug fix | Patch |
| `perf` | Performance improvement | Patch |
| `refactor` | Restructure, no behaviour change | None |
| `test` | Tests only | None |
| `docs` | Documentation only | None |
| `chore` | Dependencies, CI, tooling | None |
| `ci` | CI/CD workflow changes | None |

### Scope

Use `transport` as the scope for all changes to this crate.
Use `ci` for workflow files, `deps` for dependency updates.

---

## Pull Request Process

1. Create a branch: `git checkout -b feat/my-change`
2. Make your changes following the lifecycle
3. Commit with conventional commits
4. Open a PR targeting `main`; fill in the PR template completely
5. At least one approving review is required before merge
6. Squash-merge preferred for feature branches

### PR Review Standards

Reviews are not just about catching bugs — they actively seek improvements:
- Does the code follow the project's design principles?
- Could any existing pattern be improved?
- Are there learnings to record?

---

## Testing

- **Unit tests** live next to the code in `#[cfg(test)]` modules
- **Integration tests** live in `tests/`
- **Property-based tests** use `proptest` for roundtrips, invariants, and edge cases
- All public API must have at least one test
- Test modules use `#[allow(clippy::unwrap_used, clippy::expect_used)]`

Run with nextest for faster feedback:
```bash
cargo nextest run
```

---

## Release Process

This crate uses automated versioning and release management based on
conventional commits.

### How It Works

1. **Commit with conventional prefixes** — `feat`, `fix`, `perf`, etc.
2. **release-please** automatically creates and maintains a "Release PR" on
   GitHub that accumulates changes and proposes the next version bump.
3. When a maintainer merges the Release PR, a git tag (`v0.X.Y`) and GitHub
   Release are created automatically.
4. **git-cliff** generates the changelog from conventional commit messages.

### Version Bump Rules

| Commit prefix | Bump |
|--------------|------|
| `feat!:` or `BREAKING CHANGE:` footer | Minor (pre-1.0) / Major (post-1.0) |
| `feat(scope):` | Minor |
| `fix(scope):`, `perf(scope):` | Patch |
| `refactor`, `test`, `docs`, `chore`, `ci` | No version bump |

---

## Continuous Improvement

We believe the codebase should always be getting better.

**After every milestone:**
- Conduct a retrospective
- Record learnings and identify improvements
- Record any technical debt incurred

**During every PR review:**
- Identify outdated patterns or decisions
- Look for patterns that should be extracted or formalized
- Suggest improvements (not just catch bugs)

**When you find something better:**
- Don't just note it — act on it
- Open an issue or PR
- Plan and execute the refactoring
