## Summary

<!-- One-sentence description of what this PR does. -->

## Motivation

<!-- Why is this change needed? Link to relevant issue(s): Closes #xxx -->

## Changes

<!-- Bullet list of what changed. -->

-
-

## Reference

<!-- Did you check the vanilla Java reference? Paste the relevant class/method. -->

<details>
<summary>Java reference (if applicable)</summary>

```java
// mc-server-ref/decompiled/net/minecraft/...
```

</details>

## Testing

<!-- How did you test this? Unit tests added? Property-based tests? -->

- [ ] Unit tests added / updated
- [ ] `cargo test` passes
- [ ] `cargo clippy --all-targets -- -D warnings` passes
- [ ] `cargo fmt --check` passes

## Quality Checklist

<!-- Standard code quality checks. All must pass. -->

- [ ] Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)
- [ ] Public API has documentation (`///` doc comments with `# Errors` sections)
- [ ] No `unwrap()` / `expect()` in production paths (use `?` or proper error handling)
- [ ] No hardcoded magic numbers — use named constants
- [ ] CHANGELOG.md updated if user-visible change

## Architecture Compliance

<!-- Verify this change respects existing architecture. -->

- [ ] No upward imports (lower-tier crates must not depend on higher-tier)
- [ ] Error handling uses `thiserror` / `?` propagation (no `unwrap` in production)

## Continuous Improvement

<!-- Every PR is an opportunity to make the project better. -->

- [ ] **Checked:** Could any existing patterns be improved?
- [ ] **Checked:** Are there stale references (renamed items, moved files, changed APIs)?
- [ ] Any identified improvements are recorded (new issue or PR)
