---
name: github-code-review
description: Review GitHub pull requests for correctness, style, and security issues
version: 1.0.0
metadata:
  hermes:
    tags: [code-review, github, security]
---

# GitHub Code Review Skill

## When to Use
Use this skill when you need to review a GitHub pull request for:
- Correctness bugs
- Code style issues
- Security vulnerabilities
- Performance problems

## Procedure
1. Fetch the PR diff using the GitHub API
2. Analyze changed files for:
   - Logic errors and edge cases
   - Security vulnerabilities (SQL injection, XSS, etc.)
   - Performance anti-patterns
   - Code style violations
3. Provide structured feedback with:
   - File and line references
   - Severity (critical/high/medium/low)
   - Suggested fixes
4. Summarize overall assessment

## Pitfalls
- Don't flag style preferences as bugs
- Consider the project's existing patterns
- Don't suggest changes that break backward compatibility
- Verify suggestions actually fix the issue