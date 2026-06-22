---
name: github-code-review
description: "Review PRs: diffs, inline comments via gh or REST."
version: 1.1.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [GitHub, Code-Review, Pull-Requests, Git, Quality]
    related_skills: [github-auth, github-pr-workflow]
---

Perform code reviews on local changes before pushing, or review open PRs on GitHub. This process uses `git`, the GitHub CLI (`gh`), or `curl` against the GitHub API.

---

## CRITICAL IMPLEMENTATION RULES & PITFALLS TO AVOID

1. **Strict Bash Syntax & Curl JSON Escaping:**
   - Never use literal `\n` characters in shell commands to denote newlines. Use actual multi-line shell formatting or backslashes `\` for line continuation.
   - When using `curl -d` with JSON payloads, **avoid nested double-quotes** which break bash shell parsing. 
   - **Preferred Method:** Use a shell Heredoc to pass clean JSON payloads to `curl`:
     ```bash
     curl -s -X POST \
       -H "Authorization: token $GITHUB_TOKEN" \
       -H "Content-Type: application/json" \
       https://api.github.com/repos/$OWNER/$REPO/pulls/$PR_NUMBER/reviews \
       --data-binary @- <<EOF
     {
       "commit_id": "$HEAD_SHA",
       "event": "REQUEST_CHANGES",
       "body": "Please fix the memory leak in connection pooling before merge."
     }
     EOF
     ```

2. **Proactiveness (No Wasteful Discovery):**
   - Do not waste interactions running generic discovery commands like `git remote -v` or `ls -F` if the intent is clear. Go straight to the action.
   - If asked to check out a PR, execute the checkout commands immediately.
   - If asked to review staged changes, run the diff, read the files, analyze the code, and print the final review summary in a single workflow. Do not output a single CLI command and pause.

3. **Strict Template Adherence:**
   - Always output the code review summary in the designated markdown structure. Do not omit categories; if a category has no items, omit it or write "None".

---

## 1. Setup & Environment Authentication

Prioritize using `gh` if authenticated. Otherwise, fallback to retrieving `GITHUB_TOKEN` and remote owner/repo details via `git`.

```bash
# Setup authentication and repository variables
if command -v gh &>/dev/null && gh auth status &>/dev/null; then
  AUTH="gh"
else
  AUTH="git"
  if [ -z "$GITHUB_TOKEN" ]; then
    if _hermes_env="${HERMES_HOME:-$HOME/.hermes}/.env"; [ -f "$_hermes_env" ] && grep -q "^GITHUB_TOKEN=" "$_hermes_env"; then
      GITHUB_TOKEN=$(grep "^GITHUB_TOKEN=" "$_hermes_env" | head -1 | cut -d= -f2 | tr -d '\n\r')
    elif grep -q "github.com" ~/.git-credentials 2>/dev/null; then
      GITHUB_TOKEN=$(grep "github.com" ~/.git-credentials 2>/dev/null | head -1 | sed 's|https://[^:]*:\([^@]*\)@.*|\1|')
    fi
  fi
fi

REMOTE_URL=$(git remote get-url origin)
OWNER_REPO=$(echo "$REMOTE_URL" | sed -E 's|.*github\.com[:/]||; s|\.git$||')
OWNER=$(echo "$OWNER_REPO" | cut -d/ -f1)
REPO=$(echo "$OWNER_REPO" | cut -d/ -f2)
```

---

## 2. Reviewing Local Changes (Pre-Push)

### Core Command Flow
1. Fetch and view local changes:
   ```bash
   # Staged changes (what is ready to commit)
   git diff --staged
   ```
2. Scan for common issues:
   ```bash
   # Debug statements, TODOs, or credentials left behind
   git diff --staged | grep -n "print(\|console\.log\|TODO\|FIXME\|password\|secret\|api_key\|<<<<<<\|======="
   ```
3. Format the final output precisely using the template below.

### Review Output Format
When reviewing local changes, present findings in this structure:

```markdown
## Code Review Summary

### Critical
- **[File path]:[Line]** — Description of critical issue (e.g. security vulnerability, syntax error).
  Suggestion: [How to fix it]

### Warnings
- **[File path]:[Line]** — Code smells, performance pitfalls, or style issues.

### Suggestions
- **[File path]** — Architectural feedback, consolidation, or cleanup.

### Looks Good
- [Specific highlight of what was implemented well]
```

---

## 3. Reviewing a Pull Request on GitHub

### Step 1: Checkout the PR immediately
If asked to check out PR #N, run the checkout command directly:
```bash
# With gh:
gh pr checkout N

# With git:
git fetch origin pull/N/head:pr-N
git checkout pr-N
```

### Step 2: Run Tests & Linters
Do not guess if tests pass. Automatically find and run them:
```bash
# Examples of automated test running based on project files:
# Python: pytest
# Node.js: npm test
# Rust: cargo test
```

### Step 3: Leave Comments / Request Changes / Approve
To request changes or leave a review, use the robust JSON heredoc style:

```bash
# 1. Get HEAD SHA of the PR
if [ "$AUTH" = "gh" ]; then
  HEAD_SHA=$(gh pr view $PR_NUMBER --json headRefOid --jq '.headRefOid')
else
  HEAD_SHA=$(curl -s -H "Authorization: token $GITHUB_TOKEN" \
    https://api.github.com/repos/$OWNER/$REPO/pulls/$PR_NUMBER \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['head']['sha'])")
fi

# 2. Submit the review
if [ "$AUTH" = "gh" ]; then
  gh pr review $PR_NUMBER --request-changes --body "Please fix the memory leak in connection pooling before merge."
else
  curl -s -X POST \
    -H "Authorization: token $GITHUB_TOKEN" \
    -H "Content-Type: application/json" \
    https://api.github.com/repos/$OWNER/$REPO/pulls/$PR_NUMBER/reviews \
    --data-binary @- <<EOF
{
  "commit_id": "$HEAD_SHA",
  "event": "REQUEST_CHANGES",
  "body": "Please fix the memory leak in connection pooling before merge."
}
EOF
fi
```

---

## 4. Review Checklist

Verify code against these standards:
- **Correctness:** Handling of edge cases, empty inputs, nulls, error paths.
- **Security:** No hardcoded secrets, credentials, SQL injections, or unsafe path traversals.
- **Code Quality:** Single responsibility functions, DRY principles, clear naming.
- **Testing:** Ensure appropriate tests are run and passing.
