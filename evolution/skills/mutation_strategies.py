"""
Mutation Strategies for Skill Evolution.

Different skills benefit from different types of mutations.
This module provides skill-specific mutation strategies that GEPA can use.
"""

import re
import random
from typing import Callable, List
from pathlib import Path


# =============================================================================
# Generic Mutation Strategies (apply to any skill)
# =============================================================================

def add_clarity_instruction(content: str, skill_name: str) -> str:
    """Add a clarity/precision instruction to the skill."""
    lines = content.splitlines()
    
    # Find a good place to insert (after "## Steps" or similar)
    insert_idx = len(lines)
    for i, line in enumerate(lines):
        if line.strip().startswith("## Steps") or line.strip().startswith("## Procedure"):
            insert_idx = i + 1
            break
    
    new_instruction = "\n- Be precise and specific in your analysis. Avoid vague language."
    
    # Check if already present
    if new_instruction.strip() not in "\n".join(lines):
        lines.insert(insert_idx, new_instruction)
    
    return "\n".join(lines)


def add_conciseness_instruction(content: str, skill_name: str) -> str:
    """Add instruction to be concise and token-efficient."""
    insertion = "\n- Be concise. Every token counts. Prefer bullet points over paragraphs."
    
    if insertion.strip() not in content:
        lines = content.splitlines()
        # Insert after first ## section
        for i, line in enumerate(lines):
            if line.startswith("## ") and i > 0:
                lines.insert(i + 1, insertion)
                break
        return "\n".join(lines)
    return content


def add_edge_case_handling(content: str, skill_name: str) -> str:
    """Add instruction to handle edge cases."""
    insertion = "\n- Explicitly check for and handle edge cases (empty inputs, missing files, errors)."
    
    if insertion.strip() not in content:
        lines = content.splitlines()
        # Insert near the end of steps
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip().startswith("-"):
                lines.insert(i + 1, insertion)
                break
        return "\n".join(lines)
    return content


def add_verification_step(content: str, skill_name: str) -> str:
    """Add a verification/final check step."""
    insertion = "\n- Before finishing, verify your output meets all requirements from the skill."
    
    if insertion.strip() not in content:
        lines = content.splitlines()
        # Insert at end of steps
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip().startswith("-") and i < len(lines) - 2:
                lines.insert(i + 1, insertion)
                break
        return "\n".join(lines)
    return content


def strengthen_trigger_condition(content: str, skill_name: str) -> str:
    """Make the trigger/when-to-use condition more precise."""
    lines = content.splitlines()
    
    for i, line in enumerate(lines):
        if "when" in line.lower() and ("use" in line.lower() or "trigger" in line.lower()):
            # Make it more specific
            if "when you" in line.lower():
                lines[i] = line.replace("when you", "when you clearly need to")
            elif "use this" in line.lower():
                lines[i] = line.replace("use this", "use this skill specifically when")
            break
    
    return "\n".join(lines)


def add_example_clarification(content: str, skill_name: str) -> str:
    """Add or improve an example section."""
    lines = content.splitlines()
    
    # Check if examples exist
    has_examples = any("example" in line.lower() or "## Example" in line for line in lines)
    
    if not has_examples:
        # Add a placeholder example section at the end
        insertion = "\n## Example\n\n> **User:** [example task]\n>\n> **Agent:** [expected behavior following this skill]\n"
        return content + insertion
    
    return content


# =============================================================================
# Skill-Specific Mutation Strategies
# =============================================================================

def github_code_review_mutations(content: str, skill_name: str) -> List[Callable]:
    """Mutation strategies specific to github-code-review skill."""
    return [
        lambda c, n: add_security_focus(c, n),
        lambda c, n: add_testing_focus(c, n),
        lambda c, n: add_performance_focus(c, n),
        lambda c, n: add_documentation_focus(c, n),
        add_clarity_instruction,
        add_conciseness_instruction,
    ]


def add_security_focus(content: str, skill_name: str) -> str:
    """Add explicit security vulnerability checking."""
    insertion = "\n- **Security first:** Actively scan for SQL injection, XSS, path traversal, auth bypass, secrets exposure."
    
    if "security" not in content.lower() or "sql injection" not in content.lower():
        lines = content.splitlines()
        # Insert in the steps section
        for i, line in enumerate(lines):
            if line.strip().startswith("## Steps") or "step" in line.lower():
                lines.insert(i + 1, insertion)
                break
        return "\n".join(lines)
    return content


def add_testing_focus(content: str, skill_name: str) -> str:
    """Add emphasis on checking for tests."""
    insertion = "\n- **Tests required:** Verify new/changed code has tests. Flag missing tests as blocking."
    
    if "test" not in content.lower() or "missing test" not in content.lower():
        lines = content.splitlines()
        for i, line in enumerate(lines):
            if "step" in line.lower() and i < len(lines) - 1:
                lines.insert(i + 1, insertion)
                break
        return "\n".join(lines)
    return content


def add_performance_focus(content: str, skill_name: str) -> str:
    """Add performance consideration reminder."""
    insertion = "\n- **Performance:** Note N+1 queries, missing indexes, inefficient loops, unbounded allocations."
    
    if "performance" not in content.lower():
        lines = content.splitlines()
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip().startswith("-"):
                lines.insert(i + 1, insertion)
                break
        return "\n".join(lines)
    return content


def add_documentation_focus(content: str, skill_name: str) -> str:
    """Add documentation/comment checking."""
    insertion = "\n- **Documentation:** Check for missing docstrings, unclear variable names, outdated comments."
    
    if "document" not in content.lower() and "comment" not in content.lower():
        lines = content.splitlines()
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip().startswith("-"):
                lines.insert(i + 1, insertion)
                break
        return "\n".join(lines)
    return content


def systematic_debugging_mutations(content: str, skill_name: str) -> List[Callable]:
    """Mutation strategies for systematic-debugging skill."""
    return [
        lambda c, n: add_hypothesis_discipline(c, n),
        lambda c, n: add_minimal_repro(c, n),
        lambda c, n: add_verification_rigor(c, n),
        add_clarity_instruction,
        add_conciseness_instruction,
    ]


def add_hypothesis_discipline(content: str, skill_name: str) -> str:
    """Enforce hypothesis-before-action discipline."""
    insertion = "\n- **Hypothesis first:** Write your hypothesis BEFORE any debugging action. No shotgun debugging."
    
    if "hypothesis" not in content.lower():
        lines = content.splitlines()
        for i, line in enumerate(lines):
            if "step" in line.lower() or line.strip().startswith("##"):
                lines.insert(i + 1, insertion)
                break
        return "\n".join(lines)
    return content


def add_minimal_repro(content: str, skill_name: str) -> str:
    """Emphasize minimal reproduction."""
    insertion = "\n- **Minimal repro:** Isolate the bug to the smallest possible test case before investigating."
    
    if "minimal" not in content.lower() and "repro" not in content.lower():
        lines = content.splitlines()
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip().startswith("-"):
                lines.insert(i + 1, insertion)
                break
        return "\n".join(lines)
    return content


def add_verification_rigor(content: str, skill_name: str) -> str:
    """Add rigorous fix verification."""
    insertion = "\n- **Verify fix:** Run the exact failing test, then run full suite. Confirm no regressions."
    
    if "verify" not in content.lower() or "regress" not in content.lower():
        lines = content.splitlines()
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip().startswith("-"):
                lines.insert(i + 1, insertion)
                break
        return "\n".join(lines)
    return content


def arxiv_mutations(content: str, skill_name: str) -> List[Callable]:
    """Mutation strategies for arxiv skill."""
    return [
        lambda c, n: add_synthesis_focus(c, n),
        lambda c, n: add_citation_rigor(c, n),
        lambda c, n: add_limitation_awareness(c, n),
        add_clarity_instruction,
        add_conciseness_instruction,
    ]


def add_synthesis_focus(content: str, skill_name: str) -> str:
    """Emphasize cross-paper synthesis."""
    insertion = "\n- **Synthesize:** Don't just list papers. Compare methods, find contradictions, identify consensus."
    
    if "synthes" not in content.lower():
        lines = content.splitlines()
        for i, line in enumerate(lines):
            if "step" in line.lower() or "##" in line:
                lines.insert(i + 1, insertion)
                break
        return "\n".join(lines)
    return content


def add_citation_rigor(content: str, skill_name: str) -> str:
    """Enforce proper citation format."""
    insertion = "\n- **Citations:** Every claim must have [arXiv:XXXX.XXXXX] or proper reference. No uncited claims."
    
    if "citation" not in content.lower() and "arxiv:" not in content.lower():
        lines = content.splitlines()
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip().startswith("-"):
                lines.insert(i + 1, insertion)
                break
        return "\n".join(lines)
    return content


def add_limitation_awareness(content: str, skill_name: str) -> str:
    """Add emphasis on identifying limitations."""
    insertion = "\n- **Limitations:** Explicitly note each paper's limitations, assumptions, and what it doesn't prove."
    
    if "limitation" not in content.lower():
        lines = content.splitlines()
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip().startswith("-"):
                lines.insert(i + 1, insertion)
                break
        return "\n".join(lines)
    return content


# =============================================================================
# Mutation Registry
# =============================================================================

# Map skill names to their mutation strategies
SKILL_MUTATIONS = {
    "github-code-review": github_code_review_mutations,
    "systematic-debugging": systematic_debugging_mutations,
    "arxiv": arxiv_mutations,
}

# Generic mutations available to all skills
GENERIC_MUTATIONS = [
    add_clarity_instruction,
    add_conciseness_instruction,
    add_edge_case_handling,
    add_verification_step,
    strengthen_trigger_condition,
    add_example_clarification,
]


def get_mutation_strategies(skill_name: str) -> List[Callable]:
    """Get all mutation strategies for a skill (skill-specific + generic)."""
    strategies = []
    
    # Add skill-specific mutations
    if skill_name in SKILL_MUTATIONS:
        specific = SKILL_MUTATIONS[skill_name]("", skill_name)
        strategies.extend(specific)
    
    # Add generic mutations
    strategies.extend(GENERIC_MUTATIONS)
    
    return strategies


def apply_random_mutation(content: str, skill_name: str) -> str:
    """Apply a random mutation to the skill content."""
    strategies = get_mutation_strategies(skill_name)
    mutator = random.choice(strategies)
    return mutator(content, skill_name)


def apply_all_mutations(content: str, skill_name: str) -> List[tuple]:
    """Apply all mutations and return (name, mutated_content) pairs."""
    strategies = get_mutation_strategies(skill_name)
    results = []
    
    for mutator in strategies:
        try:
            mutated = mutator(content, skill_name)
            if mutated != content:
                results.append((mutator.__name__, mutated))
        except Exception as e:
            # Skip broken mutators
            pass
    
    return results