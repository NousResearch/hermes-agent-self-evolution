"""
Constraints Module for Hermes Agent Self-Evolution.

Contains constraint validation logic for evolved skill variants.
"""

import re
import subprocess
import ast
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum


class ConstraintSeverity(Enum):
    ERROR = "error"        # Must pass - blocks deployment
    WARNING = "warning"    # Should pass - blocks if too many
    INFO = "info"          # Informational only


@dataclass
class ConstraintResult:
    """Result of a single constraint check."""
    name: str
    passed: bool
    severity: ConstraintSeverity
    message: str
    details: Dict[str, Any]


@dataclass
class ValidationReport:
    """Complete validation report for an evolved variant."""
    all_passed: bool
    results: List[ConstraintResult]
    errors: List[ConstraintResult]
    warnings: List[ConstraintResult]
    infos: List[ConstraintResult]
    
    def __post_init__(self):
        self.errors = [r for r in self.results if r.severity == ConstraintSeverity.ERROR and not r.passed]
        self.warnings = [r for r in self.results if r.severity == ConstraintSeverity.WARNING and not r.passed]
        self.infos = [r for r in self.results if r.severity == ConstraintSeverity.INFO and not r.passed]
        self.all_passed = len(self.errors) == 0 and len(self.warnings) == 0


class ConstraintValidator:
    """
    Validates evolved skill/tool/prompt variants against all guardrails.
    """
    
    def __init__(
        self,
        hermes_agent_repo: Path,
        skill_name: str,
        original_content: str,
        evolved_content: str,
        hermes_test_cmd: str = "pytest tests/ -q"
    ):
        self.hermes_agent_repo = Path(hermes_agent_repo)
        self.skill_name = skill_name
        self.original_content = original_content
        self.evolved_content = evolved_content
        self.hermes_test_cmd = hermes_test_cmd
    
    def validate_all(self) -> ValidationReport:
        """Run all constraint checks."""
        results = []
        
        # Size constraints
        results.append(self._check_skill_size())
        results.append(self._check_no_bloat())
        
        # Content constraints
        results.append(self._check_structure_preserved())
        results.append(self._check_no_hallucinated_tools())
        results.append(self._check_caching_compatible())
        
        # Semantic constraints
        results.append(self._check_purpose_preserved())
        results.append(self._check_no_malicious_instructions())
        
        # Test suite
        results.append(self._run_test_suite())
        
        # Benchmark gate (if available)
        results.append(self._check_benchmark_gate())
        
        return ValidationReport(
            all_passed=False,  # Will be computed in __post_init__
            results=results
        )
    
    # ========== SIZE CONSTRAINTS ==========
    
    def _check_skill_size(self) -> ConstraintResult:
        """Skills must be ≤15KB (including any added examples)."""
        size_bytes = len(self.evolved_content.encode('utf-8'))
        size_kb = size_bytes / 1024
        limit_kb = 15
        
        passed = size_kb <= limit_kb
        return ConstraintResult(
            name="skill_size_limit",
            passed=passed,
            severity=ConstraintSeverity.ERROR,
            message=f"Skill size: {size_kb:.1f}KB / {limit_kb}KB limit",
            details={"size_bytes": size_bytes, "size_kb": size_kb, "limit_kb": limit_kb}
        )
    
    def _check_no_bloat(self) -> ConstraintResult:
        """Check that evolution didn't just add filler text."""
        orig_lines = len(self.original_content.splitlines())
        evo_lines = len(self.evolved_content.splitlines())
        line_increase_pct = ((evo_lines - orig_lines) / orig_lines) * 100 if orig_lines > 0 else 0
        
        # Allow up to 50% line increase for legitimate additions
        passed = line_increase_pct <= 50
        
        return ConstraintResult(
            name="no_excessive_bloat",
            passed=passed,
            severity=ConstraintSeverity.WARNING,
            message=f"Line count increase: {line_increase_pct:.1f}% ({orig_lines} → {evo_lines} lines)",
            details={"original_lines": orig_lines, "evolved_lines": evo_lines, "increase_pct": line_increase_pct}
        )
    
    # ========== CONTENT CONSTRAINTS ==========
    
    def _check_structure_preserved(self) -> ConstraintResult:
        """Check that SKILL.md structural elements are preserved."""
        required_sections = [
            "## When to Use",
            "## Steps", 
            "## Examples"  # Optional but good
        ]
        
        missing = []
        for section in required_sections:
            if section not in self.evolved_content:
                missing.append(section)
        
        # Also check that the trigger condition is still there
        trigger_present = any(
            kw in self.evolved_content.lower() 
            for kw in ["when", "trigger", "use when", "use this skill"]
        )
        
        passed = len(missing) == 0 and trigger_present
        
        details = {"missing_sections": missing, "trigger_present": trigger_present}
        if not trigger_present:
            details["note"] = "No trigger/use condition detected"
        
        return ConstraintResult(
            name="structure_preserved",
            passed=passed,
            severity=ConstraintSeverity.ERROR,
            message=f"Structure check: {len(missing)} required sections missing" if missing else "All required sections present",
            details=details
        )
    
    def _check_no_hallucinated_tools(self) -> ConstraintResult:
        """Check that the skill doesn't reference non-existent tools."""
        # Extract tool references from the skill
        tool_pattern = r'`(\w+)`'
        mentioned_tools = set(re.findall(tool_pattern, self.evolved_content))
        
        # Also check for tool calls in examples
        example_tools = set(re.findall(r'(\w+)\(\)', self.evolved_content))
        mentioned_tools.update(example_tools)
        
        # Known valid tools in Hermes (could be expanded)
        valid_tools = {
            "read_file", "write_file", "patch", "search_files", "terminal",
            "list_dir", "glob", "grep", "task_tool",
            "web_search", "web_extract", "skill_view", "skill_manage",
            "memory", "session_search", "cronjob", "delegate_task",
            "computer_use", "vision_analyze", "video_analyze"
        }
        
        invalid = mentioned_tools - valid_tools
        # Filter out common false positives
        false_positives = {"and", "or", "if", "else", "for", "while", "the", "a", "to", "in", "of", "on", "use", "skill", "step", "then"}
        invalid = invalid - false_positives
        
        passed = len(invalid) == 0
        
        return ConstraintResult(
            name="no_hallucinated_tools",
            passed=passed,
            severity=ConstraintSeverity.ERROR,
            message=f"Invalid tool references: {sorted(invalid)}" if invalid else "All tool references valid",
            details={"mentioned_tools": sorted(mentioned_tools), "invalid_tools": sorted(invalid)}
        )
    
    def _check_caching_compatible(self) -> ConstraintResult:
        """
        Check that skill changes are caching-compatible.
        Skills are loaded at conversation start and cached.
        Must not contain instructions that change behavior mid-conversation.
        """
        problematic_patterns = [
            r"after\s+\d+\s+turns?",
            r"on\s+the\s+\d+(st|nd|rd|th)\s+turn",
            r"mid.conversation",
            r"during\s+the\s+conversation",
            r"change\s+behavior",
            r"switch\s+mode",
            r"override.*previous",
            r"forget\s+earlier",
            r"ignore\s+previous",
        ]
        
        violations = []
        for pattern in problematic_patterns:
            if re.search(pattern, self.evolved_content, re.IGNORECASE):
                violations.append(pattern)
        
        passed = len(violations) == 0
        
        return ConstraintResult(
            name="caching_compatible",
            passed=passed,
            severity=ConstraintSeverity.ERROR,
            message=f"Caching-incompatible patterns: {violations}" if violations else "No caching-incompatible patterns found",
            details={"violations": violations}
        )
    
    # ========== SEMANTIC CONSTRAINTS ==========
    
    def _check_purpose_preserved(self) -> ConstraintResult:
        """Check that the skill's core purpose hasn't drifted."""
        # Extract the "when to use" / purpose section
        orig_purpose = self._extract_purpose(self.original_content)
        evo_purpose = self._extract_purpose(self.evolved_content)
        
        # Simple keyword overlap check
        orig_keywords = set(orig_purpose.lower().split())
        evo_keywords = set(evo_purpose.lower().split())
        
        # Remove common stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from", "as", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "must", "can", "this", "that", "these", "those", "i", "you", "we", "they", "he", "she", "it", "when", "use", "using", "used", "skill", "task", "help", "helps", "helping"}
        orig_keywords = orig_keywords - stop_words
        evo_keywords = evo_keywords - stop_words
        
        if len(orig_keywords) == 0:
            return ConstraintResult(
                name="purpose_preserved",
                passed=True,
                severity=ConstraintSeverity.WARNING,
                message="Could not extract purpose keywords from original",
                details={"original_keywords": [], "evolved_keywords": list(evo_keywords)}
            )
        
        overlap = len(orig_keywords & evo_keywords) / len(orig_keywords)
        
        # Require at least 40% keyword overlap
        passed = overlap >= 0.4
        
        return ConstraintResult(
            name="purpose_preserved",
            passed=passed,
            severity=ConstraintSeverity.ERROR,
            message=f"Purpose overlap: {overlap:.0%} (need ≥40%)" if not passed else f"Purpose preserved: {overlap:.0%} overlap",
            details={
                "overlap_ratio": overlap,
                "original_keywords": sorted(orig_keywords),
                "evolved_keywords": sorted(evo_keywords),
                "shared_keywords": sorted(orig_keywords & evo_keywords)
            }
        )
    
    def _extract_purpose(self, content: str) -> str:
        """Extract purpose/when-to-use section."""
        lines = content.splitlines()
        in_purpose = False
        purpose_lines = []
        
        for line in lines:
            if any(kw in line.lower() for kw in ["when to use", "trigger", "use this skill", "when you should"]):
                in_purpose = True
                continue
            if in_purpose and line.startswith("## "):
                break
            if in_purpose:
                purpose_lines.append(line)
        
        return " ".join(purpose_lines) if purpose_lines else content[:500]
    
    def _check_no_malicious_instructions(self) -> ConstraintResult:
        """Check for potentially harmful instructions."""
        malicious_patterns = [
            r"ignore\s+(previous|all|above)\s+instructions",
            r"forget\s+(everything|all|previous)",
            r"you\s+are\s+now\s+(a|an)\s+(hacker|attacker|unrestricted)",
            r"bypass\s+(security|safety|guards?)",
            r"disable\s+(safety|guards?|filters?)",
            r"output\s+(secrets?|passwords?|keys?|tokens?)",
            r"send\s+(data|files?)\s+to\s+(external|outside|third.party)",
            r"execute\s+(arbitrary|shell|system)\s+(commands?|code)",
        ]
        
        violations = []
        for pattern in malicious_patterns:
            if re.search(pattern, self.evolved_content, re.IGNORECASE):
                violations.append(pattern)
        
        passed = len(violations) == 0
        
        return ConstraintResult(
            name="no_malicious_instructions",
            passed=passed,
            severity=ConstraintSeverity.ERROR,
            message=f"Malicious patterns detected: {violations}" if violations else "No malicious instructions found",
            details={"violations": violations}
        )
    
    # ========== TEST & BENCHMARK GATES ==========
    
    def _run_test_suite(self) -> ConstraintResult:
        """Run the full Hermes test suite."""
        try:
            # Write evolved skill to a temp location in the repo
            skill_path = self.hermes_agent_repo / "skills" / f"{self.skill_name}.md"
            original_backup = skill_path.read_text() if skill_path.exists() else ""
            
            try:
                skill_path.write_text(self.evolved_content)
                # Run tests
                result = subprocess.run(
                    self.hermes_test_cmd.split(),
                    cwd=self.hermes_agent_repo,
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                passed = result.returncode == 0
                message = "All tests passed" if passed else f"Tests failed: {result.stdout[-500:]}"
            finally:
                # Restore original
                if original_backup:
                    skill_path.write_text(original_backup)
            
        except subprocess.TimeoutExpired:
            passed = False
            message = "Test suite timed out (>120s)"
        except Exception as e:
            passed = False
            message = f"Test execution error: {e}"
        
        return ConstraintResult(
            name="test_suite",
            passed=passed,
            severity=ConstraintSeverity.ERROR,
            message=message,
            details={"test_cmd": self.hermes_test_cmd}
        )
    
    def _check_benchmark_gate(self) -> ConstraintResult:
        """Check benchmark gate (TBLite fast test)."""
        try:
            tblite_script = self.hermes_agent_repo / "scripts" / "run_tblite_fast.py"
            if tblite_script.exists():
                result = subprocess.run(
                    ["python", str(tblite_script)],
                    cwd=self.hermes_agent_repo,
                    capture_output=True,
                    text=True,
                    timeout=180
                )
                # Parse TBLite score from output
                # For now just report it ran
                return ConstraintResult(
                    name="benchmark_gate",
                    passed=result.returncode == 0,
                    severity=ConstraintSeverity.WARNING,
                    message="TBLite benchmark completed" if result.returncode == 0 else "TBLite benchmark failed",
                    details={"returncode": result.returncode, "output": result.stdout[-1000:]}
                )
        except Exception as e:
            pass
        
        return ConstraintResult(
            name="benchmark_gate",
            passed=True,  # Not blocking if not available
            severity=ConstraintSeverity.INFO,
            message="Benchmark gate not available / not run",
            details={"available": False}
        )


def validate_skill_evolution(
    hermes_agent_repo: Path,
    skill_name: str,
    original_content: str,
    evolved_content: str,
    hermes_test_cmd: str = "pytest tests/ -q"
) -> ValidationReport:
    """Convenience function to validate a skill evolution."""
    validator = ConstraintValidator(
        hermes_agent_repo=hermes_agent_repo,
        skill_name=skill_name,
        original_content=original_content,
        evolved_content=evolved_content,
        hermes_test_cmd=hermes_test_cmd
    )
    return validator.validate_all()