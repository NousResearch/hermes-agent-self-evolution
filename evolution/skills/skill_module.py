"""
Skill Module Wrapper for Hermes Agent Self-Evolution.

Wraps a SKILL.md file as a DSPy module that can be optimized by GEPA.
The skill text becomes the system prompt, and the module runs the agent
on a test task, returning the result for scoring.
"""

import dspy
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass


@dataclass
class SkillModuleConfig:
    """Configuration for a skill module."""
    skill_name: str
    skill_content: str
    hermes_agent_repo: Path
    max_tokens: int = 4000
    temperature: float = 0.0


class SkillAsDSPyModule(dspy.Module):
    """
    DSPy module that wraps a Hermes skill.
    
    The skill content is injected as the system prompt, and the module
    executes the agent on a given task, returning the agent's response
    for evaluation by the fitness function.
    """
    
    def __init__(self, config: SkillModuleConfig):
        super().__init__()
        self.config = config
        self.skill_name = config.skill_name
        self.skill_content = config.skill_content
        
        # Define the signature for the skill
        self.signature = dspy.Signature(
            "task_input -> agent_response",
            instructions=self.skill_content
        )
        
        # The predictor that will be optimized
        self.predictor = dspy.Predict(self.signature)
        
        # Track optimization state
        self.optimization_history = []
    
    def forward(self, task_input: str) -> dspy.Prediction:
        """
        Run the agent with the skill on a task.
        
        In practice, this calls the Hermes agent with the skill loaded.
        For GEPA optimization, we need this to be differentiable or at least
        callable - GEPA works by mutating the instructions (skill content)
        and evaluating the results.
        """
        # This is a placeholder - in the real implementation, this would
        # call the actual Hermes agent with the skill loaded.
        # For now, we simulate by using the predictor directly.
        
        # The actual implementation should use batch_runner to run the agent
        # with the skill loaded, but for DSPy/GEPA compatibility, we wrap
        # it as a predictor that GEPA can mutate.
        
        return self.predictor(task_input=task_input)
    
    def get_skill_text(self) -> str:
        """Get the current skill text (for GEPA to mutate)."""
        return self.skill_content
    
    def set_skill_text(self, new_content: str) -> None:
        """Set new skill text (called by GEPA after mutation)."""
        self.skill_content = new_content
        # Update the signature with new instructions
        self.signature = dspy.Signature(
            "task_input -> agent_response", 
            instructions=new_content
        )
        self.predictor = dspy.Predict(self.signature)
        self.optimization_history.append({
            "timestamp": "now",
            "content_length": len(new_content)
        })


class HermesSkillModule(dspy.Module):
    """
    DSPy module that actually calls the Hermes agent.
    
    This is the production version that integrates with the real
    Hermes agent infrastructure via batch_runner.
    """
    
    def __init__(
        self,
        skill_name: str,
        hermes_agent_repo: Path,
        skill_content: Optional[str] = None
    ):
        super().__init__()
        self.skill_name = skill_name
        self.hermes_agent_repo = Path(hermes_agent_repo)
        
        # Load skill content if not provided
        if skill_content is None:
            skill_path = self.hermes_agent_repo / "skills" / f"{skill_name}.md"
            if skill_path.exists():
                self.skill_content = skill_path.read_text()
            else:
                self.skill_content = ""
        else:
            self.skill_content = skill_content
        
        # We don't use a standard predictor - instead we call the agent
        # This is a custom module that GEPA can optimize by mutating
        # the skill_content field
        
    def forward(self, task_input: str) -> str:
        """
        Run the Hermes agent on a task with this skill loaded.
        
        Returns the agent's final response text.
        """
        # This is where we'd integrate with batch_runner
        # For now, return a placeholder that allows the module to work
        # with DSPy's optimization framework
        
        import subprocess
        import json
        import tempfile
        import os
        
        # Write the current skill content to a temp file in the skills dir
        skill_path = self.hermes_agent_repo / "skills" / f"{self.skill_name}.md"
        original_content = skill_path.read_text() if skill_path.exists() else ""
        
        try:
            # Write evolved skill content
            skill_path.write_text(self.skill_content)
            
            # Run hermes agent on the task via CLI
            # This is a simplified version - real implementation would use batch_runner
            env = os.environ.copy()
            env["HERMES_SKILL_OVERRIDE"] = self.skill_name
            
            # Use hermes CLI to run the agent
            result = subprocess.run(
                ["python", "-m", "hermes", "run", "--task", task_input],
                cwd=self.hermes_agent_repo,
                capture_output=True,
                text=True,
                timeout=120,
                env=env
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return f"ERROR: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return "ERROR: Agent timed out"
        except Exception as e:
            return f"ERROR: {str(e)}"
        finally:
            # Restore original skill
            if original_content:
                skill_path.write_text(original_content)
    
    def get_instructions(self) -> str:
        """Get current instructions (skill content) for GEPA."""
        return self.skill_content
    
    def set_instructions(self, instructions: str) -> None:
        """Set new instructions (called by GEPA)."""
        self.skill_content = instructions


def create_skill_module(
    skill_name: str,
    hermes_agent_repo: Path,
    mode: str = "hermes"  # "dspy" or "hermes"
) -> dspy.Module:
    """
    Factory function to create the appropriate skill module.
    
    Args:
        skill_name: Name of the skill (e.g., "github-code-review")
        hermes_agent_repo: Path to hermes-agent repository
        mode: "dspy" for DSPy predictor wrapper, "hermes" for real agent integration
    
    Returns:
        A DSPy module ready for GEPA optimization
    """
    if mode == "dspy":
        # Load skill content
        skill_path = Path(hermes_agent_repo) / "skills" / f"{skill_name}.md"
        if skill_path.exists():
            skill_content = skill_path.read_text()
        else:
            skill_content = f"# {skill_name}\n\nSkill not found."
        
        config = SkillModuleConfig(
            skill_name=skill_name,
            skill_content=skill_content,
            hermes_agent_repo=Path(hermes_agent_repo)
        )
        return SkillAsDSPyModule(config)
    
    elif mode == "hermes":
        return HermesSkillModule(
            skill_name=skill_name,
            hermes_agent_repo=Path(hermes_agent_repo)
        )
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


# GEPA-compatible module signature
class SkillOptimizationSignature(dspy.Signature):
    """Signature for skill optimization - the skill text IS the instructions."""
    task_input: str = dspy.InputField(desc="The task to perform")
    agent_response: str = dspy.OutputField(desc="The agent's response following the skill")


# For GEPA, we need a module where the skill content can be mutated
# GEPA works by treating the instructions as optimizable parameters
class OptimizableSkillModule(dspy.Module):
    """
    A skill module where the skill content is an optimizable parameter.
    
    GEPA will mutate the skill_content field and evaluate the results.
    This is the core of how GEPA optimizes skills.
    """
    
    def __init__(self, skill_name: str, hermes_agent_repo: Path):
        super().__init__()
        self.skill_name = skill_name
        self.hermes_agent_repo = Path(hermes_agent_repo)
        
        # Load initial skill content
        skill_path = self.hermes_agent_repo / "skills" / f"{skill_name}.md"
        if skill_path.exists():
            self.skill_content = skill_path.read_text()
        else:
            self.skill_content = f"# {skill_name}\n\nSkill not found."
        
        # The predictor that uses our skill content as instructions
        self.predictor = dspy.Predict(SkillOptimizationSignature)
    
    def forward(self, task_input: str) -> dspy.Prediction:
        """Run prediction with current skill content as instructions."""
        # Update predictor instructions with current skill content
        self.predictor.signature = dspy.Signature(
            "task_input -> agent_response",
            instructions=self.skill_content
        )
        return self.predictor(task_input=task_input)
    
    def get_skill_content(self) -> str:
        return self.skill_content
    
    def set_skill_content(self, content: str) -> None:
        self.skill_content = content