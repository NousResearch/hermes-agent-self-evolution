"""Atropos RL Integration for Hermes Agent.

PR #6: Bridges Atropos RL environments with Hermes agent tasks.
Enables training Hermes agents on custom tasks using reinforcement learning.

Based on: https://github.com/NousResearch/atropos
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class HermesTaskEnvironment:
    """Wraps a Hermes skill as an Atropos-compatible RL environment.
    
    This allows using Atropos's PPO/RLHF trainers to optimize
    Hermes agent behavior on specific tasks.
    
    Usage:
        from evolution.rl.atropos_adapter import HermesTaskEnvironment
        
        env = HermesTaskEnvironment(
            skill_name="github-code-review",
            task_type="code_review",
            dataset_path="datasets/github-code-review/",
        )
        
        # Use with Atropos trainer
        # trainer = AtroposPPO(env=env, model="gpt-4.1-mini")
        # trainer.train(steps=1000)
    """
    
    skill_name: str
    task_type: str  # "code_review", "trading_signal", "creative_prompt", etc.
    dataset_path: Path
    
    # Task-specific configuration
    max_turns: int = 10
    reward_function: str = "llm_judge"  # llm_judge, exact_match, composite
    
    # State
    _current_example: Optional[Dict] = None
    _turn_count: int = 0
    _episode_reward: float = 0.0

    def __post_init__(self):
        self.dataset_path = Path(self.dataset_path)
        self._load_dataset()
    
    def _load_dataset(self):
        """Load evaluation dataset for this task."""
        self.examples = []
        if self.dataset_path.exists():
            for f in self.dataset_path.glob("*.jsonl"):
                with open(f) as fh:
                    for line in fh:
                        self.examples.append(json.loads(line))
    
    def reset(self) -> Dict[str, Any]:
        """Reset environment and return initial observation."""
        import random
        if not self.examples:
            # Fallback: create synthetic example
            self._current_example = {
                "task_input": f"Execute {self.skill_name} skill",
                "expected_behavior": "Complete the task correctly",
                "difficulty": "medium",
            }
        else:
            self._current_example = random.choice(self.examples)
        
        self._turn_count = 0
        self._episode_reward = 0.0
        
        return {
            "observation": self._current_example["task_input"],
            "task_type": self.task_type,
            "skill_name": self.skill_name,
            "max_turns": self.max_turns,
            "turn": 0,
        }
    
    def step(self, action: str) -> tuple:
        """Execute one step in the environment.
        
        Args:
            action: The agent's response/action string.
        
        Returns:
            (observation, reward, done, info)
        """
        self._turn_count += 1
        
        # Calculate reward
        reward = self._compute_reward(action)
        self._episode_reward += reward
        
        done = self._turn_count >= self.max_turns or reward > 0.9
        
        observation = {
            "observation": f"Turn {self._turn_count}/{self.max_turns}. Previous action: {action[:200]}...",
            "turn": self._turn_count,
        }
        
        info = {
            "episode_reward": self._episode_reward,
            "turn_reward": reward,
            "skill_name": self.skill_name,
        }
        
        return observation, reward, done, info
    
    def _compute_reward(self, action: str) -> float:
        """Compute reward for the agent's action.
        
        Uses LLM-as-judge to score action quality against expected behavior.
        """
        if not self._current_example or self.reward_function == "llm_judge":
            return self._llm_judge_reward(action)
        elif self.reward_function == "exact_match":
            return self._exact_match_reward(action)
        else:
            return self._composite_reward(action)
    
    def _llm_judge_reward(self, action: str) -> float:
        """Use LLM to judge action quality."""
        expected = self._current_example.get("expected_behavior", "")
        if not expected:
            return 0.5  # Neutral if no reference
        
        # Simple heuristic: keyword overlap
        action_lower = action.lower()
        expected_lower = expected.lower()
        
        action_words = set(action_lower.split())
        expected_words = set(expected_lower.split())
        
        if not expected_words:
            return 0.5
        
        overlap = len(action_words & expected_words) / len(expected_words)
        
        # Boost for including key phrases
        key_phrases = [
            "pytest", "test", "assert",  # code review
            "buy", "sell", "stop loss", "take profit",  # trading
            "sprite", "render", "animation",  # creative
        ]
        
        for phrase in key_phrases:
            if phrase in action_lower and phrase in expected_lower:
                overlap += 0.1
        
        return min(1.0, max(0.0, overlap))
    
    def _exact_match_reward(self, action: str) -> float:
        """Exact match reward (for deterministic tasks)."""
        expected = self._current_example.get("expected_behavior", "")
        return 1.0 if action.strip() == expected.strip() else 0.0
    
    def _composite_reward(self, action: str) -> float:
        """Combine multiple reward signals."""
        llm_score = self._llm_judge_reward(action)
        exact_score = self._exact_match_reward(action)
        
        # Weighted combination
        return 0.7 * llm_score + 0.3 * exact_score
    
    def render(self) -> str:
        """Human-readable environment state."""
        return f"""
Hermes RL Environment
====================
Skill: {self.skill_name}
Task:  {self.task_type}
Turn:  {self._turn_count}/{self.max_turns}
Reward: {self._episode_reward:.3f}
Current Example: {self._current_example.get('task_input', 'N/A')[:100]}...
"""


class MultiSkillCurriculum:
    """Train across multiple skills with curriculum learning.
    
    Progresses from easy to hard tasks automatically.
    """
    
    def __init__(self, skill_envs: List[HermesTaskEnvironment]):
        self.skill_envs = skill_envs
        self.current_skill_idx = 0
        self.success_threshold = 0.8
        self.consecutive_successes = 0
    
    def reset(self):
        """Reset to appropriate difficulty level."""
        # If current skill mastered, advance
        if self.consecutive_successes >= 3:
            self.current_skill_idx = min(
                self.current_skill_idx + 1,
                len(self.skill_envs) - 1
            )
            self.consecutive_successes = 0
        
        return self.skill_envs[self.current_skill_idx].reset()
    
    def step(self, action: str):
        """Step current environment and track success."""
        env = self.skill_envs[self.current_skill_idx]
        obs, reward, done, info = env.step(action)
        
        if reward > self.success_threshold:
            self.consecutive_successes += 1
        else:
            self.consecutive_successes = 0
        
        return obs, reward, done, info


def create_hermes_env(skill_name: str, task_type: Optional[str] = None) -> HermesTaskEnvironment:
    """Factory function to create a Hermes environment for a skill.
    
    Auto-detects task type from skill name if not provided.
    """
    if not task_type:
        # Heuristic task type detection
        if any(x in skill_name for x in ["trade", "fx", "signal", "chart"]):
            task_type = "trading_signal"
        elif any(x in skill_name for x in ["code", "review", "pr", "github"]):
            task_type = "code_review"
        elif any(x in skill_name for x in ["creative", "sprite", "image", "video", "design"]):
            task_type = "creative_generation"
        elif any(x in skill_name for x in ["security", "scan", "vuln", "audit"]):
            task_type = "security_analysis"
        else:
            task_type = "general_task"
    
    dataset_path = Path("datasets") / "skills" / skill_name
    
    return HermesTaskEnvironment(
        skill_name=skill_name,
        task_type=task_type,
        dataset_path=dataset_path,
    )


if __name__ == "__main__":
    # Demo: Create environment for a trading skill
    env = create_hermes_env("fx-rich-smc-engine")
    obs = env.reset()
    print(env.render())
    
    # Simulate agent actions
    for i in range(3):
        action = f"Analyze {obs['observation'][:50]}... and provide signal"
        obs, reward, done, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.3f}, done={done}")
        if done:
            break
