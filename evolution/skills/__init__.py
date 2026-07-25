"""
Hermes Agent Self-Evolution - Skills Package

Phase 1: Skill Evolution via DSPy + GEPA
"""

from .skill_module import (
    SkillModuleConfig,
    SkillAsDSPyModule,
    HermesSkillModule,
    create_skill_module,
    SkillOptimizationSignature,
    OptimizableSkillModule,
)
from .mutation_strategies import (
    get_mutation_strategies,
    apply_random_mutation,
    apply_all_mutations,
    SKILL_MUTATIONS,
    GENERIC_MUTATIONS,
)

__all__ = [
    "SkillModuleConfig",
    "SkillAsDSPyModule", 
    "HermesSkillModule",
    "create_skill_module",
    "SkillOptimizationSignature",
    "OptimizableSkillModule",
    "get_mutation_strategies",
    "apply_random_mutation",
    "apply_all_mutations",
    "SKILL_MUTATIONS",
    "GENERIC_MUTATIONS",
]