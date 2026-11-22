"""Run a reproducible offline adaptive-strategy demonstration.

Usage: python -m evolution.adaptive.cli --episodes 300 --seed 7
No API keys, Hermes installation, network, or GPU required.
"""
import argparse
import random
from .benchmark import optimisation_phase
from .core import ContextualBandit, ExperienceStore, Reward, choose_strategy

STRATEGIES = ("direct", "structured", "tool_assisted")
CONTEXTS = ("simple", "security", "complex")
# Synthetic success rates FOR TESTING ONLY, not Hermes measurements.
SUCCESS = {
    "simple": (0.95, 0.84, 0.79),
    "security": (0.48, 0.76, 0.94),
    "complex": (0.42, 0.91, 0.82),
}
COST = {"direct": 0.1, "structured": 0.4, "tool_assisted": 0.9}
TIME = {"direct": 0.1, "structured": 0.4, "tool_assisted": 0.9}
SKILL_VERSION = "synthetic-sim/1"  # no real Hermes skills are used here


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--db", default=":memory:")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    rng = random.Random(args.seed)
    policy = ContextualBandit(STRATEGIES, epsilon=0.15, seed=args.seed)

    def executor(context, strategy):
        rate = SUCCESS[context][STRATEGIES.index(strategy)]
        return {"quality": float(rng.random() < rate),
                "cost": COST[strategy], "time": TIME[strategy]}

    def verifier(result):
        return Reward(result["quality"], result["cost"], result["time"])

    rewards = []
    # The whole run is optimisation: the sealed benchmark split stays unreachable.
    with optimisation_phase(), ExperienceStore(args.db) as store:
        for _ in range(args.episodes):
            context = rng.choice(CONTEXTS)
            record = choose_strategy(policy, context, executor, verifier, store,
                                     skill_version=SKILL_VERSION)
            rewards.append(record["reward"])
    print(f"Synthetic episodes: {args.episodes}; average utility: {sum(rewards)/len(rewards):.3f}")
    for context in CONTEXTS:
        print(f"{context}: " + ", ".join(
            f"{a} n={policy.counts.get((context,a),0)} "
            f"mean={policy.means.get((context,a),0):.3f}" for a in STRATEGIES))


if __name__ == "__main__":
    main()
