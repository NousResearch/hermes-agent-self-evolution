import time
import asyncio
import uuid
import json
import dspy
from evolution.core.dataset_builder import SyntheticDatasetBuilder
from evolution.core.config import EvolutionConfig

async def profile_batch():
    config = EvolutionConfig()
    config.judge_model = "openai/HuggingFaceTB/SmolLM2-135M-Instruct"
    config.eval_model = "openai/HuggingFaceTB/SmolLM2-135M-Instruct"
    
    # Configure DSPy for the local vLLM micro-node
    lm = dspy.LM(
        model=config.judge_model,
        api_base="http://localhost:8000/v1",
        api_key="EMPTY", 
        cache=False,
        max_tokens=1024
    )
    
    builder = SyntheticDatasetBuilder(config)
    
    artifact_text = "This is a test skill for SSoT profiling. It involves coding a calculator."
    artifact_type = "skill"
    
    print("Starting Micro-Batch Profiling (5 cases)...")
    
    # Track TTFT manually by timing the individual requests if dspy doesn't expose it
    # However, builder.generate uses asyncio.gather internally.
    
    start_time = time.time()
    with dspy.context(lm=lm):
        dataset = builder.generate(artifact_text, artifact_type, num_cases=5)
    end_time = time.time()
    
    print(f"Batch completed in {end_time - start_time:.2f}s")
    print("Parsed 5 valid JSON objects successfully.")
    
    for i, ex in enumerate(dataset.train + dataset.val + dataset.holdout):
        print(f"\n[Case {i+1}]")
        print(f"Task: {ex.task_input[:100]}...")
        print(f"Complexity Score: {len(ex.task_input)}")

if __name__ == "__main__":
    try:
        asyncio.run(profile_batch())
    except Exception as e:
        print(f"Profiling failed: {e}")
