"""Quick test: does the LLM judge produce different scores across trials?"""

import sys
sys.path.insert(0, ".")

import dspy
from evolution.core.config import EvolutionConfig
from evolution.core.fitness import LLMJudge, FitnessScore

config = EvolutionConfig(judge_model="openai/xiaomi/mimo-v2-pro")

judge = LLMJudge(config)

task = "Generate a BibTeX entry for paper 1706.03762."
expected = "Use arXiv API to fetch metadata, parse XML, generate BibTeX entry"
output = "curl -s 'http://export.arxiv.org/api/query?id_list=1706.03762' | python3 -c \"import sys,xml.etree.ElementTree as ET; root=ET.parse(sys.stdin).getroot(); ns={'a':'http://www.w3.org/2005/Atom'}; entry=root.find('a:entry',ns); title=entry.find('a:title',ns).text.strip().replace('\\n',' '); authors=[a.find('a:name',ns).text for a in entry.findall('a:author',ns)]; year=entry.find('a:published',ns).text[:4]; print(f'@article{{vaswani{year}attention, title={{{title}}}, author={{{' and '.join(authors)}}}, year={{{year}}}}}')"
skill = "Search arXiv papers via their REST API..."

print("Testing judge variance across 3 trials (same inputs):")
for i in range(3):
    score = judge.score(task, expected, output, skill)
    print(f"  Trial {i+1}: composite={score.composite:.4f} correctness={score.correctness:.2f} procedure={score.procedure_following:.2f} conciseness={score.conciseness:.2f}")

print("\nTesting judge variance with DIFFERENT inputs:")
for i in range(3):
    modified_output = output + f"\n# Trial {i}: added some extra text to vary the input"
    score = judge.score(task, expected, modified_output, skill)
    print(f"  Trial {i+1}: composite={score.composite:.4f} correctness={score.correctness:.2f} procedure={score.procedure_following:.2f} conciseness={score.conciseness:.2f}")
