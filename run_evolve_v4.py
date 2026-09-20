import os, sys
os.environ['HERMES_PROXY_URL'] = 'https://inference-api.nousresearch.com/v1'
os.environ['HERMES_PROXY_KEY'] = 'eyJhbG...F8LA'
sys.path.insert(0, '.')
from evolution.skills.evolve_skill import evolve
evolve(
    skill_name='requesting-code-review',
    iterations=2,
    eval_source='synthetic',
    hermes_repo=r'C:/Users/emili/AppData/Local/hermes/hermes-agent',
    optimizer_model='meituan/longcat-2.0:free',
    eval_model='meituan/longcat-2.0:free',
)
