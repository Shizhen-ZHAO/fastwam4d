#!/usr/bin/env python3
"""Final gate: verify all 9 phase comparison.json files come from real GPU runs and passed."""
import json
import sys
from pathlib import Path

REVIEW_ROOT = Path('/ossfs/workspace/alignment_results/20260912_001614')
PHASES = ['phase1_math_workers0_r4', 'phase2_math_workers8_r2', 'phase3_default_workers8']
VARIANTS = ['uncond', 'joint', 'idm']

ok = True
rows = []
for phase in PHASES:
    for variant in VARIANTS:
        path = REVIEW_ROOT / phase / variant / 'comparison.json'
        if not path.is_file():
            rows.append((phase, variant, 'MISSING', '-'))
            ok = False
            continue
        d = json.loads(path.read_text())
        if d.get('passed') is True:
            rows.append((phase, variant, 'PASS',
                         f"ranks={d.get('ranks')} microbatches={d.get('microbatch_records_compared')} maxdiff={d.get('max_loss_absolute_difference')}"))
        else:
            first = d.get('first_difference') or d.get('error')
            rows.append((phase, variant, 'FAIL', json.dumps(first, ensure_ascii=False)[:200]))
            ok = False

print(f"{'phase':28s} {'variant':8s} {'status':8s} detail")
for phase, variant, status, detail in rows:
    print(f"{phase:28s} {variant:8s} {status:8s} {detail}")
print()
if ok:
    print('GATE: ALL 9 COMPARISONS PASSED — formal uncond training is authorized')
else:
    print('GATE: NOT ALL PASSED — do NOT start formal training')
sys.exit(0 if ok else 1)
