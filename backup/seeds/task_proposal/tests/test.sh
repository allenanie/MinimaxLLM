#!/bin/bash
# Verifier. Must write a reward in [0,1] to /logs/verifier/reward.txt.
#
# Checks the answer, not the existence of a file: a verifier that is satisfied by
# an empty summary.json would make this a worthless task, and the Robust Harness
# Game exists precisely to penalize agents that exploit that kind of gap.
set -uo pipefail

mkdir -p /logs/verifier

python3 - <<'PY' > /logs/verifier/reward.txt
import json

EXPECTED_TOTAL = 61.75
EXPECTED_SKIPPED = 3

try:
    with open("/app/summary.json") as handle:
        payload = json.load(handle)
except Exception:
    print("0.0")
    raise SystemExit

score = 0.0
if isinstance(payload, dict):
    total = payload.get("total")
    skipped = payload.get("skipped")
    if isinstance(total, (int, float)) and abs(float(total) - EXPECTED_TOTAL) < 0.01:
        score += 0.5
    if isinstance(skipped, int) and skipped == EXPECTED_SKIPPED:
        score += 0.5

print(f"{score:.1f}")
PY
