#!/bin/bash
# Reference solution. Executed by Harbor's oracle agent as the solvability gate:
# if this does not score 1.0, the task is rejected before any harness sees it.
set -euo pipefail

python3 - <<'PY'
import json

total = 0.0
skipped = 0
with open("/app/records.csv") as handle:
    for line in handle:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) != 3 or not parts[2].strip():
            skipped += 1
            continue
        try:
            total += float(parts[1])
        except ValueError:
            skipped += 1

with open("/app/summary.json", "w") as handle:
    json.dump({"total": round(total, 2), "skipped": skipped}, handle)
PY
