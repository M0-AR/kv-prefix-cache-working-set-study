#!/usr/bin/env bash
# 60-second terminal demo: runs the full verified suite with narration.
# Usage: ./scripts/demo.sh        (add ./scripts/record_demo.sh to capture)
set -e
cd "$(dirname "$0")/.."
echo "# KV prefix-cache working-set study — 60-second demo"
echo "# Every number below is computed live on this machine."
echo ""
echo "$ docker compose up --build   # or: python experiments/run_all.py"
sleep 1
python3 experiments/run_all.py 2>&1 | grep -E "===|max_abs_err|working_set|burst_source|sequential_hit|ALL EXPERIMENTS" | head -n 20
echo ""
echo "# Full paper: README.md  |  Interactive site: preview.html (docs/ for Pages)"
