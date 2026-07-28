#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

bash -n scripts/*.sh
python3 -m compileall -q scripts tests
python3 scripts/check_environment.py
pytest -q tests

cd ..
pytest -q test/unit test/test_hook_demo.py test/test_simulation_time_predictor.py
