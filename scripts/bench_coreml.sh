#!/bin/bash
# Compares Core ML compute units / precisions for inswapper. Needs convert_coreml.sh first. Log: bench.log
set -uo pipefail
cd "$(dirname "$0")/.."
MODELS="${1:-$HOME/Library/Application Support/FaceSwapMac/models}"
exec > >(tee bench.log) 2>&1
[ -x .venv-coreml/bin/python ] || { echo "Сначала запусти scripts/convert_coreml.sh"; exit 1; }
echo "Подключи зарядку и закрой тяжёлые приложения — замер займёт 2-3 минуты."
PYTHONWARNINGS=ignore .venv-coreml/bin/python scripts/bench_coreml.py "$MODELS"
echo "== Готово. Лог: bench.log"
