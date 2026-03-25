#!/bin/bash
set -e

BASE_DIR="/app/jaxmarl/environments/overcooked"
LOG_FILE="$BASE_DIR/layout_log.txt"
ANALYZED_FILE="$BASE_DIR/layout_log_Analyzed.txt"
SEED="${1:-0}"

echo "[1] overcooked_vae.py 실행 중... (seed=$SEED)"
PYTHONPATH=/app python "$BASE_DIR/overcooked_vae.py" --seed "$SEED" > "$LOG_FILE" 2>&1

echo "[2] log_parser.py 실행 중..."
cd /app
python jaxmarl/environments/overcooked/log_parser.py

echo "[3] 완료"
echo "사용한 seed     : $SEED"
echo "로그 파일      : $LOG_FILE"
echo "분석 결과 파일 : $ANALYZED_FILE"