#!/usr/bin/env bash
# pt50 LoRA 학습 종료(락 해제 + GPU1 한산)를 기다렸다가 K=8 재평가 자동 실행
set -uo pipefail
STATUS_LOG="$HOME/Isaac-GR00T/auto_eval_groot_status.log"
log() { echo "[$(date '+%F %T')] $*" >> "$STATUS_LOG"; }

log "=== auto_k8_watcher: pt50 학습 종료 대기 ==="
quiet=0
while [ "$quiet" -lt 3 ]; do
    sleep 120
    locks=$(ls "$HOME"/locks/gpu*.pt50.lock "$HOME"/locks/gpu1.*.lock 2>/dev/null | grep -v groot-eval-k8 | wc -l)
    used=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)
    bigproc=$(nvidia-smi -i 1 --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '$1>2048' | wc -l)
    if [ "$locks" -eq 0 ] && [ "$used" -lt 8000 ] && [ "$bigproc" -eq 0 ]; then
        quiet=$((quiet+1)); log "K8 대기: GPU1 quiet $quiet/3"
    else
        quiet=0
    fi
done
log "GPU1 확보 — K=8 재평가 시작"
exec bash "$HOME/Isaac-GR00T/run_eval_groot_k8.sh"
