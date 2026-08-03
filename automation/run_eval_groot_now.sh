#!/usr/bin/env bash
# GR00T B1K closed-loop 평가 즉시 실행 (사용자 지시: 전부 GPU 0)
set -uo pipefail

STATUS_LOG="$HOME/Isaac-GR00T/auto_eval_groot_status.log"
SERVER_LOG="$HOME/Isaac-GR00T/serve_b1k_groot.log"
EVAL_LOG="$HOME/Isaac-GR00T/eval_b1k_groot.log"
COORD="$HOME/CLAUDE_GPU_COORDINATION.md"
LOCK0="$HOME/locks/gpu0.groot-eval.lock"
CKPT="$HOME/checkpoints/groot-b1k-radio/groot-n17-b1k-turning_on_radio/checkpoint-10000"
OUT="$HOME/b1k_eval_2025/groot_finetuned"
PORT=8002

log() { echo "[$(date '+%F %T')] $*" >> "$STATUS_LOG"; }
coord_msg() { echo "- ($(date '+%F %T'), 세션 B 자동) $*" >> "$COORD"; }

SERVER_PID=""
cleanup() {
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
    rm -f "$LOCK0"
}
trap cleanup EXIT

log "=== run_eval_groot_now: GPU0에서 즉시 실행 ==="

cd "$HOME/Isaac-GR00T"
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH="$HOME/.local/ffmpeg61/lib" NO_ALBUMENTATIONS_UPDATE=1 \
    uv run --no-sync python examples/B1K/serve_b1k_groot.py \
    --model-path "$CKPT" \
    --dataset-path "$HOME/DATASETS/behavior/b1k-radio-gr00t" \
    --task-name turning_on_radio \
    --port $PORT \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

for i in $(seq 1 60); do
    sleep 10
    if curl -s -o /dev/null "http://localhost:$PORT/healthz"; then
        log "서버 준비 완료 (~${i}0초)"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        log "서버 시작 실패 — $SERVER_LOG 확인"
        coord_msg "❌ GR00T 서버 시작 실패. gpu0.groot-eval.lock 해제"
        exit 1
    fi
done

log "eval 클라이언트 시작 (GPU0, 2025 순수 환경 venv)"
mkdir -p "$OUT"
cd "$HOME/BEHAVIOR-1K-2025"
CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES \
    "$HOME/BEHAVIOR-1K-2025/.venv/bin/python" OmniGibson/omnigibson/learning/eval.py \
    policy=websocket \
    task.name=turning_on_radio \
    model.port=$PORT \
    log_path="$OUT" \
    env_wrapper._target_=omnigibson.learning.wrappers.RGBWrapper \
    > "$EVAL_LOG" 2>&1
rc=$?

kill "$SERVER_PID" 2>/dev/null
SERVER_PID=""

if [ $rc -eq 0 ]; then
    n_metrics=$(ls "$OUT/metrics" 2>/dev/null | wc -l)
    log "=== GR00T 평가 완료 (exit 0, metrics ${n_metrics}개): $OUT ==="
    coord_msg "✅ 세션 B GR00T 평가 완료 ($OUT). gpu0.groot-eval.lock 해제"
else
    log "=== GR00T 평가 실패 (exit $rc): $EVAL_LOG ==="
    coord_msg "❌ 세션 B GR00T 평가 실패 (exit $rc). gpu0.groot-eval.lock 해제"
fi
exit $rc
