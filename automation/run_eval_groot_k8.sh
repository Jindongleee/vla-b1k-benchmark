#!/usr/bin/env bash
# GR00T B1K 재평가: denoising K=8 ablation (GPU 1, port 8003)
set -uo pipefail

STATUS_LOG="$HOME/Isaac-GR00T/auto_eval_groot_status.log"
SERVER_LOG="$HOME/Isaac-GR00T/serve_b1k_groot_k8.log"
EVAL_LOG="$HOME/Isaac-GR00T/eval_b1k_groot_k8.log"
COORD="$HOME/CLAUDE_GPU_COORDINATION.md"
LOCK="$HOME/locks/gpu1.groot-eval-k8.lock"
CKPT="$HOME/checkpoints/groot-b1k-radio/groot-n17-b1k-turning_on_radio/checkpoint-10000"
OUT="$HOME/b1k_eval_2025/groot_finetuned_k8"
PORT=8003

log() { echo "[$(date '+%F %T')] $*" >> "$STATUS_LOG"; }
coord_msg() { echo "- ($(date '+%F %T'), 세션 B 자동) $*" >> "$COORD"; }

SERVER_PID=""
cleanup() {
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
    rm -f "$LOCK"
}
trap cleanup EXIT

touch "$LOCK"
log "=== run_eval_groot_k8: K=8 재평가 시작 (GPU1, port $PORT) ==="

cd "$HOME/Isaac-GR00T"
CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH="$HOME/.local/ffmpeg61/lib" NO_ALBUMENTATIONS_UPDATE=1 \
    uv run --no-sync python examples/B1K/serve_b1k_groot.py \
    --model-path "$CKPT" \
    --dataset-path "$HOME/DATASETS/behavior/b1k-radio-gr00t" \
    --task-name turning_on_radio \
    --denoising-steps 8 \
    --port $PORT \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

for i in $(seq 1 60); do
    sleep 10
    if curl -s -o /dev/null "http://localhost:$PORT/healthz"; then
        log "K=8 서버 준비 완료 (~${i}0초)"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        log "K=8 서버 시작 실패 — $SERVER_LOG 확인"
        coord_msg "❌ GR00T K=8 서버 시작 실패. gpu1.groot-eval-k8.lock 해제"
        exit 1
    fi
done

log "K=8 eval 클라이언트 시작 (GPU1, 2025 순수 환경 venv)"
mkdir -p "$OUT"
cd "$HOME/BEHAVIOR-1K-2025"
CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=YES \
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
    log "=== K=8 평가 완료 (exit 0, metrics ${n_metrics}개): $OUT ==="
    coord_msg "✅ 세션 B GR00T K=8 재평가 완료 ($OUT). gpu1.groot-eval-k8.lock 해제"
else
    log "=== K=8 평가 실패 (exit $rc): $EVAL_LOG ==="
    coord_msg "❌ 세션 B GR00T K=8 재평가 실패 (exit $rc). gpu1.groot-eval-k8.lock 해제"
fi
exit $rc
