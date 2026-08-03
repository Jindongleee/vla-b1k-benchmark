#!/usr/bin/env bash
# 세션 A의 슬롯 허가(MSG_TO_RLC.md 삭제) + GPU 조용해짐을 기다렸다가
# GR00T B1K closed-loop 평가를 자동 실행 (서버 GPU0 + OmniGibson GPU1)
set -uo pipefail

STATUS_LOG="$HOME/Isaac-GR00T/auto_eval_groot_status.log"
SERVER_LOG="$HOME/Isaac-GR00T/serve_b1k_groot.log"
EVAL_LOG="$HOME/Isaac-GR00T/eval_b1k_groot.log"
COORD="$HOME/CLAUDE_GPU_COORDINATION.md"
MSG="$HOME/locks/MSG_TO_RLC.md"
LOCK0="$HOME/locks/gpu0.groot-eval.lock"
LOCK1="$HOME/locks/gpu1.groot-eval.lock"
CKPT="$HOME/checkpoints/groot-b1k-radio/groot-n17-b1k-turning_on_radio/checkpoint-10000"
OUT="$HOME/b1k_eval_2025/groot_finetuned"
PORT=8002

log() { echo "[$(date '+%F %T')] $*" >> "$STATUS_LOG"; }
coord_msg() { echo "- ($(date '+%F %T'), 세션 B 자동) $*" >> "$COORD"; }

SERVER_PID=""
cleanup() {
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
    rm -f "$LOCK0" "$LOCK1"
}
trap cleanup EXIT

gpu_quiet() {
    local idx=$1
    local used bigproc
    used=$(nvidia-smi -i "$idx" --query-gpu=memory.used --format=csv,noheader,nounits)
    bigproc=$(nvidia-smi -i "$idx" --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '$1>2048' | wc -l)
    [ "$used" -lt 8000 ] && [ "$bigproc" -eq 0 ]
}

log "=== auto_eval_groot 시작. 슬롯 신호(MSG_TO_RLC.md 삭제) + GPU 0/1 조용 대기 ==="

quiet=0
while [ "$quiet" -lt 3 ]; do
    sleep 60
    if [ -f "$MSG" ]; then
        quiet=0; continue   # 아직 요청 미수락
    fi
    rlc_locks=$(ls "$HOME"/locks/gpu*.rlc-*.lock 2>/dev/null | wc -l)
    if [ "$rlc_locks" -eq 0 ] && gpu_quiet 0 && gpu_quiet 1; then
        quiet=$((quiet+1)); log "슬롯 오픈 확인 $quiet/3"
    else
        quiet=0
    fi
done

touch "$LOCK0" "$LOCK1"
log "슬롯 확보. GR00T 서버 시작 (GPU0, port $PORT)"
coord_msg "세션 B GR00T closed-loop 평가 시작 (GPU0 서버 + GPU1 sim, 락 gpu0/gpu1.groot-eval). 예상 ~1.5시간"

# 1) GR00T 정책 서버 (GPU 0)
cd "$HOME/Isaac-GR00T"
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH="$HOME/.local/ffmpeg61/lib" NO_ALBUMENTATIONS_UPDATE=1 \
    uv run --no-sync python examples/B1K/serve_b1k_groot.py \
    --model-path "$CKPT" \
    --dataset-path "$HOME/DATASETS/behavior/b1k-radio-gr00t" \
    --task-name turning_on_radio \
    --port $PORT \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

# 서버 준비 대기 (healthz, 최대 10분: 모델 로드 포함)
for i in $(seq 1 60); do
    sleep 10
    if curl -s -o /dev/null "http://localhost:$PORT/healthz"; then
        log "서버 준비 완료 (${i}0초)"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        log "서버 프로세스 사망 — $SERVER_LOG 확인"
        coord_msg "❌ GR00T 서버 시작 실패. 슬롯 반납"
        exit 1
    fi
done

# 2) OmniGibson eval 클라이언트 (GPU 1)
log "eval 클라이언트 시작 (GPU1)"
mkdir -p "$OUT"
cd "$HOME/BEHAVIOR-1K-2025"
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$HOME/BEHAVIOR-1K-2025/joylo" OMNI_KIT_ACCEPT_EULA=YES \
    "$HOME/openpi-comet/.venv/bin/python" OmniGibson/omnigibson/learning/eval.py \
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
    log "=== 평가 완료 (exit 0, metrics ${n_metrics}개). 결과: $OUT ==="
    coord_msg "✅ 세션 B GR00T 평가 완료 (~$OUT). GPU 락 해제, 슬롯 반납. 고마워!"
else
    log "=== 평가 실패 (exit $rc). $EVAL_LOG 확인 ==="
    coord_msg "❌ 세션 B GR00T 평가 실패 (exit $rc). GPU 락 해제, 슬롯 반납"
fi
exit $rc
