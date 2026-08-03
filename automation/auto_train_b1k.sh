#!/usr/bin/env bash
# B1K 평가 종료 + GPU1 락 해제를 기다렸다가 GR00T 파인튜닝을 자동 시작 (v2: 락 규약 지원)
# 조율 규약: ~/CLAUDE_GPU_COORDINATION.md
# 로그: ~/Isaac-GR00T/auto_train_b1k_status.log (상태) / train_b1k_radio.log (학습)
set -uo pipefail

cd "$HOME/Isaac-GR00T"
STATUS_LOG="$HOME/Isaac-GR00T/auto_train_b1k_status.log"
TRAIN_LOG="$HOME/Isaac-GR00T/train_b1k_radio.log"
DRYRUN_LOG="$HOME/Isaac-GR00T/dryrun_b1k_radio.log"
MY_LOCK="$HOME/locks/gpu1.groot-train.lock"
COORD="$HOME/CLAUDE_GPU_COORDINATION.md"

log() { echo "[$(date '+%F %T')] $*" >> "$STATUS_LOG"; }
coord_msg() { echo "- ($(date '+%F %T'), 세션 B 자동) $*" >> "$COORD"; }

cleanup() { rm -f "$MY_LOCK"; }
trap cleanup EXIT

other_locks() {
    ls "$HOME"/locks/gpu1.*.lock 2>/dev/null | grep -v "gpu1.groot-train.lock" | wc -l
}

log "=== auto_train_b1k v2 시작. GPU1 확보 대기 (락 규약 + 사용량 기준) ==="

# 1) GPU 1이 조용하고 다른 락이 없는 상태가 3분 연속 유지될 때까지 대기
quiet=0
while [ "$quiet" -lt 3 ]; do
    sleep 60
    used=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)
    bigproc=$(nvidia-smi -i 1 --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '$1>2048' | wc -l)
    locks=$(other_locks)
    if [ "$locks" -eq 0 ] && [ "$used" -lt 8000 ] && [ "$bigproc" -eq 0 ]; then
        quiet=$((quiet+1)); log "GPU1 quiet check $quiet/3 (used=${used}MiB, locks=0)"
    else
        quiet=0
        log "GPU1 busy (used=${used}MiB, bigproc=${bigproc}, other_locks=${locks}) — 대기"
    fi
done

# 2) 락 선점
touch "$MY_LOCK"
log "GPU1 확보 + 락 생성($MY_LOCK). 드라이런(20 step) 시작"
coord_msg "GPU1 확보, GR00T 드라이런 시작 (락: gpu1.groot-train.lock)"

export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH="$HOME/.local/ffmpeg61/lib"
export WANDB_MODE=offline
export NO_ALBUMENTATIONS_UPDATE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON_ARGS=(
    --base-model-path "$HOME/checkpoints/GR00T-N1.7-3B"
    --dataset-path "$HOME/DATASETS/behavior/b1k-radio-gr00t"
    --embodiment-tag NEW_EMBODIMENT
    --modality-config-path examples/B1K/b1k_r1pro_config.py
    --num-gpus 1
    --global-batch-size 8
    --gradient-accumulation-steps 4
    --dataloader-num-workers 4
)

# 3) 드라이런
uv run --no-sync python gr00t/experiment/launch_finetune.py \
    "${COMMON_ARGS[@]}" \
    --output-dir /tmp/groot_dryrun_b1k \
    --max-steps 20 --save-steps 100000 \
    > "$DRYRUN_LOG" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    log "드라이런 실패 (exit $rc) — 본학습 시작 안 함. $DRYRUN_LOG 확인 필요"
    coord_msg "GR00T 드라이런 실패 (exit $rc) — GPU1 락 해제함. 로그: dryrun_b1k_radio.log"
    exit 1
fi
log "드라이런 통과. 본 학습(10k step) 시작"
coord_msg "GR00T 드라이런 통과, 본 학습 시작 (10k step, 예상 6~12시간)"
rm -rf /tmp/groot_dryrun_b1k

# 4) 본 학습: 10k step, effective batch 32 (8 x accum 4)
uv run --no-sync python gr00t/experiment/launch_finetune.py \
    "${COMMON_ARGS[@]}" \
    --output-dir "$HOME/checkpoints/groot-b1k-radio" \
    --experiment-name groot-n17-b1k-turning_on_radio \
    --max-steps 10000 --save-steps 1000 --save-total-limit 3 \
    > "$TRAIN_LOG" 2>&1
rc=$?
if [ $rc -eq 0 ]; then
    log "=== 본 학습 완료 (exit 0). 체크포인트: ~/checkpoints/groot-b1k-radio ==="
    coord_msg "✅ GR00T 본 학습 완료. GPU1 락 해제. 체크포인트: ~/checkpoints/groot-b1k-radio"
else
    log "=== 본 학습 실패 (exit $rc). $TRAIN_LOG 확인 필요 ==="
    coord_msg "❌ GR00T 본 학습 실패 (exit $rc). GPU1 락 해제. 로그: train_b1k_radio.log"
fi
exit $rc
