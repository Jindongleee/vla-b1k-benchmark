#!/usr/bin/env bash
# v3: pt12 평가 종료 대기 → checkpoint-2000에서 GR00T 학습 재개
# 조율 규약: ~/CLAUDE_GPU_COORDINATION.md
set -uo pipefail

cd "$HOME/Isaac-GR00T"
STATUS_LOG="$HOME/Isaac-GR00T/auto_train_b1k_status.log"
TRAIN_LOG="$HOME/Isaac-GR00T/train_b1k_radio.log"
MY_LOCK="$HOME/locks/gpu1.groot-train.lock"
COORD="$HOME/CLAUDE_GPU_COORDINATION.md"

log() { echo "[$(date '+%F %T')] $*" >> "$STATUS_LOG"; }
coord_msg() { echo "- ($(date '+%F %T'), 세션 B 자동) $*" >> "$COORD"; }

cleanup() { rm -f "$MY_LOCK"; }
trap cleanup EXIT

other_locks() {
    ls "$HOME"/locks/gpu1.*.lock 2>/dev/null | grep -v "gpu1.groot-train.lock" | wc -l
}

log "=== auto_resume_b1k v3 시작. pt12 평가 종료 대기 후 checkpoint에서 재개 ==="

quiet=0
while [ "$quiet" -lt 3 ]; do
    sleep 60
    used=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)
    bigproc=$(nvidia-smi -i 1 --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '$1>2048' | wc -l)
    locks=$(other_locks)
    if [ "$locks" -eq 0 ] && [ "$used" -lt 8000 ] && [ "$bigproc" -eq 0 ]; then
        quiet=$((quiet+1)); log "GPU1 quiet check $quiet/3 (used=${used}MiB)"
    else
        quiet=0
        log "GPU1 busy (used=${used}MiB, bigproc=${bigproc}, locks=${locks}) — 대기"
    fi
done

touch "$MY_LOCK"
log "GPU1 확보 + 락 생성. checkpoint에서 학습 재개"
coord_msg "GR00T 학습 재개 (checkpoint-2000부터, 락: gpu1.groot-train.lock). 완료까지 GPU1 사용 예정 (~4.5시간)"

export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH="$HOME/.local/ffmpeg61/lib"
export WANDB_MODE=offline
export NO_ALBUMENTATIONS_UPDATE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run --no-sync python gr00t/experiment/launch_finetune.py \
    --base-model-path "$HOME/checkpoints/GR00T-N1.7-3B" \
    --dataset-path "$HOME/DATASETS/behavior/b1k-radio-gr00t" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/B1K/b1k_r1pro_config.py \
    --num-gpus 1 \
    --global-batch-size 8 \
    --gradient-accumulation-steps 4 \
    --dataloader-num-workers 4 \
    --output-dir "$HOME/checkpoints/groot-b1k-radio" \
    --experiment-name groot-n17-b1k-turning_on_radio \
    --max-steps 10000 --save-steps 1000 --save-total-limit 3 \
    --resume-from-checkpoint \
    >> "$TRAIN_LOG" 2>&1
rc=$?
if [ $rc -eq 0 ]; then
    log "=== 본 학습 완료 (exit 0). 체크포인트: ~/checkpoints/groot-b1k-radio ==="
    coord_msg "✅ GR00T 본 학습 완료. GPU1 락 해제. 이제 GPU1 자유롭게 사용 가능"
else
    log "=== 본 학습 실패 (exit $rc). $TRAIN_LOG 확인 필요 ==="
    coord_msg "❌ GR00T 학습 실패 (exit $rc). GPU1 락 해제됨"
fi
exit $rc
