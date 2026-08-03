# 상세 결과

BEHAVIOR-1K `turning_on_radio`, OmniGibson closed-loop, 10개 고정 인스턴스.
평가 완료: 2026-07-14.

## 평가 조건

| 항목 | 값 |
|---|---|
| 태스크 | `turning_on_radio` (BEHAVIOR-1K) |
| 로봇 | Galaxea R1 Pro (action 23-dim, state 256-dim packed) |
| 인스턴스 | 109, 139, 181, 187, 197, 203, 211, 214, 242, 295 (10개 고정) |
| 관측 | RGBWrapper (head / left_wrist / right_wrist) |
| 서빙 | msgpack websocket 정책 서버 |
| 청크 | 16-step, receding horizon (`replan_interval` 마다 재계획) |
| 성공 판정 | `q_score.final == 1.0` |
| 타임아웃 | 4300 simulator steps |

## 정책 목록

메트릭 디렉터리명과 정책의 대응.

| `eval_results/metrics/` | 정책 | 설명 |
|---|---|---|
| `pt50_lora_finetuned` | pt50 + LoRA 10k | 50-task 사전학습 체크포인트에 LoRA 파인튜닝 |
| `lora_finetuned` | pt12 + LoRA 10k | 12-task 사전학습 체크포인트에 LoRA 파인튜닝 |
| `pt50_original` | pt50 원본 | 50-task 사전학습, 파인튜닝 없음 |
| `pt12_original` | pt12 원본 | 12-task 사전학습, 파인튜닝 없음 |
| `rlc_checkpoint2_original` | RLC ckpt2 원본 | 공개 체크포인트, 파인튜닝 없음 |
| `rlc_finetuned_step5000` | RLC 풀 FT 5k | batch 32 축소 적용 |
| `rlc_finetuned_step20000` | RLC 풀 FT 20k | batch 32 축소 적용, 완주 |
| `groot_finetuned` | GR00T N1.7 FT (K=4) | 10k step, denoising step 4 (기본) |
| `groot_finetuned_k8` | GR00T N1.7 FT (K=8) | 동일 체크포인트, denoising step 8 |

## 성공률

```
pt50_lora_finetuned          4/10  [109, 139, 203, 242]
lora_finetuned               3/10  [211, 214, 242]
pt50_original                3/10  [181, 211, 242]
rlc_checkpoint2_original     3/10  [139, 181, 187]
pt12_original                1/10  [139]
groot_finetuned              0/10  [—]
groot_finetuned_k8           0/10  [—]
rlc_finetuned_step20000      0/10  [—]
rlc_finetuned_step5000       0/10  [—]
```

`python scripts/aggregate_metrics.py` 로 재생성 가능.

## 인스턴스별 교차표

| 인스턴스 | 성공한 정책 수 | 성공 정책 |
|---|---|---|
| 242 | 3 | pt50+LoRA, pt12+LoRA, pt50 원본 |
| 139 | 3 | pt50+LoRA, pt12 원본, RLC ckpt2 |
| 211 | 2 | pt12+LoRA, pt50 원본 |
| 181 | 2 | pt50 원본, RLC ckpt2 |
| 109 | 1 | pt50+LoRA |
| 203 | 1 | pt50+LoRA |
| 214 | 1 | pt12+LoRA |
| 187 | 1 | RLC ckpt2 |
| 197 | 0 | — |
| 295 | 0 | — |

197과 295는 9개 정책 전부 실패했다. 평균 성공률만 보고 정책을 비교하면 이 구조가 보이지 않는다.

## 학습 설정

### GR00T N1.7 파인튜닝

| 항목 | 값 |
|---|---|
| 베이스 | GR00T-N1.7-3B |
| step | 10,000 |
| global batch | 8 |
| gradient accumulation | 4 (effective batch 32) |
| embodiment tag | `NEW_EMBODIMENT` |
| modality config | `b1k/b1k_r1pro_config.py` |
| 멀티GPU | DDP (`patches/training_config_ddp.patch`) |
| dataloader workers | 4 |

### RLC 풀 파인튜닝

| 항목 | 원 레시피 | 본 실험 |
|---|---|---|
| global batch | 2048 | **32** |
| learning rate | 1e-4 | 2.5e-5 (sqrt 스케일링) |
| step | — | 5,000 / 20,000 |

배치 축소는 가용 GPU 제약에 따른 것이다. **따라서 이 결과는 원 레시피의 성능이 아니라,
축소 조건에서 재현을 시도한 결과다.**

## 관찰

### loss와 closed-loop 성능의 괴리

RLC 풀 파인튜닝에서 training loss는 0.50까지 정상 하강했으나 성공률은 3/10 → 0/10으로 붕괴했다.
LR sqrt 스케일링으로도 복구되지 않았다.

실패 양상(영상 확인): 라디오 접근과 팔 뻗기까지는 수행하나 스위치 조작에 실패한 뒤 배회,
전 에피소드가 4300스텝 타임아웃.

같은 체크포인트의 before(3/10) / after(0/10) 비교이므로, 성능 저하의 원인을 파인튜닝 과정으로
좁힐 수 있다. catastrophic forgetting으로 추정한다.

### 0/10에도 두 가지 종류가 있다

`q_score`는 0/1 이진값이라 부분점수가 없다. 성공률만 보면 0/10인 4개 정책이 전부 같아 보인다.
그러나 실패 에피소드의 **이동량**(`agent_distance`)을 보면 실패 방식이 명확히 갈린다.

| 정책 | 성공 | base | left arm | right arm |
|---|---|---|---|---|
| pt50 + LoRA | 4/10 | 2.04 | 11.00 | 11.50 |
| RLC ckpt2 원본 | 3/10 | 1.65 | 4.34 | 5.82 |
| pt50 원본 | 3/10 | 1.05 | 13.89 | 14.93 |
| pt12 + LoRA | 3/10 | 1.09 | 11.16 | 12.54 |
| pt12 원본 | 1/10 | 1.42 | 11.05 | 13.36 |
| GR00T N1.7 FT (K=8) | 0/10 | 3.14 | 26.47 | 29.38 |
| GR00T N1.7 FT (K=4) | 0/10 | 3.91 | 33.80 | 37.66 |
| RLC 풀 FT 20k | 0/10 | 13.62 | 93.63 | 105.10 |
| RLC 풀 FT 5k | 0/10 | 13.10 | 94.39 | 115.30 |

실패 에피소드만 평균낸 값이다. `python scripts/failure_profile.py` 로 재생성 가능.

세 구간으로 나뉜다.

1. **성공하는 정책의 실패** — base 1~2, 팔 4~14. 태스크를 시도하다 완수하지 못한 것.
2. **GR00T N1.7 파인튜닝** — 팔 26~38. 정상 대비 **2~3배**.
3. **RLC 풀 파인튜닝** — base 13, 팔 94~115. 정상 대비 **8~10배**.

3번은 정책이 의미 있는 행동 분포를 잃고 궤적을 마구 생성한다는 뜻이다.
"loss는 정상인데 성공률이 0"이라는 관찰에 **정책 붕괴**라는 구체적 해석을 붙일 수 있는 근거다.
영상에서 관찰한 "배회" 양상이 수치로도 확인된다.

2번(GR00T)은 붕괴까지 가지는 않았으나 정상 범위를 벗어났다. 10k step / effective batch 32로는
이 태스크에 수렴하지 못했다는 쪽에 가깝다.

### denoising step ablation

동일 체크포인트에 대해 K=4 → K=8. 성공률은 변화 없음(0/10 → 0/10).

다만 이동량은 줄었다 — 팔 33.80/37.66 → 26.47/29.38. denoising step을 늘리면 행동이 다소
안정되지만 성공으로 이어지지는 않았다. **추론 시 샘플링 예산이 아니라 정책 자체의 문제**임을 시사한다.

### LoRA vs 풀 파인튜닝

| | 파인튜닝 전 | 파인튜닝 후 | 방식 |
|---|---|---|---|
| pt12 | 1/10 | 3/10 | LoRA |
| pt50 | 3/10 | 4/10 | LoRA |
| RLC ckpt2 | 3/10 | 0/10 | 풀 FT (batch 32) |

이 데이터 규모·배치 조건에서는 LoRA가 안전했다.

## 원본 데이터

`eval_results/metrics/<정책명>/turning_on_radio_<인스턴스>_0.json`

각 파일 구조:

```json
{
  "agent_distance":            { "base": ..., "left": ..., "right": ... },
  "normalized_agent_distance": { "base": ..., "left": ..., "right": ... },
  "q_score":                   { "final": 1.0 },
  "time":                      { "simulator_steps": 2580,
                                 "simulator_time": 86.0,
                                 "normalized_time": 0.833 }
}
```

평가 영상(~930MB)은 저장소 크기 문제로 포함하지 않았다.
