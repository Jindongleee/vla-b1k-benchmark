# 데이터 및 모델 준비

이 저장소는 데이터셋과 체크포인트를 포함하지 않는다. 각 배포처의 라이선스 조건을 따라 직접 받아야 한다.

## 1. BEHAVIOR-1K / OmniGibson

평가 환경과 태스크. 설치 및 데이터 접근 방법은 공식 문서를 따른다.

- 프로젝트: https://behavior.stanford.edu/
- OmniGibson: https://github.com/StanfordVL/OmniGibson

**재배포 금지**: 데이터셋과 3D 에셋은 별도 라이선스 동의가 필요하며, 이 저장소는 이를 재배포하지 않는다.

## 2. GR00T N1.7 베이스 모델

- https://github.com/NVIDIA/Isaac-GR00T
- 체크포인트는 NVIDIA 공식 배포처(HuggingFace)에서 받는다.

기대 경로: `$HOME/checkpoints/GR00T-N1.7-3B`

## 3. 학습용 데이터셋 변환

BEHAVIOR-1K 2025 challenge demos(LeRobot v2.1)를 GR00T 학습 포맷으로 변환한 뒤,
`meta/modality.json`이 `b1k/b1k_r1pro_config.py`의 슬라이스 정의와 일치해야 한다.

기대 경로: `$HOME/DATASETS/behavior/b1k-radio-gr00t`

### state[256] / action[23] 레이아웃

packed 벡터의 인덱스 레이아웃은 OmniGibson `learning/utils/eval_utils.py`의
standard-track-legal 키 정의를 따른다.

action 23-dim:

```
base_vel(3) + torso(4) + left_arm(7) + left_gripper(1) + right_arm(7) + right_gripper(1)
```

전부 absolute joint/velocity command이므로 모든 키가 `ActionRepresentation.ABSOLUTE`.

## 4. 비교 baseline 체크포인트

BEHAVIOR-1K 2025 Challenge 참가팀들이 공개한 체크포인트(pt12 / pt50 / RLC checkpoint_2)를
평가에만 사용했다. 각 팀의 공개 배포처에서 받아야 한다.
