"""GR00T policy websocket server for BEHAVIOR-1K evaluation.

B1K의 OmniGibson eval 클라이언트(`omnigibson.learning.eval policy=websocket`)가
기대하는 프로토콜(msgpack websocket, act(obs) -> (23,) 액션)로 Gr00tPolicy를 서빙한다.
2위팀 openpi-comet의 serve_b1k.py와 동일한 역할의 GR00T 버전.

사용 예:
    CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$HOME/.local/ffmpeg61/lib \
    uv run --no-sync python examples/B1K/serve_b1k_groot.py \
        --model-path ~/checkpoints/groot-b1k-radio/groot-n17-b1k-turning_on_radio/checkpoint-10000 \
        --dataset-path ~/DATASETS/behavior/b1k-radio-gr00t \
        --task-name turning_on_radio --port 8001
"""

import dataclasses
import json
import logging
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import tyro

sys.path.append(str(Path(__file__).parent))
from b1k_network_utils import WebsocketPolicyServer  # noqa: E402

from gr00t.policy.gr00t_policy import Gr00tPolicy  # noqa: E402

logger = logging.getLogger("serve_b1k_groot")

# R1Pro proprio[256] 슬라이스 (OmniGibson learning/utils/eval_utils.py 기준,
# 학습 시 meta/modality.json과 동일해야 함)
STATE_SLICES = {
    "base_qvel": slice(253, 256),
    "trunk_qpos": slice(236, 240),
    "left_arm_qpos": slice(158, 165),
    "left_gripper_qpos": slice(193, 195),
    "right_arm_qpos": slice(197, 204),
    "right_gripper_qpos": slice(232, 234),
}

# B1K eval 관측 키 → GR00T video 모달리티 키
CAMERA_KEYS = {
    "head": "robot_r1::robot_r1:zed_link:Camera:0::rgb",
    "left_wrist": "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb",
    "right_wrist": "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb",
}

# 액션 23차원 조립 순서 (학습 modality config의 action 키 순서와 동일)
ACTION_KEYS = ["base", "torso", "left_arm", "left_gripper", "right_arm", "right_gripper"]


def _to_uint8_rgb(img) -> np.ndarray:
    arr = np.asarray(img)
    arr = arr[..., :3]  # RGBA -> RGB
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


class Gr00tB1KWrapper:
    """Receding-horizon 래퍼: replan_interval 스텝마다 GR00T로 16-스텝 청크를 재계획."""

    def __init__(self, policy: Gr00tPolicy, task_prompt: str, replan_interval: int = 16):
        self.policy = policy
        self.task_prompt = task_prompt
        self.replan_interval = replan_interval
        self.action_queue: deque = deque()
        self.step_counter = 0
        self.last_chunk: np.ndarray | None = None  # (H, 23) 추론 실패 시 재사용

    def reset(self):
        self.action_queue.clear()
        self.step_counter = 0
        self.last_chunk = None
        logger.info("policy reset")

    def _build_observation(self, obs: dict) -> dict:
        video = {}
        for gr00t_key, b1k_key in CAMERA_KEYS.items():
            frame = _to_uint8_rgb(obs[b1k_key])  # (H, W, 3)
            video[gr00t_key] = frame[None, None]  # (B=1, T=1, H, W, 3)

        proprio = np.asarray(obs["robot_r1::proprio"], dtype=np.float32).reshape(-1)
        assert proprio.shape[0] == 256, f"proprio dim {proprio.shape[0]} != 256"
        state = {k: proprio[s][None, None] for k, s in STATE_SLICES.items()}  # (1, 1, D)

        return {
            "video": video,
            "state": state,
            "language": {"task": [[self.task_prompt]]},
        }

    def _predict_chunk(self, obs: dict) -> np.ndarray:
        observation = self._build_observation(obs)
        result = self.policy.get_action(observation)
        action_dict = result[0] if isinstance(result, tuple) else result
        parts = []
        for key in ACTION_KEYS:
            arr = np.asarray(action_dict[key], dtype=np.float32)  # (1, H, D)
            assert arr.ndim == 3 and arr.shape[0] == 1, f"{key}: unexpected shape {arr.shape}"
            parts.append(arr[0])  # (H, D)
        chunk = np.concatenate(parts, axis=-1)  # (H, 23)
        assert chunk.shape[-1] == 23, f"action dim {chunk.shape[-1]} != 23"
        return chunk

    def act(self, obs: dict) -> torch.Tensor:
        if len(self.action_queue) == 0:
            try:
                chunk = self._predict_chunk(obs)
                self.last_chunk = chunk
            except Exception:
                if self.last_chunk is None:
                    raise
                logger.exception("inference failed at step %d — 직전 청크 재사용", self.step_counter)
                chunk = self.last_chunk
            for a in chunk[: self.replan_interval]:
                self.action_queue.append(a)
        action = self.action_queue.popleft()
        self.step_counter += 1
        if self.step_counter % 150 == 0:
            logger.info("step %d", self.step_counter)
        return torch.from_numpy(np.ascontiguousarray(action))


@dataclasses.dataclass
class Args:
    model_path: str
    dataset_path: str = str(Path.home() / "DATASETS/behavior/b1k-radio-gr00t")
    task_name: str = "turning_on_radio"
    task_prompt: str | None = None  # None이면 dataset meta/tasks.jsonl에서 조회
    embodiment_tag: str = "NEW_EMBODIMENT"
    port: int = 8001
    device: str = "cuda:0"
    replan_interval: int = 16  # 16 = 청크 전체 실행 후 재계획, 8 = 절반 실행(RHC)
    denoising_steps: int = 4  # flow matching K (기본 4 = 논문/학습 설정)


def resolve_task_prompt(args: Args) -> str:
    if args.task_prompt:
        return args.task_prompt
    tasks_file = Path(args.dataset_path) / "meta/tasks.jsonl"
    with open(tasks_file) as f:
        for line in f:
            t = json.loads(line)
            if t.get("task_name") == args.task_name:
                return t["task"]
    raise ValueError(f"task_name '{args.task_name}' not in {tasks_file}; --task-prompt로 직접 지정 필요")


def main(args: Args) -> None:
    prompt = resolve_task_prompt(args)
    logger.info("task prompt: %s", prompt)

    policy = Gr00tPolicy(
        embodiment_tag=args.embodiment_tag,
        model_path=args.model_path,
        device=args.device,
    )
    if args.denoising_steps != 4:
        n_set = 0
        targets = [policy.model] + list(policy.model.modules())
        for m in targets:
            if hasattr(m, "num_inference_timesteps"):
                m.num_inference_timesteps = args.denoising_steps
                n_set += 1
        assert n_set > 0, "num_inference_timesteps 속성을 가진 모듈을 찾지 못함"
        logger.info("denoising steps: 4 -> %d (%d개 모듈)", args.denoising_steps, n_set)
    wrapper = Gr00tB1KWrapper(policy, task_prompt=prompt, replan_interval=args.replan_interval)

    logger.info("serving on 0.0.0.0:%d (replan_interval=%d)", args.port, args.replan_interval)
    server = WebsocketPolicyServer(policy=wrapper, host="0.0.0.0", port=args.port, metadata={})
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
