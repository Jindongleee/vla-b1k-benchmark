#!/usr/bin/env python3
"""실패 에피소드의 end-effector / base 이동량으로 실패 양상을 분류한다.

성공률만 보면 0/10은 전부 같아 보이지만, 이동량을 보면 실패 방식이 갈린다.
 - 이동량이 작다  → 태스크를 시도하다 완수하지 못함
 - 이동량이 크다  → 의미 없는 궤적을 생성 (정책 붕괴 의심)

사용:
    python scripts/failure_profile.py
"""

import argparse
import json
from pathlib import Path


def profile(policy_dir: Path) -> dict:
    eps = [json.loads(f.read_text()) for f in sorted(policy_dir.glob("*.json"))]
    if not eps:
        return {}
    n_success = sum(1 for e in eps if e["q_score"]["final"] == 1.0)
    failed = [e for e in eps if e["q_score"]["final"] != 1.0] or eps
    k = len(failed)
    return {
        "policy": policy_dir.name,
        "success": n_success,
        "total": len(eps),
        "base": sum(e["agent_distance"]["base"] for e in failed) / k,
        "left": sum(e["agent_distance"]["left"] for e in failed) / k,
        "right": sum(e["agent_distance"]["right"] for e in failed) / k,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", default="eval_results/metrics")
    args = ap.parse_args()

    rows = [r for d in sorted(Path(args.metrics_dir).iterdir()) if d.is_dir() and (r := profile(d))]
    rows.sort(key=lambda r: -r["success"])

    print(f"{'정책':<28} {'성공':>6}  {'base':>7} {'left':>7} {'right':>7}")
    print("-" * 62)
    for r in rows:
        print(
            f"{r['policy']:<28} {r['success']:>2}/{r['total']:<3}"
            f"  {r['base']:7.2f} {r['left']:7.2f} {r['right']:7.2f}"
        )
    print("\n* 실패 에피소드만 평균낸 이동거리 (agent_distance)")


if __name__ == "__main__":
    main()
