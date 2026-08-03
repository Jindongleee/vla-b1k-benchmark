#!/usr/bin/env python3
"""eval_results/metrics/<policy>/*.json 을 읽어 정책별 성공률 표를 재생성한다.

성공 판정: q_score.final == 1.0 (OmniGibson closed-loop 평가 기준)
RESULTS.md 의 표는 이 스크립트의 출력으로 검증할 수 있다.

사용:
    python scripts/aggregate_metrics.py
    python scripts/aggregate_metrics.py --metrics-dir eval_results/metrics --markdown
"""

import argparse
import json
import re
from pathlib import Path


def instance_id(path: Path) -> int:
    """turning_on_radio_109_0.json -> 109"""
    m = re.search(r"_(\d+)_\d+\.json$", path.name)
    return int(m.group(1)) if m else -1


def collect(policy_dir: Path) -> dict:
    successes, total = [], 0
    for f in sorted(policy_dir.glob("*.json"), key=instance_id):
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        total += 1
        if d.get("q_score", {}).get("final") == 1.0:
            successes.append(instance_id(f))
    return {"policy": policy_dir.name, "success": successes, "total": total}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", default="eval_results/metrics")
    ap.add_argument("--markdown", action="store_true", help="마크다운 표로 출력")
    args = ap.parse_args()

    root = Path(args.metrics_dir)
    if not root.is_dir():
        raise SystemExit(f"경로 없음: {root}")

    rows = [collect(d) for d in sorted(root.iterdir()) if d.is_dir()]
    rows.sort(key=lambda r: (-len(r["success"]), r["policy"]))

    if args.markdown:
        print("| 정책 | 성공률 | 성공 인스턴스 |")
        print("|---|---|---|")
        for r in rows:
            inst = ", ".join(map(str, r["success"])) or "—"
            print(f"| `{r['policy']}` | {len(r['success'])}/{r['total']} | {inst} |")
    else:
        for r in rows:
            inst = ", ".join(map(str, r["success"])) or "—"
            print(f"{r['policy']:<28} {len(r['success'])}/{r['total']:<3} [{inst}]")


if __name__ == "__main__":
    main()
