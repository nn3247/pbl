"""
Step 7-0: 学習前 閾値決定 (Pre-Learning Threshold Decision) 単体 CLI エントリポイント
(learning/downstream_task/threshold_decision/main.py)

実行例:
  uv run python learning/downstream_task/threshold_decision/main.py
  uv run python learning/downstream_task/threshold_decision/main.py --criteria youden_j
  uv run python learning/downstream_task/threshold_decision/main.py --window 10s
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys

_LEARNING_ROOT = Path(__file__).resolve().parents[2]
if str(_LEARNING_ROOT) not in sys.path:
    sys.path.insert(0, str(_LEARNING_ROOT))

from downstream_task.threshold_decision.pipeline import ThresholdDecisionPipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LOSO-CV 完全リークフリー 学習前 閾値決定パイプライン (Youden's J Index / Balanced Accuracy 最大化)"
    )
    parser.add_argument("--config", type=str, default=None, help="config.yaml のパス")
    parser.add_argument("--window", type=str, default=None, help="ウィンドウサイズ (例: 10s)")
    parser.add_argument(
        "--criteria",
        type=str,
        nargs="+",
        default=None,
        help="閾値決定基準 (youden_j, f1_macro, pr_balance)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="対象モデル名フィルタ (省略時は gmm_soft_label)",
    )
    args = parser.parse_args()

    pipeline = ThresholdDecisionPipeline(config_path=args.config)
    pipeline.run(
        window_size=args.window,
        criteria=args.criteria,
        target_models=args.models,
    )


if __name__ == "__main__":
    main()

