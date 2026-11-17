"""
learning/downstream_task/main.py
下流回帰タスク (Step 7-1: Base Model [Visual/Vital] ➔ Step 7-2: Meta Model) CLI エントリーポイント
"""

from __future__ import annotations
import argparse
import copy
from pathlib import Path
import sys
import time

_CURRENT_DIR = Path(__file__).resolve().parent
_LEARNING_DIR = _CURRENT_DIR.parent
for _p in (_CURRENT_DIR, _LEARNING_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from config_loader import load_config, log
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("downstream_task")

    def load_config(config_filename: str = "config.yaml") -> dict:
        return {}

try:
    from .pipeline import DownstreamTaskPipeline
    from .threshold_decision import SUPPORTED_THRESHOLD_CRITERIA
    from .base_model import DEFAULT_BASE_MODELS
    from .meta_model import DEFAULT_META_MODELS
except (ImportError, ValueError):
    from downstream_task.pipeline import DownstreamTaskPipeline
    from downstream_task.threshold_decision import SUPPORTED_THRESHOLD_CRITERIA
    from downstream_task.base_model import DEFAULT_BASE_MODELS
    from downstream_task.meta_model import DEFAULT_META_MODELS


def main():
    parser = argparse.ArgumentParser(
        description="下流タスク LOSO-CV パイプライン (Step 7-0: Threshold Decision [Youden's J] ➔ Step 7-1: Base Model ➔ Step 7-2: Meta Model)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="設定YAMLファイル名 (デフォルト: config.yaml)"
    )
    parser.add_argument(
        "--stage",
        nargs="+",
        choices=[
            "all",
            "threshold_decision", "threshold", "th",
            "base_model", "base",
            "meta_model", "meta",
            "downstream_task", "downstream", "loso_cv", "loso",
        ],
        default=["all"],
        help="実行ステージ: threshold_decision (学習前閾値決定) / base_model / meta_model / all"
    )
    parser.add_argument(
        "--window-size",
        "--window-sizes",
        "-w",
        nargs="+",
        default=None,
        help="ウィンドウサイズの指定 (例: 10s / 5s 10s 15s / all)"
    )
    parser.add_argument(
        "--criteria",
        nargs="+",
        default=None,
        help=f"閾値決定基準 (デフォルト: youden_j, 選択肢: {list(SUPPORTED_THRESHOLD_CRITERIA)})"
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=["visual", "vital"],
        default=None,
        help="Base Model 学習対象モダリティ (デフォルト: visual vital)"
    )
    parser.add_argument(
        "--base-models",
        "--models",
        dest="base_models",
        nargs="+",
        default=None,
        help=f"Base Model 学習対象アルゴリズム (デフォルト: {DEFAULT_BASE_MODELS})"
    )
    parser.add_argument(
        "--meta-models",
        nargs="+",
        default=None,
        help=f"Meta Model 学習対象アルゴリズム (デフォルト: {DEFAULT_META_MODELS})"
    )
    parser.add_argument(
        "--visual-model",
        type=str,
        default=None,
        help="Meta Model で使用する Visual ベースモデル指定 (未指定時は config.yaml を参照)"
    )
    parser.add_argument(
        "--vital-model",
        type=str,
        default=None,
        help="Meta Model で使用する Vital ベースモデル指定 (未指定時は config.yaml を参照)"
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=None,
        help="Base Model の Early Stopping ラウンド数の上書き指定"
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=None,
        help="各クラスタに対する重み w_k (例: --weights 1.0 0.0)"
    )
    parser.add_argument(
        "--covariance-type",
        "--covariance",
        "--cov",
        type=str,
        default=None,
        choices=["full", "tied", "diag", "spherical"],
        help="Fold 内 GMM の共分散タイプ"
    )
    parser.add_argument(
        "--no-subj-norm",
        action="store_true",
        help="被験者別Zスコア標準化を無効化する場合に指定"
    )

    args = parser.parse_args()

    ALL_SUPPORTED_WINDOWS = ["5s", "10s", "15s"]
    base_cfg = load_config(args.config)

    if args.window_size:
        raw_list = [w.strip() for w in args.window_size]
        if "all" in [w.lower() for w in raw_list]:
            target_windows = ALL_SUPPORTED_WINDOWS
        else:
            target_windows = []
            for w in raw_list:
                w_lower = w.lower()
                if w_lower in ["5", "5s"]:
                    target_windows.append("5s")
                elif w_lower in ["10", "10s"]:
                    target_windows.append("10s")
                elif w_lower in ["15", "15s"]:
                    target_windows.append("15s")
                elif w_lower == "all":
                    target_windows.extend(ALL_SUPPORTED_WINDOWS)
                else:
                    target_windows.append(w)
            target_windows = list(dict.fromkeys(target_windows))
    else:
        default_ws = base_cfg.get("dataset", {}).get("window_size", "10s")
        target_windows = [default_ws]

    stages_set = {s.lower() for s in args.stage}
    run_th = any(
        s in stages_set
        for s in (
            "all",
            "threshold_decision", "threshold", "th",
            "base_model", "base",
            "downstream_task", "downstream", "loso_cv", "loso",
        )
    )
    run_base = any(
        s in stages_set
        for s in ("all", "base_model", "base", "downstream_task", "downstream", "loso_cv", "loso")
    )
    run_meta = any(
        s in stages_set
        for s in ("all", "meta_model", "meta", "downstream_task", "downstream", "loso_cv", "loso")
    )

    log.info("=" * 75)
    log.info(" 🚀 [Downstream Task] LOSO-CV 学習前閾値決定 (Youden's J) + 2段階学習・推論パイプライン開始")
    log.info(f" ⏱️ 対象ウィンドウサイズ : {target_windows}")
    log.info(f" 🎯 実行ステージ         : ThresholdDecision={run_th}, Base={run_base}, Meta={run_meta}")
    log.info("=" * 75)

    total_start_time = time.time()

    for idx, ws in enumerate(target_windows, 1):
        if len(target_windows) > 1:
            log.info("=" * 75)
            log.info(f" 🔄 [{idx}/{len(target_windows)}] ウィンドウサイズ: {ws} の処理を開始します")
            log.info("=" * 75)

        cfg = copy.deepcopy(base_cfg)
        cfg.setdefault("dataset", {})["window_size"] = ws

        if args.early_stopping is not None:
            cfg.setdefault("downstream_task", {}).setdefault("base_model", {}).setdefault(
                "early_stopping", {}
            )["stopping_rounds"] = args.early_stopping

        pipeline = DownstreamTaskPipeline(cfg)
        th_res = None
        if run_th:
            th_res = pipeline.run_threshold_decision(
                criteria=args.criteria,
                weights=args.weights,
                covariance_type=args.covariance_type,
                apply_subject_norm=not args.no_subj_norm,
            )
        if run_base:
            pipeline.run_base_models(
                modalities=args.modalities,
                models=args.base_models,
                weights=args.weights,
                covariance_type=args.covariance_type,
                apply_subject_norm=not args.no_subj_norm,
                precomputed_folds=th_res,
            )
        if run_meta:
            pipeline.run_meta_models(
                models=args.meta_models,
                visual_model=args.visual_model,
                vital_model=args.vital_model,
            )

    total_elapsed = time.time() - total_start_time
    log.info("=" * 75)
    log.info(f" 🎉 全ウィンドウサイズ ({target_windows}) の処理が完了しました (総所要時間: {total_elapsed:.2f} 秒)")
    log.info("=" * 75)


if __name__ == "__main__":
    main()
