"""
learning/downstream_task/threshold_decision パッケージ初期化モジュール
LOSO-CV 完全データリークフリー 学習前 閾値決定（Threshold Decision）
3大基準:
  1. youden_j (Youden's J index = Sensitivity + Specificity - 1, Balanced Accuracy 最大化) [デフォルト]
  2. f1_macro (F1-Macro 最大化)
  3. pr_balance (Precision-Recall バランス / PR均衡点)
"""

from .criteria import (
    SUPPORTED_THRESHOLD_CRITERIA,
    CRITERION_DISPLAY_NAMES,
    ThresholdSearchResult,
    normalize_criterion_name,
    compute_threshold_curve_metrics,
    find_optimal_threshold,
    find_all_criteria_thresholds,
)
from .runner import run_loso_threshold_tuning, run_prelearning_fold_threshold_tuning
from .reporter import (
    generate_threshold_visualizations,
    save_single_threshold_report,
    save_threshold_decision_summary_report,
)
from .pipeline import ThresholdDecisionPipeline, load_saved_fold_thresholds

__all__ = [
    "SUPPORTED_THRESHOLD_CRITERIA",
    "CRITERION_DISPLAY_NAMES",
    "ThresholdSearchResult",
    "normalize_criterion_name",
    "compute_threshold_curve_metrics",
    "find_optimal_threshold",
    "find_all_criteria_thresholds",
    "run_loso_threshold_tuning",
    "run_prelearning_fold_threshold_tuning",
    "generate_threshold_visualizations",
    "save_single_threshold_report",
    "save_threshold_decision_summary_report",
    "ThresholdDecisionPipeline",
    "load_saved_fold_thresholds",
]

