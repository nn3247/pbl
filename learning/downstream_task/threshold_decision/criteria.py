"""
閾値決定（Threshold Decision）の3大数学的最適化基準モジュール
(learning/downstream_task/threshold_decision/criteria.py)

実装する3大基準:
1. `youden_j` (Youden's J index / Balanced Accuracy 最大化):
   J(th) = Sensitivity(th) + Specificity(th) - 1 = TPR(th) - FPR(th) = 2 * BalancedAccuracy(th) - 1
   クラス不均衡（集中 82.8% : 休憩 17.2%）に対して理論的に不変であり、休憩の見逃しを防ぐ第一選択基準。

2. `f1_macro` (F1-Macro 最大化):
   F1_macro(th) = 0.5 * (F1_pos(th) + F1_neg(th))
   多数派（集中=1）と少数派（休憩=0）の F1 スコアを対等に評価し、休憩クラスの検出精度を直接引き上げる基準。

3. `pr_balance` (Precision-Recall バランス / PR均衡点):
   Precision(th) と Recall(th) のトレードオフ均衡点（|Precision - Recall| 最小化、または F1 との複合均衡）を求める基準。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve


SUPPORTED_THRESHOLD_CRITERIA: Tuple[str, ...] = (
    "youden_j",
    "f1_macro",
    "pr_balance",
)

CRITERION_DISPLAY_NAMES: Dict[str, str] = {
    "youden_j": "Youden's J Index (Balanced Accuracy 最大化)",
    "f1_macro": "F1-Macro 最大化 (集中・休憩 両クラス調和平均)",
    "pr_balance": "Precision-Recall バランス (PR均衡点)",
}

CRITERION_ALIASES: Dict[str, str] = {
    "youden": "youden_j",
    "youden_j": "youden_j",
    "youdens_j": "youden_j",
    "j_index": "youden_j",
    "balanced_acc": "youden_j",
    "balanced_accuracy": "youden_j",
    "f1": "f1_macro",
    "f1_macro": "f1_macro",
    "macro_f1": "f1_macro",
    "pr": "pr_balance",
    "pr_balance": "pr_balance",
    "precision_recall": "pr_balance",
    "pr_breakeven": "pr_balance",
}


def normalize_criterion_name(name: str) -> str:
    """閾値決定基準名を正規化します"""
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in CRITERION_ALIASES:
        raise ValueError(
            f"未対応の閾値決定基準です: '{name}'. "
            f"利用可能な基準: {list(SUPPORTED_THRESHOLD_CRITERIA)}"
        )
    return CRITERION_ALIASES[key]


@dataclass
class ThresholdSearchResult:
    """単一データセット（LOSO の1つの Fold 内検証データ等）での最適閾値探索結果"""
    criterion: str
    optimal_threshold: float
    criterion_score: float
    youden_j: float
    balanced_acc: float
    f1_macro: float
    f1_pos: float
    f1_neg: float
    sensitivity: float
    specificity: float
    precision_pos: float
    precision_neg: float
    accuracy: float
    pr_gap: float


def compute_threshold_curve_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    true_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    指定された閾値グリッド `thresholds` の各点について、
    TP, FP, TN, FN から導出される全指標（Youden's J, Balanced Acc, F1-Macro, PR差など）を
    ベクトル化演算で高速かつ厳密に一括計算します。
    """
    y_true_b = (np.asarray(y_true, dtype=float).ravel() >= true_threshold).astype(np.int32)
    y_score = np.asarray(y_pred, dtype=float).ravel()

    valid = np.isfinite(y_score) & np.isfinite(y_true_b)
    y_true_b = y_true_b[valid]
    y_score = y_score[valid]

    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 181)
    th_arr = np.asarray(thresholds, dtype=float).ravel()

    if len(y_true_b) == 0 or len(np.unique(y_true_b)) < 2:
        return pd.DataFrame({
            "threshold": th_arr,
            "youden_j": np.zeros_like(th_arr),
            "balanced_acc": np.full_like(th_arr, 0.5),
            "f1_macro": np.zeros_like(th_arr),
            "f1_pos": np.zeros_like(th_arr),
            "f1_neg": np.zeros_like(th_arr),
            "sensitivity": np.zeros_like(th_arr),
            "specificity": np.zeros_like(th_arr),
            "precision_pos": np.zeros_like(th_arr),
            "precision_neg": np.zeros_like(th_arr),
            "accuracy": np.zeros_like(th_arr),
            "pr_gap": np.ones_like(th_arr),
            "pr_balance_score": np.zeros_like(th_arr),
        })

    # (K, N) のブロードキャスト比較: pred_b[k, i] = 1 if y_score[i] >= th_arr[k] else 0
    pred_pos = y_score[None, :] >= th_arr[:, None]
    true_pos = (y_true_b == 1)[None, :]
    true_neg = ~true_pos
    pred_neg = ~pred_pos

    tp = np.sum(pred_pos & true_pos, axis=1).astype(float)
    fp = np.sum(pred_pos & true_neg, axis=1).astype(float)
    tn = np.sum(pred_neg & true_neg, axis=1).astype(float)
    fn = np.sum(pred_neg & true_pos, axis=1).astype(float)

    n_total = float(len(y_true_b))
    eps = 1e-12

    # 感度 (Recall / TPR) & 特異度 (TNR)
    sensitivity = tp / np.maximum(tp + fn, eps)
    specificity = tn / np.maximum(tn + fp, eps)

    # Youden's J = Sensitivity + Specificity - 1, Balanced Accuracy = (Sensitivity + Specificity) / 2
    youden_j = sensitivity + specificity - 1.0
    balanced_acc = 0.5 * (sensitivity + specificity)

    # 適合率 (Precision Pos / Neg)
    precision_pos = tp / np.maximum(tp + fp, eps)
    precision_neg = tn / np.maximum(tn + fn, eps)

    # F1-Pos (集中クラス=1) & F1-Neg (休憩クラス=0) & F1-Macro
    f1_pos = (2.0 * precision_pos * sensitivity) / np.maximum(precision_pos + sensitivity, eps)
    f1_neg = (2.0 * precision_neg * specificity) / np.maximum(precision_neg + specificity, eps)
    f1_macro = 0.5 * (f1_pos + f1_neg)

    # 正解率 (Accuracy)
    accuracy = (tp + tn) / np.maximum(n_total, eps)

    # PR-Balance:
    # 少数派である休憩クラス (Class 0) と多数派である集中クラス (Class 1) の双方で
    # Precision と Recall が極端に崩壊せず均衡する点を評価する。
    # 特に不均衡データでは Macro-Precision と Macro-Recall の均衡点 (|MacroP - MacroR| が最小かつ F1_macro が高い点)
    macro_precision = 0.5 * (precision_pos + precision_neg)
    macro_recall = 0.5 * (sensitivity + specificity)
    pr_gap = np.abs(macro_precision - macro_recall)

    # 両クラスが少なくとも1サンプル以上予測される有効領域で F1_macro - |MacroP - MacroR| を最大化
    pr_balance_score = f1_macro - pr_gap

    return pd.DataFrame({
        "threshold": th_arr,
        "youden_j": youden_j,
        "balanced_acc": balanced_acc,
        "f1_macro": f1_macro,
        "f1_pos": f1_pos,
        "f1_neg": f1_neg,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision_pos": precision_pos,
        "precision_neg": precision_neg,
        "accuracy": accuracy,
        "pr_gap": pr_gap,
        "pr_balance_score": pr_balance_score,
    })


def _build_candidate_thresholds(
    y_true_b: np.ndarray,
    y_score: np.ndarray,
    search_range: Tuple[float, float] = (0.05, 0.95),
    search_step: float = 0.005,
    use_exact_curve_points: bool = True,
) -> np.ndarray:
    """
    グリッド閾値に加えて、`roc_curve` および `precision_recall_curve` の厳密な変化点（ユニーク予測値）を
    候補閾値として統合します。
    """
    low, high = float(search_range[0]), float(search_range[1])
    n_grid = max(int(round((high - low) / max(search_step, 1e-4))) + 1, 19)
    grid_ths = np.linspace(low, high, n_grid)

    if not use_exact_curve_points or len(np.unique(y_true_b)) < 2:
        return grid_ths

    try:
        _, _, roc_ths = roc_curve(y_true_b, y_score)
        _, _, pr_ths = precision_recall_curve(y_true_b, y_score)
        exact_ths = np.concatenate([roc_ths[np.isfinite(roc_ths)], pr_ths[np.isfinite(pr_ths)]])
        # 極端な外れ値閾値（全サンプル0または全サンプル1になる端点）を除外するためスコア範囲内 [min_s, max_s] かつ [0.01, 0.99] に制限
        s_min = max(float(np.percentile(y_score, 1.0)), 0.01)
        s_max = min(float(np.percentile(y_score, 99.0)), 0.99)
        low_eff = min(low, s_min)
        high_eff = max(high, s_max)
        exact_ths = exact_ths[(exact_ths >= low_eff) & (exact_ths <= high_eff)]
        merged = np.unique(np.concatenate([grid_ths, exact_ths]))
        return merged if len(merged) > 0 else grid_ths
    except Exception:
        return grid_ths


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    criterion: str = "youden_j",
    search_range: Tuple[float, float] = (0.05, 0.95),
    search_step: float = 0.005,
    use_exact_curve_points: bool = True,
    fallback_threshold: float = 0.5,
    true_threshold: float = 0.5,
) -> ThresholdSearchResult:
    """
    検証データ (`y_true`, `y_pred`) のみを用いて、指定された基準 (`youden_j`, `f1_macro`, `pr_balance`) を
    最大化する最適カットオフ閾値 `th*` を決定します。

    ※ テスト被験者データは一切入力せず、必ず各 LOSO Fold 内の学習・検証データのみを渡してください。
    """
    canon_crit = normalize_criterion_name(criterion)
    y_true_b = (np.asarray(y_true, dtype=float).ravel() >= true_threshold).astype(np.int32)
    y_score = np.asarray(y_pred, dtype=float).ravel()

    valid = np.isfinite(y_score) & np.isfinite(y_true_b)
    y_true_b = y_true_b[valid]
    y_score = y_score[valid]

    if len(y_true_b) == 0 or len(np.unique(y_true_b)) < 2:
        return ThresholdSearchResult(
            criterion=canon_crit,
            optimal_threshold=float(fallback_threshold),
            criterion_score=0.0,
            youden_j=0.0,
            balanced_acc=0.5,
            f1_macro=0.0,
            f1_pos=0.0,
            f1_neg=0.0,
            sensitivity=0.0,
            specificity=0.0,
            precision_pos=0.0,
            precision_neg=0.0,
            accuracy=0.0,
            pr_gap=1.0,
        )

    candidates = _build_candidate_thresholds(
        y_true_b=y_true_b,
        y_score=y_score,
        search_range=search_range,
        search_step=search_step,
        use_exact_curve_points=use_exact_curve_points,
    )
    curve_df = compute_threshold_curve_metrics(
        y_true=y_true_b,
        y_pred=y_score,
        thresholds=candidates,
        true_threshold=0.5,
    )

    if canon_crit == "youden_j":
        score_col = "youden_j"
    elif canon_crit == "f1_macro":
        score_col = "f1_macro"
    elif canon_crit == "pr_balance":
        score_col = "pr_balance_score"
    else:
        score_col = "youden_j"

    scores = curve_df[score_col].values
    if not np.isfinite(scores).any():
        best_idx = int(np.argmin(np.abs(curve_df["threshold"].values - fallback_threshold)))
    else:
        max_val = np.nanmax(scores)
        # 同率最大点が複数ある場合は、その中央値（最もロバストなマージン中央の閾値）を選択
        tie_indices = np.where(np.isclose(scores, max_val, atol=1e-7))[0]
        best_idx = int(tie_indices[len(tie_indices) // 2])

    row = curve_df.iloc[best_idx]
    return ThresholdSearchResult(
        criterion=canon_crit,
        optimal_threshold=float(row["threshold"]),
        criterion_score=float(row[score_col]),
        youden_j=float(row["youden_j"]),
        balanced_acc=float(row["balanced_acc"]),
        f1_macro=float(row["f1_macro"]),
        f1_pos=float(row["f1_pos"]),
        f1_neg=float(row["f1_neg"]),
        sensitivity=float(row["sensitivity"]),
        specificity=float(row["specificity"]),
        precision_pos=float(row["precision_pos"]),
        precision_neg=float(row["precision_neg"]),
        accuracy=float(row["accuracy"]),
        pr_gap=float(row["pr_gap"]),
    )


def find_all_criteria_thresholds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    criteria: Optional[List[str]] = None,
    search_range: Tuple[float, float] = (0.05, 0.95),
    search_step: float = 0.005,
    use_exact_curve_points: bool = True,
    fallback_threshold: float = 0.5,
) -> Dict[str, ThresholdSearchResult]:
    """3大基準すべて（または指定リスト）に対して一括で最適閾値を探索します"""
    crit_list = [normalize_criterion_name(c) for c in (criteria or list(SUPPORTED_THRESHOLD_CRITERIA))]
    results: Dict[str, ThresholdSearchResult] = {}
    for c in crit_list:
        results[c] = find_optimal_threshold(
            y_true=y_true,
            y_pred=y_pred,
            criterion=c,
            search_range=search_range,
            search_step=search_step,
            use_exact_curve_points=use_exact_curve_points,
            fallback_threshold=fallback_threshold,
        )
    return results


__all__ = [
    "SUPPORTED_THRESHOLD_CRITERIA",
    "CRITERION_DISPLAY_NAMES",
    "CRITERION_ALIASES",
    "normalize_criterion_name",
    "ThresholdSearchResult",
    "compute_threshold_curve_metrics",
    "find_optimal_threshold",
    "find_all_criteria_thresholds",
]

