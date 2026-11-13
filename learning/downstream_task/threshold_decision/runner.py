"""
LOSO-CV 完全データリークフリー 閾値決定（Threshold Tuning）実行エンジン
(learning/downstream_task/threshold_decision/runner.py)

テスト被験者（Test）のデータを一切参照せず、各 LOSO Fold 内の
学習・検証データ（Train/Val）のみから 3大基準（Youden's J, F1-Macro, PR-Balance）に基づき
最適カットオフ閾値 th*_k を決定し、テスト被験者の二値化判定へ適用します。
"""

from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut

from .criteria import (
    CRITERION_DISPLAY_NAMES,
    SUPPORTED_THRESHOLD_CRITERIA,
    compute_threshold_curve_metrics,
    find_all_criteria_thresholds,
    find_optimal_threshold,
    normalize_criterion_name,
)


def _compute_binary_classification_detail(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float | np.ndarray,
    true_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    単一閾値またはサンプル別閾値配列を用いて、二値分類の詳細指標
    (Balanced Acc, Youden's J, F1-Macro, F1-Pos, F1-Neg, Sensitivity, Specificity, CM要素等) を計算します。
    """
    y_true_arr = np.asarray(y_true, dtype=float).ravel()
    y_pred_arr = np.asarray(y_pred, dtype=float).ravel()

    if np.ndim(threshold) == 0:
        th_arr = np.full_like(y_pred_arr, float(threshold))
    else:
        th_arr = np.asarray(threshold, dtype=float).ravel()

    valid = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr) & np.isfinite(th_arr)
    y_true_arr = y_true_arr[valid]
    y_pred_arr = y_pred_arr[valid]
    th_arr = th_arr[valid]

    if len(y_true_arr) == 0:
        return {
            "accuracy": np.nan,
            "balanced_acc": np.nan,
            "youden_j": np.nan,
            "f1_macro": np.nan,
            "f1_pos": np.nan,
            "f1_neg": np.nan,
            "sensitivity": np.nan,
            "specificity": np.nan,
            "precision_pos": np.nan,
            "precision_neg": np.nan,
            "roc_auc": np.nan,
            "pr_auc": np.nan,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
        }

    y_true_b = (y_true_arr >= true_threshold).astype(int)
    y_pred_b = (y_pred_arr >= th_arr).astype(int)

    acc = float(accuracy_score(y_true_b, y_pred_b))
    f1_mac = float(f1_score(y_true_b, y_pred_b, average="macro", zero_division=0))
    f1_pos = float(f1_score(y_true_b, y_pred_b, pos_label=1, zero_division=0))
    f1_neg = float(f1_score(y_true_b, y_pred_b, pos_label=0, zero_division=0))
    prec_pos = float(precision_score(y_true_b, y_pred_b, pos_label=1, zero_division=0))
    prec_neg = float(precision_score(y_true_b, y_pred_b, pos_label=0, zero_division=0))
    sens = float(recall_score(y_true_b, y_pred_b, pos_label=1, zero_division=0))
    spec = float(recall_score(y_true_b, y_pred_b, pos_label=0, zero_division=0))

    if len(np.unique(y_true_b)) > 1:
        b_acc = float(balanced_accuracy_score(y_true_b, y_pred_b))
        youden = float(sens + spec - 1.0)
        auc = float(roc_auc_score(y_true_b, y_pred_arr))
        pr_auc = float(average_precision_score(y_true_b, y_pred_arr))
    else:
        b_acc = np.nan
        youden = np.nan
        auc = np.nan
        pr_auc = np.nan

    cm = confusion_matrix(y_true_b, y_pred_b, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    return {
        "accuracy": acc,
        "balanced_acc": b_acc,
        "youden_j": youden,
        "f1_macro": f1_mac,
        "f1_pos": f1_pos,
        "f1_neg": f1_neg,
        "sensitivity": sens,
        "specificity": spec,
        "precision_pos": prec_pos,
        "precision_neg": prec_neg,
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def _select_inner_validation_subset(
    train_idx: np.ndarray,
    subjects: np.ndarray,
    random_state: int = 42,
    val_subject_ratio: float = 0.2,
    split_mode: str = "train_oof",
) -> np.ndarray:
    """
    LOSO Fold 内の非テストデータ (`train_idx`) から、閾値探索に用いる検証インデックスを選択します。

    - `split_mode == "train_oof"` (デフォルト・推奨):
      テスト被験者以外の全被験者 (`train_idx`) の予測スコアを使用します。
      全14名の非テスト被験者データを使用するためクラス比率の偏りが小さく、最も安定した閾値が得られます。
    - `split_mode == "group_val"`:
      非テスト被験者の中からさらに `GroupShuffleSplit` で検証用サブセット（約20%の被験者）を切り出して閾値を探索します。
    """
    mode = str(split_mode).strip().lower()
    if mode != "group_val":
        return train_idx

    train_subjects = subjects[train_idx]
    unique_tr_subs = np.unique(train_subjects)
    if len(unique_tr_subs) < 3:
        return train_idx

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=min(max(float(val_subject_ratio), 0.1), 0.5),
        random_state=random_state,
    )
    _, inner_val_pos = next(gss.split(train_idx, groups=train_subjects))
    return train_idx[inner_val_pos]


def run_prelearning_fold_threshold_tuning(
    loso_folds: Sequence,
    y_true_all: np.ndarray,
    subjects: np.ndarray,
    criterion: str = "youden_j",
    search_range: Tuple[float, float] = (0.05, 0.95),
    search_step: float = 0.005,
    use_exact_curve_points: bool = True,
    fallback_threshold: float = 0.5,
    validation_split_mode: str = "train_oof",
    val_subject_ratio: float = 0.2,
    random_state: int = 42,
    model_tag: str = "gmm_soft_label",
    log=None,
) -> Dict:
    """
    学習前（Base Model / Meta Model 学習前）に、各 LOSO Fold 内で完結して計算された
    GMM ソフトラベル (`f_data.y_soft_train`) と正解タスクラベル (`y_true_all[f_data.train_idx]`) から
    Youden's J Index（Balanced Accuracy 最大化）等の基準で最適閾値 th*_k を決定し、
    Fold 内のテストデータ (`f_data.y_soft_test`) で評価します。
    """
    canon_crit = normalize_criterion_name(criterion)
    y_true_arr = np.asarray(y_true_all, dtype=float).ravel()
    sub_arr = np.asarray(subjects, dtype=str).ravel()
    n_samples = len(y_true_arr)
    n_folds = len(loso_folds)

    oof_preds = np.full(n_samples, np.nan, dtype=float)
    oof_thresholds = np.full(n_samples, float(fallback_threshold), dtype=float)
    oof_pred_binary_05 = np.zeros(n_samples, dtype=int)
    oof_pred_binary_tuned = np.zeros(n_samples, dtype=int)

    fold_records: List[Dict] = []
    fold_threshold_map: Dict[int, float] = {}
    user_threshold_map: Dict[str, float] = {}

    mode = str(validation_split_mode).strip().lower()

    for f_data in loso_folds:
        fold = int(f_data.fold)
        test_user = str(f_data.test_user)
        train_idx = np.asarray(f_data.train_idx, dtype=int)
        test_idx = np.asarray(f_data.test_idx, dtype=int)

        if mode == "group_val":
            val_global_idx = _select_inner_validation_subset(
                train_idx=train_idx,
                subjects=sub_arr,
                random_state=random_state + fold,
                val_subject_ratio=val_subject_ratio,
                split_mode="group_val",
            )
            inner_mask = np.isin(train_idx, val_global_idx)
            y_val_true = y_true_arr[train_idx][inner_mask]
            y_val_pred = np.asarray(f_data.y_soft_train, dtype=float)[inner_mask]
            n_val = int(np.sum(inner_mask))
        else:
            y_val_true = y_true_arr[train_idx]
            y_val_pred = np.asarray(f_data.y_soft_train, dtype=float)
            n_val = len(train_idx)

        # 1. 非テスト被験者 (Train/Val) のデータのみで最適閾値 th*_k を探索 (完全データリークフリー)
        search_res = find_optimal_threshold(
            y_true=y_val_true,
            y_pred=y_val_pred,
            criterion=canon_crit,
            search_range=search_range,
            search_step=search_step,
            use_exact_curve_points=use_exact_curve_points,
            fallback_threshold=fallback_threshold,
        )
        th_k = float(search_res.optimal_threshold)
        f_data.optimal_threshold = th_k
        f_data.threshold_criterion = canon_crit

        fold_threshold_map[fold] = th_k
        user_threshold_map[test_user] = th_k

        # 2. テスト被験者へ固定閾値 0.50 と最適閾値 th*_k の双方を適用して比較評価
        y_te_true = np.asarray(f_data.y_true_test, dtype=float)
        y_te_pred = np.asarray(f_data.y_soft_test, dtype=float)

        oof_preds[test_idx] = y_te_pred
        oof_thresholds[test_idx] = th_k
        oof_pred_binary_05[test_idx] = (y_te_pred >= 0.5).astype(int)
        oof_pred_binary_tuned[test_idx] = (y_te_pred >= th_k).astype(int)

        det_05 = _compute_binary_classification_detail(y_te_true, y_te_pred, threshold=0.5)
        det_tuned = _compute_binary_classification_detail(y_te_true, y_te_pred, threshold=th_k)

        fold_records.append({
            "fold": fold,
            "test_user": test_user,
            "n_train": int(len(train_idx)),
            "n_val": int(n_val),
            "n_test": int(len(test_idx)),
            "criterion": canon_crit,
            "optimal_threshold": round(th_k, 4),
            "val_criterion_score": round(search_res.criterion_score, 4),
            "val_youden_j": round(search_res.youden_j, 4),
            "val_balanced_acc": round(search_res.balanced_acc, 4),
            "val_f1_macro": round(search_res.f1_macro, 4),
            "val_sensitivity": round(search_res.sensitivity, 4),
            "val_specificity": round(search_res.specificity, 4),
            # Test (Fixed 0.5)
            "acc_05": det_05["accuracy"],
            "balanced_acc_05": det_05["balanced_acc"],
            "youden_j_05": det_05["youden_j"],
            "f1_macro_05": det_05["f1_macro"],
            "f1_pos_05": det_05["f1_pos"],
            "f1_neg_05": det_05["f1_neg"],
            "sensitivity_05": det_05["sensitivity"],
            "specificity_05": det_05["specificity"],
            # Test (Tuned th*_k)
            "acc_tuned": det_tuned["accuracy"],
            "balanced_acc_tuned": det_tuned["balanced_acc"],
            "youden_j_tuned": det_tuned["youden_j"],
            "f1_macro_tuned": det_tuned["f1_macro"],
            "f1_pos_tuned": det_tuned["f1_pos"],
            "f1_neg_tuned": det_tuned["f1_neg"],
            "sensitivity_tuned": det_tuned["sensitivity"],
            "specificity_tuned": det_tuned["specificity"],
            "roc_auc": det_tuned["roc_auc"],
            "pr_auc": det_tuned["pr_auc"],
            # 改善幅 (Tuned - Fixed 0.5)
            "delta_balanced_acc": (
                det_tuned["balanced_acc"] - det_05["balanced_acc"]
                if pd.notna(det_tuned["balanced_acc"]) and pd.notna(det_05["balanced_acc"])
                else np.nan
            ),
            "delta_f1_macro": det_tuned["f1_macro"] - det_05["f1_macro"],
            "delta_f1_neg": det_tuned["f1_neg"] - det_05["f1_neg"],
        })

    fold_df = pd.DataFrame(fold_records)

    baseline_oof = _compute_binary_classification_detail(y_true_arr, oof_preds, threshold=0.5)
    tuned_oof = _compute_binary_classification_detail(y_true_arr, oof_preds, threshold=oof_thresholds)

    def _mean_std(col: str) -> Tuple[float, float]:
        vals = fold_df[col].dropna().values
        if len(vals) == 0:
            return np.nan, np.nan
        return float(np.mean(vals)), float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)

    th_mean, th_std = _mean_std("optimal_threshold")
    th_min = float(fold_df["optimal_threshold"].min()) if not fold_df.empty else float(fallback_threshold)
    th_max = float(fold_df["optimal_threshold"].max()) if not fold_df.empty else float(fallback_threshold)

    baseline_fold_stats = {
        "accuracy": _mean_std("acc_05"),
        "balanced_acc": _mean_std("balanced_acc_05"),
        "youden_j": _mean_std("youden_j_05"),
        "f1_macro": _mean_std("f1_macro_05"),
        "f1_pos": _mean_std("f1_pos_05"),
        "f1_neg": _mean_std("f1_neg_05"),
        "sensitivity": _mean_std("sensitivity_05"),
        "specificity": _mean_std("specificity_05"),
    }
    tuned_fold_stats = {
        "accuracy": _mean_std("acc_tuned"),
        "balanced_acc": _mean_std("balanced_acc_tuned"),
        "youden_j": _mean_std("youden_j_tuned"),
        "f1_macro": _mean_std("f1_macro_tuned"),
        "f1_pos": _mean_std("f1_pos_tuned"),
        "f1_neg": _mean_std("f1_neg_tuned"),
        "sensitivity": _mean_std("sensitivity_tuned"),
        "specificity": _mean_std("specificity_tuned"),
        "roc_auc": _mean_std("roc_auc"),
        "pr_auc": _mean_std("pr_auc"),
    }

    curve_df = compute_threshold_curve_metrics(
        y_true=y_true_arr,
        y_pred=oof_preds,
        thresholds=np.linspace(0.02, 0.98, 193),
    )
    global_all_crit = find_all_criteria_thresholds(
        y_true=y_true_arr,
        y_pred=oof_preds,
        search_range=search_range,
        search_step=search_step,
        use_exact_curve_points=use_exact_curve_points,
        fallback_threshold=fallback_threshold,
    )

    if log is not None:
        log.info(
            f"🎯 [Pre-Learning Threshold Tuning: {model_tag} | {canon_crit} ({n_folds} Folds)] "
            f"Mean th*={th_mean:.4f}±{th_std:.4f} (range=[{th_min:.3f}, {th_max:.3f}]) | "
            f"OOF BalancedAcc: {baseline_oof['balanced_acc']:.4f} (0.5) ➔ {tuned_oof['balanced_acc']:.4f} (th*) "
            f"[{tuned_oof['balanced_acc'] - baseline_oof['balanced_acc']:+.4f}] | "
            f"F1-Macro: {baseline_oof['f1_macro']:.4f} ➔ {tuned_oof['f1_macro']:.4f} "
            f"[{tuned_oof['f1_macro'] - baseline_oof['f1_macro']:+.4f}]"
        )

    return {
        "model_tag": model_tag,
        "criterion": canon_crit,
        "criterion_label": CRITERION_DISPLAY_NAMES.get(canon_crit, canon_crit),
        "mean_threshold": th_mean,
        "std_threshold": th_std,
        "min_threshold": th_min,
        "max_threshold": th_max,
        "fold_threshold_map": fold_threshold_map,
        "user_threshold_map": user_threshold_map,
        "oof_preds": oof_preds,
        "oof_thresholds": oof_thresholds,
        "oof_pred_binary_05": oof_pred_binary_05,
        "oof_pred_binary_tuned": oof_pred_binary_tuned,
        "baseline_oof_detail": baseline_oof,
        "tuned_oof_detail": tuned_oof,
        "baseline_fold_stats": baseline_fold_stats,
        "tuned_fold_stats": tuned_fold_stats,
        "fold_df": fold_df,
        "curve_df": curve_df,
        "global_all_criteria_thresholds": {
            k: v.optimal_threshold for k, v in global_all_crit.items()
        },
        "improvement": {
            "balanced_acc": tuned_oof["balanced_acc"] - baseline_oof["balanced_acc"],
            "youden_j": tuned_oof["youden_j"] - baseline_oof["youden_j"],
            "f1_macro": tuned_oof["f1_macro"] - baseline_oof["f1_macro"],
            "f1_neg": tuned_oof["f1_neg"] - baseline_oof["f1_neg"],
            "accuracy": tuned_oof["accuracy"] - baseline_oof["accuracy"],
            "specificity": tuned_oof["specificity"] - baseline_oof["specificity"],
            "sensitivity": tuned_oof["sensitivity"] - baseline_oof["sensitivity"],
        },
    }


def run_loso_threshold_tuning(
    y_true_all: np.ndarray,
    y_pred_all: np.ndarray,
    subjects: np.ndarray,
    criterion: str = "youden_j",
    search_range: Tuple[float, float] = (0.05, 0.95),
    search_step: float = 0.005,
    use_exact_curve_points: bool = True,
    fallback_threshold: float = 0.5,
    validation_split_mode: str = "train_oof",
    val_subject_ratio: float = 0.2,
    random_state: int = 42,
    model_tag: str = "model",
    log=None,
) -> Dict:
    """
    LOSO-CV の各 Fold について、テスト被験者以外のデータ（Train/Val）のみから
    最適カットオフ閾値 `th*_k` を決定し、テスト被験者の二値化判定へ適用します。

    同時に比較ベースラインとして固定閾値 `th = 0.50` の結果も集計し、
    閾値最適化による改善幅（ΔBalanced Accuracy, ΔF1-Macro, ΔSpecificity 等）を算出します。
    """
    canon_crit = normalize_criterion_name(criterion)
    y_true_arr = np.asarray(y_true_all, dtype=float).ravel()
    y_pred_arr = np.asarray(y_pred_all, dtype=float).ravel()
    sub_arr = np.asarray(subjects, dtype=str).ravel()

    n_samples = len(y_true_arr)
    oof_thresholds = np.full(n_samples, float(fallback_threshold), dtype=float)
    oof_pred_binary_05 = (y_pred_arr >= 0.5).astype(int)
    oof_pred_binary_tuned = np.zeros(n_samples, dtype=int)

    logo = LeaveOneGroupOut()
    fold_records: List[Dict] = []
    fold_threshold_map: Dict[int, float] = {}
    user_threshold_map: Dict[str, float] = {}

    for fold, (train_idx, test_idx) in enumerate(logo.split(y_pred_arr, y_true_arr, groups=sub_arr), 1):
        test_user = str(sub_arr[test_idx[0]])

        # 1. テスト被験者以外のデータから検証サブセットを選択（完全データリークフリー）
        val_idx = _select_inner_validation_subset(
            train_idx=train_idx,
            subjects=sub_arr,
            random_state=random_state + fold,
            val_subject_ratio=val_subject_ratio,
            split_mode=validation_split_mode,
        )

        # 2. 検証データのみで最適閾値 th*_k を探索
        search_res = find_optimal_threshold(
            y_true=y_true_arr[val_idx],
            y_pred=y_pred_arr[val_idx],
            criterion=canon_crit,
            search_range=search_range,
            search_step=search_step,
            use_exact_curve_points=use_exact_curve_points,
            fallback_threshold=fallback_threshold,
        )
        th_k = float(search_res.optimal_threshold)
        oof_thresholds[test_idx] = th_k
        oof_pred_binary_tuned[test_idx] = (y_pred_arr[test_idx] >= th_k).astype(int)
        fold_threshold_map[fold] = th_k
        user_threshold_map[test_user] = th_k

        # 3. テスト被験者への適用評価: 固定 0.50 vs 最適閾値 th*_k
        y_te_true = y_true_arr[test_idx]
        y_te_pred = y_pred_arr[test_idx]

        det_05 = _compute_binary_classification_detail(y_te_true, y_te_pred, threshold=0.5)
        det_tuned = _compute_binary_classification_detail(y_te_true, y_te_pred, threshold=th_k)

        fold_records.append({
            "fold": fold,
            "test_user": test_user,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "criterion": canon_crit,
            "optimal_threshold": round(th_k, 4),
            "val_criterion_score": round(search_res.criterion_score, 4),
            "val_youden_j": round(search_res.youden_j, 4),
            "val_balanced_acc": round(search_res.balanced_acc, 4),
            "val_f1_macro": round(search_res.f1_macro, 4),
            "val_sensitivity": round(search_res.sensitivity, 4),
            "val_specificity": round(search_res.specificity, 4),
            # Test (Fixed 0.5)
            "acc_05": det_05["accuracy"],
            "balanced_acc_05": det_05["balanced_acc"],
            "youden_j_05": det_05["youden_j"],
            "f1_macro_05": det_05["f1_macro"],
            "f1_pos_05": det_05["f1_pos"],
            "f1_neg_05": det_05["f1_neg"],
            "sensitivity_05": det_05["sensitivity"],
            "specificity_05": det_05["specificity"],
            # Test (Tuned th*_k)
            "acc_tuned": det_tuned["accuracy"],
            "balanced_acc_tuned": det_tuned["balanced_acc"],
            "youden_j_tuned": det_tuned["youden_j"],
            "f1_macro_tuned": det_tuned["f1_macro"],
            "f1_pos_tuned": det_tuned["f1_pos"],
            "f1_neg_tuned": det_tuned["f1_neg"],
            "sensitivity_tuned": det_tuned["sensitivity"],
            "specificity_tuned": det_tuned["specificity"],
            "roc_auc": det_tuned["roc_auc"],
            "pr_auc": det_tuned["pr_auc"],
            # 改善幅 (Tuned - Fixed 0.5)
            "delta_balanced_acc": (
                det_tuned["balanced_acc"] - det_05["balanced_acc"]
                if pd.notna(det_tuned["balanced_acc"]) and pd.notna(det_05["balanced_acc"])
                else np.nan
            ),
            "delta_f1_macro": det_tuned["f1_macro"] - det_05["f1_macro"],
            "delta_f1_neg": det_tuned["f1_neg"] - det_05["f1_neg"],
        })

    fold_df = pd.DataFrame(fold_records)

    # 全体 OOF における Fixed 0.50 vs Tuned (サンプル別 Fold 最適閾値) の比較
    baseline_oof = _compute_binary_classification_detail(y_true_arr, y_pred_arr, threshold=0.5)
    tuned_oof = _compute_binary_classification_detail(y_true_arr, y_pred_arr, threshold=oof_thresholds)

    # Fold 間統計量 (Mean ± Std)
    def _mean_std(col: str) -> Tuple[float, float]:
        vals = fold_df[col].dropna().values
        if len(vals) == 0:
            return np.nan, np.nan
        return float(np.mean(vals)), float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)

    th_mean, th_std = _mean_std("optimal_threshold")
    th_min = float(fold_df["optimal_threshold"].min()) if not fold_df.empty else float(fallback_threshold)
    th_max = float(fold_df["optimal_threshold"].max()) if not fold_df.empty else float(fallback_threshold)

    baseline_fold_stats = {
        "accuracy": _mean_std("acc_05"),
        "balanced_acc": _mean_std("balanced_acc_05"),
        "youden_j": _mean_std("youden_j_05"),
        "f1_macro": _mean_std("f1_macro_05"),
        "f1_pos": _mean_std("f1_pos_05"),
        "f1_neg": _mean_std("f1_neg_05"),
        "sensitivity": _mean_std("sensitivity_05"),
        "specificity": _mean_std("specificity_05"),
    }
    tuned_fold_stats = {
        "accuracy": _mean_std("acc_tuned"),
        "balanced_acc": _mean_std("balanced_acc_tuned"),
        "youden_j": _mean_std("youden_j_tuned"),
        "f1_macro": _mean_std("f1_macro_tuned"),
        "f1_pos": _mean_std("f1_pos_tuned"),
        "f1_neg": _mean_std("f1_neg_tuned"),
        "sensitivity": _mean_std("sensitivity_tuned"),
        "specificity": _mean_std("specificity_tuned"),
        "roc_auc": _mean_std("roc_auc"),
        "pr_auc": _mean_std("pr_auc"),
    }

    # 可視化用の全体スコア曲線 (参考用: 閾値と各指標の関係曲線)
    curve_df = compute_threshold_curve_metrics(
        y_true=y_true_arr,
        y_pred=y_pred_arr,
        thresholds=np.linspace(0.02, 0.98, 193),
    )
    global_all_crit = find_all_criteria_thresholds(
        y_true=y_true_arr,
        y_pred=y_pred_arr,
        search_range=search_range,
        search_step=search_step,
        use_exact_curve_points=use_exact_curve_points,
        fallback_threshold=fallback_threshold,
    )

    if log is not None:
        log.info(
            f"🎯 [Threshold Tuning: {model_tag} | {canon_crit}] "
            f"Mean th*={th_mean:.4f}±{th_std:.4f} (range=[{th_min:.3f}, {th_max:.3f}]) | "
            f"OOF BalancedAcc: {baseline_oof['balanced_acc']:.4f} (0.5) ➔ {tuned_oof['balanced_acc']:.4f} (th*) "
            f"[{tuned_oof['balanced_acc'] - baseline_oof['balanced_acc']:+.4f}] | "
            f"F1-Macro: {baseline_oof['f1_macro']:.4f} ➔ {tuned_oof['f1_macro']:.4f} "
            f"[{tuned_oof['f1_macro'] - baseline_oof['f1_macro']:+.4f}]"
        )

    return {
        "model_tag": model_tag,
        "criterion": canon_crit,
        "criterion_label": CRITERION_DISPLAY_NAMES.get(canon_crit, canon_crit),
        "mean_threshold": th_mean,
        "std_threshold": th_std,
        "min_threshold": th_min,
        "max_threshold": th_max,
        "fold_threshold_map": fold_threshold_map,
        "user_threshold_map": user_threshold_map,
        "oof_preds": y_pred_arr,
        "oof_thresholds": oof_thresholds,
        "oof_pred_binary_05": oof_pred_binary_05,
        "oof_pred_binary_tuned": oof_pred_binary_tuned,
        "baseline_oof_detail": baseline_oof,
        "tuned_oof_detail": tuned_oof,
        "baseline_fold_stats": baseline_fold_stats,
        "tuned_fold_stats": tuned_fold_stats,
        "fold_df": fold_df,
        "curve_df": curve_df,
        "global_all_criteria_thresholds": {
            k: v.optimal_threshold for k, v in global_all_crit.items()
        },
        "improvement": {
            "balanced_acc": tuned_oof["balanced_acc"] - baseline_oof["balanced_acc"],
            "youden_j": tuned_oof["youden_j"] - baseline_oof["youden_j"],
            "f1_macro": tuned_oof["f1_macro"] - baseline_oof["f1_macro"],
            "f1_neg": tuned_oof["f1_neg"] - baseline_oof["f1_neg"],
            "accuracy": tuned_oof["accuracy"] - baseline_oof["accuracy"],
            "specificity": tuned_oof["specificity"] - baseline_oof["specificity"],
            "sensitivity": tuned_oof["sensitivity"] - baseline_oof["sensitivity"],
        },
    }


__all__ = [
    "SUPPORTED_THRESHOLD_CRITERIA",
    "run_prelearning_fold_threshold_tuning",
    "run_loso_threshold_tuning",
]

