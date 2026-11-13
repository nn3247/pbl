"""
メタモデル (Visual + Vital 予測値統合) LOSO-CV 学習・評価・最終推論実行エンジン
(learning/downstream_task/meta_model/runner.py)
"""

from __future__ import annotations
from pathlib import Path
import time
from typing import Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

try:
    from ..metrics import compute_fold_metrics, compute_oof_and_fold_summary
    from ..threshold_decision.criteria import find_optimal_threshold, normalize_criterion_name
    from .models import create_meta_model, normalize_meta_model_name
    from .reporter import generate_visualizations, save_markdown_report
except (ImportError, ValueError):
    from downstream_task.metrics import compute_fold_metrics, compute_oof_and_fold_summary
    from downstream_task.threshold_decision.criteria import find_optimal_threshold, normalize_criterion_name
    from downstream_task.meta_model.models import create_meta_model, normalize_meta_model_name
    from downstream_task.meta_model.reporter import generate_visualizations, save_markdown_report


def run_single_meta_model_loso(
    meta_model_name: str,
    selected_visual_model: str,
    selected_vital_model: str,
    y_pred_visual_oof: np.ndarray,
    y_pred_vital_oof: np.ndarray,
    y_gmm_soft_oof: np.ndarray,
    sample_weight_oof: np.ndarray,
    meta_df: pd.DataFrame,
    window_size: str,
    meta_output_dir: Path,
    cfg: Optional[dict],
    log,
    oof_thresholds: Optional[np.ndarray] = None,
) -> Dict:
    """
    指定された Visual ベースモデルと Vital ベースモデルの OOF 予測値を入力特徴量とし、
    単一のメタモデル (logistic_regression, ridge, lasso, elastic_net) を
    LOSO-CV で学習・評価・保存します。
    """
    canonical_name = normalize_meta_model_name(meta_model_name)

    dt_cfg = cfg.get("downstream_task", {}) if cfg else {}
    loso_cfg = cfg.get("loso_cv", {}) if cfg else {}
    th_cfg = dt_cfg.get("threshold_decision", {})

    seed = int(loso_cfg.get("random_state", dt_cfg.get("random_state", 42)))
    save_plots = bool(dt_cfg.get("save_plots", True))
    save_models = bool(dt_cfg.get("save_models", True))

    th_enabled = bool(th_cfg.get("enabled", True))
    th_criterion = normalize_criterion_name(th_cfg.get("default_criterion", "youden_j"))
    th_sr = th_cfg.get("search_range", [0.05, 0.95])
    th_search_range = (float(th_sr[0]), float(th_sr[1]))
    th_search_step = float(th_cfg.get("search_step", 0.005))
    th_use_exact = bool(th_cfg.get("use_exact_curve_points", True))
    th_fallback = float(th_cfg.get("fallback_threshold", 0.5))

    subject_col = "username" if "username" in meta_df.columns else "subject_id"
    label_col = "label" if "label" in meta_df.columns else "label_mean"

    subjects = meta_df[subject_col].astype(str).values
    labels = meta_df[label_col].values.astype(float)
    unique_users = np.unique(subjects)
    n_users = len(unique_users)
    n_samples = len(meta_df)

    feature_names = [
        f"visual_{selected_visual_model}",
        f"vital_{selected_vital_model}",
    ]
    X_meta_all = np.column_stack([
        np.asarray(y_pred_visual_oof, dtype=np.float64),
        np.asarray(y_pred_vital_oof, dtype=np.float64),
    ])
    y_soft_all = np.asarray(y_gmm_soft_oof, dtype=np.float64)
    w_all = np.asarray(sample_weight_oof, dtype=np.float64)

    oof_meta_preds = np.full(n_samples, np.nan, dtype=np.float32)
    fold_thresholds_arr = np.full(n_samples, float(th_fallback), dtype=np.float32)
    fold_records: List[Dict] = []
    fold_models = []
    vis_weights: List[float] = []
    vit_weights: List[float] = []
    intercepts: List[float] = []

    meta_output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"meta_{canonical_name}"

    log.info("-" * 75)
    log.info(
        f"▶ [Meta Model: {canonical_name.upper()}] LOSO-CV 開始 "
        f"(Visual='{selected_visual_model}', Vital='{selected_vital_model}', "
        f"Folds={n_users}, Threshold={th_criterion})"
    )
    start_time = time.time()

    logo = LeaveOneGroupOut()
    for fold, (train_idx, test_idx) in enumerate(logo.split(X_meta_all, labels, groups=subjects), 1):
        test_user = str(subjects[test_idx[0]])

        X_tr = X_meta_all[train_idx]
        y_tr = y_soft_all[train_idx]
        w_tr = w_all[train_idx]
        X_te = X_meta_all[test_idx]

        # 学習前決定の最適閾値 th*_k の取得 (未指定時は Train の GMM ソフトラベルから Youden's J で決定)
        if oof_thresholds is not None and np.isfinite(oof_thresholds[test_idx[0]]):
            th_k = float(oof_thresholds[test_idx[0]])
        elif th_enabled:
            th_res = find_optimal_threshold(
                y_true=labels[train_idx],
                y_pred=y_tr,
                criterion=th_criterion,
                search_range=th_search_range,
                search_step=th_search_step,
                use_exact_curve_points=th_use_exact,
                fallback_threshold=th_fallback,
            )
            th_k = float(th_res.optimal_threshold)
        else:
            th_k = float(th_fallback)

        fold_thresholds_arr[test_idx] = th_k

        meta_reg = create_meta_model(
            model_name=canonical_name,
            cfg=cfg,
            random_state=seed,
        )
        meta_reg.fit(
            X=X_tr,
            y=y_tr,
            sample_weight=w_tr,
            feature_names=feature_names,
        )

        y_pred_te = meta_reg.predict(X_te).astype(np.float32)
        oof_meta_preds[test_idx] = y_pred_te
        fold_models.append(meta_reg)

        coefs = meta_reg.get_coefficients()
        w_v = float(coefs.get(feature_names[0], 0.0))
        w_b = float(coefs.get(feature_names[1], 0.0))
        b_0 = float(coefs.get("intercept", 0.0))
        vis_weights.append(w_v)
        vit_weights.append(w_b)
        intercepts.append(b_0)

        f_metrics = compute_fold_metrics(
            fold=fold,
            test_user=test_user,
            y_true=labels[test_idx],
            y_pred=y_pred_te,
            threshold=th_k,
        )
        f_metrics["visual_weight"] = round(w_v, 4)
        f_metrics["vital_weight"] = round(w_b, 4)
        f_metrics["intercept"] = round(b_0, 4)
        fold_records.append(f_metrics)

        auc_str = f"{f_metrics['roc_auc']:.3f}" if pd.notna(f_metrics["roc_auc"]) else "N/A"
        pr_str = f"{f_metrics['pr_auc']:.3f}" if pd.notna(f_metrics["pr_auc"]) else "N/A"
        r_str = f"{f_metrics['pearson_r']:.3f}" if pd.notna(f_metrics["pearson_r"]) else "N/A"
        bacc_str = f"{f_metrics['balanced_acc']:.3f}" if pd.notna(f_metrics["balanced_acc"]) else "N/A"
        log.info(
            f"  Fold [{fold:2d}/{n_users:2d}] User: {test_user:<13} | "
            f"th*: {th_k:.3f} | MAE: {f_metrics['mae']:.4f} | r: {r_str} | "
            f"B-Acc: {bacc_str} | AUC: {auc_str} | PR-AUC: {pr_str} | "
            f"w_vis={w_v:.3f}, w_vital={w_b:.3f}"
        )

    elapsed = time.time() - start_time

    eval_res = compute_oof_and_fold_summary(
        y_true_all=labels,
        y_pred_all=oof_meta_preds,
        fold_records=fold_records,
        threshold=fold_thresholds_arr,
    )
    fold_df = eval_res["fold_df"]
    stats_dict = eval_res["stats_dict"]

    mean_vis_w = float(np.mean(vis_weights)) if vis_weights else 0.0
    mean_vit_w = float(np.mean(vit_weights)) if vit_weights else 0.0
    mean_intercept = float(np.mean(intercepts)) if intercepts else 0.0

    log.info(
        f"✅ [Meta Model: {canonical_name.upper()}] 完了 ({elapsed:.2f}s) | "
        f"Mean th*={eval_res['optimal_threshold_mean']:.4f} | "
        f"BalancedAcc={eval_res['balanced_acc']:.4f} | F1-Macro={eval_res['f1_macro']:.4f} | "
        f"ROC-AUC={eval_res['roc_auc']:.4f} | PR-AUC={eval_res['pr_auc']:.4f} | "
        f"Pearson r={eval_res['pearson_r']:.4f} | MAE={eval_res['mae']:.4f} | "
        f"MeanWeights=(vis:{mean_vis_w:.3f}, vital:{mean_vit_w:.3f})"
    )

    # 1. Fold別詳細 CSV 保存
    fold_csv = meta_output_dir / f"loso_{tag}_folds.csv"
    fold_df.to_csv(fold_csv, index=False)

    # 2. OOF 最終予測値 CSV 保存 (学習前決定の最適閾値 th*_k による二値判定を含む)
    oof_df = meta_df.copy()
    oof_df["y_gmm_soft_oof"] = y_gmm_soft_oof
    oof_df[f"y_pred_visual_{selected_visual_model}"] = y_pred_visual_oof
    oof_df[f"y_pred_vital_{selected_vital_model}"] = y_pred_vital_oof
    oof_df["optimal_threshold"] = fold_thresholds_arr
    oof_df["threshold_criterion"] = th_criterion
    oof_df["y_pred_oof"] = oof_meta_preds
    oof_df["y_pred_binary"] = (oof_meta_preds >= fold_thresholds_arr).astype(int)
    oof_csv = meta_output_dir / f"loso_{tag}_oof_predictions.csv"
    oof_df.to_csv(oof_csv, index=False)

    # 3. メタモデルの保存 (全データ学習済み最終メタモデル + Fold別モデル群 + 最適閾値情報)
    model_path = meta_output_dir / f"{canonical_name}_model.joblib"
    if save_models:
        final_meta_reg = create_meta_model(
            model_name=canonical_name,
            cfg=cfg,
            random_state=seed,
        )
        final_meta_reg.fit(
            X=X_meta_all,
            y=y_soft_all,
            sample_weight=w_all,
            feature_names=feature_names,
        )
        extra_meta = {
            "meta_model_name": canonical_name,
            "window_size": window_size,
            "selected_visual_model": selected_visual_model,
            "selected_vital_model": selected_vital_model,
            "threshold_criterion": th_criterion,
            "mean_optimal_threshold": eval_res["optimal_threshold_mean"],
            "std_optimal_threshold": eval_res["optimal_threshold_std"],
            "final_coefficients": final_meta_reg.get_coefficients(),
            "mean_fold_coefficients": {
                "visual_weight": mean_vis_w,
                "vital_weight": mean_vit_w,
                "intercept": mean_intercept,
            },
            "fold_models": [
                {
                    "fold": int(r["fold"]),
                    "test_user": r["test_user"],
                    "optimal_threshold": float(r["optimal_threshold"]),
                    "estimator": m.model,
                    "coefficients": m.get_coefficients(),
                }
                for r, m in zip(fold_records, fold_models)
            ],
            "metrics": {
                "optimal_threshold_mean": eval_res["optimal_threshold_mean"],
                "optimal_threshold_std": eval_res["optimal_threshold_std"],
                "roc_auc": eval_res["roc_auc"],
                "pr_auc": eval_res["pr_auc"],
                "pearson_r": eval_res["pearson_r"],
                "spearman_rho": eval_res["spearman_rho"],
                "mae": eval_res["mae"],
                "rmse": eval_res["rmse"],
                "r2": eval_res["r2"],
                "accuracy": eval_res["accuracy"],
                "balanced_acc": eval_res["balanced_acc"],
                "f1_macro": eval_res["f1_macro"],
            },
        }
        final_meta_reg.save(model_path, extra_metadata=extra_meta)
        log.info(f"💾 学習済みメタモデルを保存しました: {model_path}")

    # 4. 可視化プロット保存
    plot_path = None
    if save_plots:
        plot_path = generate_visualizations(
            output_dir=meta_output_dir,
            y_true=labels,
            y_pred=oof_meta_preds,
            fold_df=fold_df,
            model_name=tag,
            mae=eval_res["mae"],
            rmse=eval_res["rmse"],
            r2=eval_res["r2"],
            r_val=eval_res["pearson_r"],
            acc=eval_res["accuracy"],
            balanced_acc=eval_res["balanced_acc"],
            auc=eval_res["roc_auc"],
            pr_auc=eval_res["pr_auc"],
            log=log,
            threshold=fold_thresholds_arr,
        )

    # 5. 個別 Markdown レポート保存
    report_path = save_markdown_report(
        output_dir=meta_output_dir,
        window_size=window_size,
        model_name=tag,
        n_users=n_users,
        mae=eval_res["mae"],
        rmse=eval_res["rmse"],
        r2=eval_res["r2"],
        r_val=eval_res["pearson_r"],
        rho_val=eval_res["spearman_rho"],
        acc=eval_res["accuracy"],
        balanced_acc=eval_res["balanced_acc"],
        auc=eval_res["roc_auc"],
        pr_auc=eval_res["pr_auc"],
        f1_macro=eval_res["f1_macro"],
        stats_dict=stats_dict,
        fold_df=fold_df,
        log=log,
        threshold_criterion=th_criterion,
    )

    eval_res.update({
        "model_name": canonical_name,
        "selected_visual_model": selected_visual_model,
        "selected_vital_model": selected_vital_model,
        "visual_weight": mean_vis_w,
        "vital_weight": mean_vit_w,
        "intercept": mean_intercept,
        "oof_preds": oof_meta_preds,
        "oof_thresholds": fold_thresholds_arr,
        "oof_csv": str(oof_csv),
        "model_path": str(model_path) if save_models else None,
        "plot_path": str(plot_path) if plot_path else None,
        "report_path": str(report_path),
    })
    return eval_res
