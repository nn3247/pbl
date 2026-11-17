"""
Step 7-0: 学習前 閾値決定（Pre-Learning Threshold Decision）パイプライン
(learning/downstream_task/threshold_decision/pipeline.py)

Base Model および Meta Model の学習前に実行し、各 LOSO Fold 内の
非テストデータ（Train）の GMM ソフト確率ラベル `y_soft_train` と正解ラベル `label` から
Balanced Accuracy 最大化（Youden's J Index: `youden_j`）を適用して最適なカットオフ閾値 $th^*_k$ を決定します。
決定された Fold 別最適閾値は `models/{window_size}/threshold_decision/fold_optimal_thresholds.csv` に保存され、
後続の Base Model / Meta Model の学習・評価・推論パイプラインへ引き継がれます。
"""

from __future__ import annotations
import logging
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

_CURRENT_DIR = Path(__file__).resolve().parent
_DOWNSTREAM_DIR = _CURRENT_DIR.parent
_LEARNING_ROOT = _DOWNSTREAM_DIR.parent
for _p in (_CURRENT_DIR, _DOWNSTREAM_DIR, _LEARNING_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from config_loader import load_config, log
except ImportError:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("threshold_decision")

    def load_config(cfg_file: str = "config.yaml") -> dict:
        return {}

try:
    from ..data_loader import load_embeddings_and_meta, load_explicit_features
    from ..base_model.runner import LOSOFoldData, prepare_loso_folds
except (ImportError, ValueError):
    from downstream_task.data_loader import load_embeddings_and_meta, load_explicit_features
    from downstream_task.base_model.runner import LOSOFoldData, prepare_loso_folds

from .criteria import SUPPORTED_THRESHOLD_CRITERIA, normalize_criterion_name
from .reporter import (
    generate_threshold_visualizations,
    save_single_threshold_report,
    save_threshold_decision_summary_report,
)
from .runner import run_loso_threshold_tuning, run_prelearning_fold_threshold_tuning


def load_saved_fold_thresholds(
    window_model_dir: Path,
    criterion: Optional[str] = None,
) -> Tuple[Dict[int, float], Dict[str, float]]:
    """
    `models/{window_size}/threshold_decision/fold_optimal_thresholds.csv` から
    学習前に決定された Fold 別の最適閾値 (fold -> th*, test_user -> th*) を読み込みます。
    """
    th_csv = Path(window_model_dir) / "threshold_decision" / "fold_optimal_thresholds.csv"
    if not th_csv.exists():
        return {}, {}

    try:
        df = pd.read_csv(th_csv)
        if criterion is not None and "criterion" in df.columns:
            canon = normalize_criterion_name(criterion)
            sub = df[df["criterion"] == canon]
            if sub.empty:
                return {}, {}
            df = sub

        fold_map: Dict[int, float] = {}
        user_map: Dict[str, float] = {}
        for _, r in df.iterrows():
            th_val = float(r["optimal_threshold"])
            if "fold" in r and pd.notna(r["fold"]):
                fold_map[int(r["fold"])] = th_val
            if "test_user" in r and pd.notna(r["test_user"]):
                user_map[str(r["test_user"])] = th_val
        return fold_map, user_map
    except Exception:
        return {}, {}


class ThresholdDecisionPipeline:
    """
    LOSO-CV 完全データリークフリー 学習前 閾値決定パイプライン (Step 7-0)。
    Balanced Accuracy 最大化（Youden's J Index: `youden_j`）を用いて、
    各 LOSO Fold の学習データから最適カットオフ閾値 $th^*_k$ を決定します。
    """

    def __init__(self, cfg: Optional[dict] = None, config_path: Optional[str] = None):
        if cfg is None:
            cfg = load_config(config_path) if config_path else load_config()
        self.cfg = cfg
        self.paths = cfg.get("paths", {})
        self.dataset_cfg = cfg.get("dataset", {})
        self.clustering_cfg = cfg.get("clustering", {})
        self.gmm_cfg = self.clustering_cfg.get("gmm", {})
        self.downstream_cfg = cfg.get("downstream_task", {})
        self.th_cfg = self.downstream_cfg.get("threshold_decision", {})
        self.loso_cfg = cfg.get("loso_cv", {})

        self.window_size = self.dataset_cfg.get("window_size", "10s")
        self.learning_root = _LEARNING_ROOT

        data_root = Path(self.paths.get("data_root", "data"))
        if not data_root.is_absolute():
            data_root = self.learning_root / data_root
        self.data_dir = data_root / self.window_size
        self.data_dir.mkdir(parents=True, exist_ok=True)

        input_data_root = Path(self.paths.get("input_data_root", "../preprocess/data/windowed"))
        if not input_data_root.is_absolute():
            input_data_root = self.learning_root / input_data_root
        self.input_data_dir = input_data_root / self.window_size

        model_root = Path(self.paths.get("model_dir", "models"))
        if not model_root.is_absolute():
            model_root = self.learning_root / model_root
        self.model_root = model_root
        self.window_model_dir = model_root / self.window_size
        self.output_dir = self.window_model_dir / "threshold_decision"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.default_cov = (
            self.loso_cfg.get("covariance_type")
            or self.gmm_cfg.get("covariance_type", "full")
        )
        self.seed = int(self.loso_cfg.get("random_state", self.downstream_cfg.get("random_state", 42)))

        self.default_criterion = normalize_criterion_name(
            self.th_cfg.get("default_criterion", "youden_j")
        )
        raw_criteria = self.th_cfg.get("criteria", ["youden_j"])
        self.criteria: List[str] = list(
            dict.fromkeys([normalize_criterion_name(c) for c in raw_criteria])
        )
        if self.default_criterion not in self.criteria:
            self.criteria.insert(0, self.default_criterion)

        sr = self.th_cfg.get("search_range", [0.05, 0.95])
        self.search_range: Tuple[float, float] = (float(sr[0]), float(sr[1]))
        self.search_step: float = float(self.th_cfg.get("search_step", 0.005))
        self.use_exact_curve_points: bool = bool(self.th_cfg.get("use_exact_curve_points", True))
        self.fallback_threshold: float = float(self.th_cfg.get("fallback_threshold", 0.5))
        self.validation_split_mode: str = str(self.th_cfg.get("validation_split_mode", "train_oof"))
        self.val_subject_ratio: float = float(self.th_cfg.get("val_subject_ratio", 0.2))
        self.save_plots: bool = bool(self.th_cfg.get("save_plots", self.downstream_cfg.get("save_plots", True)))

    def run(
        self,
        window_size: Optional[str] = None,
        criteria: Optional[Sequence[str]] = None,
        target_models: Optional[Sequence[str]] = None,
        weights: Optional[Union[Tuple[float, float], List[float]]] = None,
        covariance_type: Optional[str] = None,
        apply_subject_norm: Optional[bool] = None,
        loso_folds: Optional[List[LOSOFoldData]] = None,
        meta_df: Optional[pd.DataFrame] = None,
        vis_feat_names: Optional[List[str]] = None,
        vital_feat_names: Optional[List[str]] = None,
        oof_gmm_soft: Optional[np.ndarray] = None,
        oof_sample_weight: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        学習前の閾値決定（Youden's J Index による Balanced Accuracy 最大化）を実行し、
        各 LOSO Fold の最適閾値 $th^*_k$ を算出・保存します。
        """
        ws = window_size or self.window_size
        window_model_dir = self.model_root / ws
        output_dir = window_model_dir / "threshold_decision"
        output_dir.mkdir(parents=True, exist_ok=True)

        active_criteria = (
            list(dict.fromkeys([normalize_criterion_name(c) for c in criteria]))
            if criteria
            else list(self.criteria)
        )
        primary_criterion = (
            active_criteria[0]
            if criteria and len(active_criteria) > 0
            else self.default_criterion
        )
        if primary_criterion not in active_criteria:
            active_criteria.insert(0, primary_criterion)

        cov_type = covariance_type or self.default_cov
        vis_pls = int(self.loso_cfg.get("n_vis_pls", 3))
        bio_pls = int(self.loso_cfg.get("n_bio_pls", 3))
        subj_norm = (
            apply_subject_norm
            if apply_subject_norm is not None
            else bool(self.loso_cfg.get("apply_subject_norm", True))
        )

        if weights is not None and len(weights) >= 2:
            w_tuple = (float(weights[0]), float(weights[1]))
        else:
            default_weights = self.loso_cfg.get("weights", [1.0, 0.0])
            w_tuple = (float(default_weights[0]), float(default_weights[1]))

        log.info("=" * 78)
        log.info(f"🎯 [Step 7-0: 学習前 閾値決定 (Pre-Learning Threshold Decision) 開始] [Window: {ws}]")
        log.info(
            f"   採用基準: '{primary_criterion}' (Balanced Accuracy 最大化 / Youden's J Index) | "
            f"探索基準一覧: {active_criteria} | 検証モード: {self.validation_split_mode}"
        )
        log.info(f"   保存先ディレクトリ: {output_dir}")
        log.info("=" * 78)

        start_t = time.time()

        # 1. LOSO Fold データが未構築の場合は特徴量ロード & Fold 分割を実行
        if loso_folds is None or meta_df is None or oof_gmm_soft is None:
            data_dir = self.data_dir.parent / ws
            input_data_dir = self.input_data_dir.parent / ws
            emb_vis, emb_bio, meta_df = load_embeddings_and_meta(
                data_dir=data_dir,
                input_data_dir=input_data_dir,
                learning_root=self.learning_root,
                window_size=ws,
                log=log,
            )
            explicit_feats, explicit_names = load_explicit_features(
                meta_df=meta_df,
                input_data_dir=input_data_dir,
                data_dir=data_dir,
                window_size=ws,
                log=log,
            )
            loso_folds, vis_feat_names, vital_feat_names, oof_gmm_soft, oof_sample_weight = prepare_loso_folds(
                emb_vis=emb_vis,
                emb_bio=emb_bio,
                meta_df=meta_df,
                explicit_feats=explicit_feats,
                explicit_names=explicit_names,
                cfg=self.cfg,
                log=log,
                n_vis_pls=vis_pls,
                n_bio_pls=bio_pls,
                cov_type=cov_type,
                weights=w_tuple,
                apply_subject_norm=subj_norm,
                compute_thresholds=False,
            )

        subject_col = "username" if "username" in meta_df.columns else "subject_id"
        label_col = "label" if "label" in meta_df.columns else "label_mean"
        subjects = meta_df[subject_col].astype(str).values
        labels = meta_df[label_col].values.astype(float)

        summary_rows: List[Dict] = []
        all_results: Dict[str, Dict] = {}
        primary_tuning_res: Optional[Dict] = None

        # 2. 学習前 GMM ソフトラベルに対する各基準の LOSO Fold 閾値最適化
        model_tag = "gmm_soft_label"
        for crit in active_criteria:
            tuning_res = run_prelearning_fold_threshold_tuning(
                loso_folds=loso_folds,
                y_true_all=labels,
                subjects=subjects,
                criterion=crit,
                search_range=self.search_range,
                search_step=self.search_step,
                use_exact_curve_points=self.use_exact_curve_points,
                fallback_threshold=self.fallback_threshold,
                validation_split_mode=self.validation_split_mode,
                val_subject_ratio=self.val_subject_ratio,
                random_state=self.seed,
                model_tag=model_tag,
                log=log,
            )

            if crit == primary_criterion:
                primary_tuning_res = tuning_res

            fold_csv_path = output_dir / f"loso_{model_tag}_threshold_{crit}_folds.csv"
            tuning_res["fold_df"].to_csv(fold_csv_path, index=False)

            tuned_df = meta_df.copy()
            tuned_df["y_soft_oof"] = tuning_res["oof_preds"]
            tuned_df["threshold_criterion"] = crit
            tuned_df["optimal_threshold"] = tuning_res["oof_thresholds"]
            tuned_df["y_pred_binary_05"] = tuning_res["oof_pred_binary_05"]
            tuned_df["y_pred_binary_tuned"] = tuning_res["oof_pred_binary_tuned"]
            oof_out_path = output_dir / f"loso_{model_tag}_threshold_{crit}_oof.csv"
            tuned_df.to_csv(oof_out_path, index=False)

            plot_path = None
            if self.save_plots:
                plot_path = generate_threshold_visualizations(
                    output_dir=output_dir,
                    model_tag=model_tag,
                    y_true=labels,
                    y_pred=tuning_res["oof_preds"],
                    tuning_res=tuning_res,
                    log=log,
                )

            report_path = save_single_threshold_report(
                output_dir=output_dir,
                window_size=ws,
                model_tag=model_tag,
                tuning_res=tuning_res,
                log=log,
            )

            base_d = tuning_res["baseline_oof_detail"]
            tuned_d = tuning_res["tuned_oof_detail"]
            imp = tuning_res["improvement"]

            summary_rows.append({
                "scope": "pre_learning",
                "model": model_tag,
                "criterion": crit,
                "mean_th": tuning_res["mean_threshold"],
                "std_th": tuning_res["std_threshold"],
                "min_th": tuning_res["min_threshold"],
                "max_th": tuning_res["max_threshold"],
                "balanced_acc_05": base_d["balanced_acc"],
                "balanced_acc_tuned": tuned_d["balanced_acc"],
                "delta_balanced_acc": imp["balanced_acc"],
                "youden_j_05": base_d["youden_j"],
                "youden_j_tuned": tuned_d["youden_j"],
                "delta_youden_j": imp["youden_j"],
                "f1_macro_05": base_d["f1_macro"],
                "f1_macro_tuned": tuned_d["f1_macro"],
                "delta_f1_macro": imp["f1_macro"],
                "f1_neg_05": base_d["f1_neg"],
                "f1_neg_tuned": tuned_d["f1_neg"],
                "delta_f1_neg": imp["f1_neg"],
                "specificity_05": base_d["specificity"],
                "specificity_tuned": tuned_d["specificity"],
                "sensitivity_05": base_d["sensitivity"],
                "sensitivity_tuned": tuned_d["sensitivity"],
                "accuracy_05": base_d["accuracy"],
                "accuracy_tuned": tuned_d["accuracy"],
                "roc_auc": tuned_d["roc_auc"],
                "pr_auc": tuned_d["pr_auc"],
                "oof_csv": str(oof_out_path),
                "fold_csv": str(fold_csv_path),
                "report_path": str(report_path),
                "plot_path": str(plot_path) if plot_path else None,
            })

            all_results[f"{model_tag}__{crit}"] = tuning_res

        # 3. プライマリ基準 (youden_j) で決定した閾値を LOSO Fold データに確定反映し、共通 CSV に保存
        if primary_tuning_res is not None:
            for f_data in loso_folds:
                f_data.optimal_threshold = float(
                    primary_tuning_res["fold_threshold_map"].get(int(f_data.fold), self.fallback_threshold)
                )
                f_data.threshold_criterion = primary_criterion

            fold_opt_csv = output_dir / "fold_optimal_thresholds.csv"
            primary_tuning_res["fold_df"].to_csv(fold_opt_csv, index=False)
            log.info(
                f"💾 学習前決定の Fold 別最適閾値 ({primary_criterion}) を保存しました: {fold_opt_csv} "
                f"(Mean th* = {primary_tuning_res['mean_threshold']:.4f} ± {primary_tuning_res['std_threshold']:.4f})"
            )

        # 4. 全体の比較サマリー CSV & マスターレポート保存
        comparison_df = pd.DataFrame(summary_rows)
        comp_csv = output_dir / "threshold_decision_comparison_summary.csv"
        comparison_df.to_csv(comp_csv, index=False)

        master_report = save_threshold_decision_summary_report(
            output_dir=output_dir,
            window_size=ws,
            default_criterion=primary_criterion,
            comparison_df=comparison_df,
            log=log,
        )

        elapsed_total = time.time() - start_t
        log.info("=" * 78)
        log.info(
            f"✅ [Step 7-0: 学習前 閾値決定完了] (所要時間: {elapsed_total:.2f} 秒) | "
            f"採用基準='{primary_criterion}' | 平均最適閾値 th*={primary_tuning_res['mean_threshold']:.4f} | "
            f"レポート: {master_report}"
        )
        log.info("=" * 78)

        return {
            "window_size": ws,
            "output_dir": str(output_dir),
            "default_criterion": primary_criterion,
            "criteria": active_criteria,
            "mean_optimal_threshold": (
                primary_tuning_res["mean_threshold"] if primary_tuning_res else self.fallback_threshold
            ),
            "std_optimal_threshold": (
                primary_tuning_res["std_threshold"] if primary_tuning_res else 0.0
            ),
            "fold_threshold_map": (
                primary_tuning_res["fold_threshold_map"] if primary_tuning_res else {}
            ),
            "user_threshold_map": (
                primary_tuning_res["user_threshold_map"] if primary_tuning_res else {}
            ),
            "oof_thresholds": (
                primary_tuning_res["oof_thresholds"] if primary_tuning_res else None
            ),
            "loso_folds": loso_folds,
            "vis_feat_names": vis_feat_names,
            "vital_feat_names": vital_feat_names,
            "oof_gmm_soft": oof_gmm_soft,
            "oof_sample_weight": oof_sample_weight,
            "meta_df": meta_df,
            "comparison_df": comparison_df,
            "comparison_csv": str(comp_csv),
            "report_path": str(master_report),
            "results": all_results,
        }


__all__ = ["ThresholdDecisionPipeline", "load_saved_fold_thresholds"]
