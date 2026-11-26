"""
メタモデル (Visual + Vital 予測値統合) LOSO-CV 学習・評価・最終推論パイプライン
(learning/downstream_task/meta_model/pipeline.py)
"""

from __future__ import annotations
import argparse
import copy
import logging
from pathlib import Path
import re
import sys
import time
from typing import Dict, List, Optional, Sequence
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
    log = logging.getLogger("meta_model")

    def load_config(cfg_file: str = "config.yaml") -> dict:
        return {}

try:
    from ..base_model.models import normalize_base_model_name
    from ..base_model.pipeline import BaseModelPipeline
    from ..metrics import build_summary_row, compute_oof_and_fold_summary
    from ..threshold_decision.pipeline import load_saved_fold_thresholds
    from .models import DEFAULT_META_MODELS, normalize_meta_model_name
    from .reporter import plot_model_comparison, save_meta_model_comparison_report
    from .runner import run_single_meta_model_loso
except (ImportError, ValueError):
    from downstream_task.base_model.models import normalize_base_model_name
    from downstream_task.base_model.pipeline import BaseModelPipeline
    from downstream_task.metrics import build_summary_row, compute_oof_and_fold_summary
    from downstream_task.threshold_decision.pipeline import load_saved_fold_thresholds
    from downstream_task.meta_model.models import DEFAULT_META_MODELS, normalize_meta_model_name
    from downstream_task.meta_model.reporter import plot_model_comparison, save_meta_model_comparison_report
    from downstream_task.meta_model.runner import run_single_meta_model_loso


class MetaModelPipeline:
    """
    config.yaml で指定された Visual ベースモデルと Vital ベースモデルの推論結果を用いて
    4種のメタモデル (LogisticRegression, Ridge, Lasso, ElasticNet) を作成・評価・保存し、
    最終的な推論結果 (final_inference_predictions.csv) と report.md を出力するパイプライン。
    """

    def __init__(self, cfg: Optional[dict] = None):
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg
        self.paths = cfg.get("paths", {})
        self.dataset_cfg = cfg.get("dataset", {})
        self.downstream_cfg = cfg.get("downstream_task", {})
        self.meta_model_cfg = self.downstream_cfg.get("meta_model", {})

        self.window_size = self.dataset_cfg.get("window_size", "10s")
        self.learning_root = _LEARNING_ROOT

        model_root = Path(self.paths.get("model_dir", "models"))
        if not model_root.is_absolute():
            model_root = self.learning_root / model_root

        self.base_model_dir = model_root / self.window_size / "base_model"
        self.output_dir = model_root / self.window_size / "meta_model"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_best_base_model(self, modality: str) -> str:
        """
        base_model 評価結果（base_models_comparison.csv / {modality}_models_comparison.csv / report.md）から
        最高 ROC-AUC のモデル名を自動取得します。未生成の場合は BaseModelPipeline を自動実行します。
        """
        mod_key = modality.lower().strip()
        mod_csv = self.base_model_dir / mod_key / f"{mod_key}_models_comparison.csv"
        all_csv = self.base_model_dir / "base_models_comparison.csv"
        report_md = self.base_model_dir / "report.md"

        def _pick_from_csv() -> tuple[Optional[str], Optional[float]]:
            df_sub = pd.DataFrame()
            if mod_csv.exists():
                df_sub = pd.read_csv(mod_csv)
            elif all_csv.exists():
                df_all = pd.read_csv(all_csv)
                if "modality" in df_all.columns:
                    df_sub = df_all[df_all["modality"].astype(str).str.lower() == mod_key]
            if not df_sub.empty and "model" in df_sub.columns and "roc_auc" in df_sub.columns:
                valid = df_sub[df_sub["roc_auc"].notna()]
                if not valid.empty:
                    best_row = valid.loc[valid["roc_auc"].idxmax()]
                    return str(best_row["model"]), float(best_row["roc_auc"])
            return None, None

        def _pick_from_report() -> Optional[str]:
            if not report_md.exists():
                return None
            text = report_md.read_text(encoding="utf-8")
            m = re.search(rf"最高 ROC-AUC モデル \({re.escape(mod_key)}\)\*\*:\s*`([^`]+)`", text)
            if m:
                return m.group(1).strip()
            return None

        best_model, best_auc = _pick_from_csv()
        if best_model is None:
            best_model = _pick_from_report()

        if best_model is None:
            log.info(
                f"ℹ️ [{mod_key.upper()}] base_model 比較結果が見つからないため、"
                f"先に BaseModelPipeline で全ベースモデルを評価します..."
            )
            BaseModelPipeline(self.cfg).run(modalities=[mod_key])
            best_model, best_auc = _pick_from_csv()
            if best_model is None:
                best_model = _pick_from_report()

        if best_model is None:
            log.warning(f"⚠️ [{mod_key.upper()}] 最高モデルの特定に失敗したため 'lightgbm' にフォールバックします。")
            return "lightgbm"

        auc_msg = f" (ROC-AUC={best_auc:.4f})" if best_auc is not None else " (from report.md)"
        log.info(f"🏆 [{mod_key.upper()}] base_model 最高モデルを自動選択しました: '{best_model}'{auc_msg}")
        return normalize_base_model_name(best_model)

    def _resolve_selected_base_models(
        self,
        visual_model: Optional[str] = None,
        vital_model: Optional[str] = None,
    ) -> tuple[str, str]:
        """config.yaml または引数から採用する visual / vital ベースモデル名を解決します ('auto' で最高モデル自動選択)"""
        selected_cfg = self.meta_model_cfg.get("selected_base_models", {})
        vis_raw = (
            visual_model
            or selected_cfg.get("visual")
            or self.meta_model_cfg.get("visual_model")
            or self.downstream_cfg.get("visual_base_model")
            or "auto"
        )
        vit_raw = (
            vital_model
            or selected_cfg.get("vital")
            or self.meta_model_cfg.get("vital_model")
            or self.downstream_cfg.get("vital_base_model")
            or "auto"
        )

        vis_resolved = (
            self._resolve_best_base_model("visual")
            if str(vis_raw).lower().strip() in ("auto", "best")
            else normalize_base_model_name(vis_raw)
        )
        vit_resolved = (
            self._resolve_best_base_model("vital")
            if str(vit_raw).lower().strip() in ("auto", "best")
            else normalize_base_model_name(vit_raw)
        )
        return vis_resolved, vit_resolved

    def _load_or_build_base_oof(
        self,
        modality: str,
        base_model_name: str,
    ) -> pd.DataFrame:
        """
        指定されたモダリティ・ベースモデルの OOF 予測 CSV を読み込みます。
        未作成の場合は BaseModelPipeline を自動実行して生成します。
        """
        oof_csv = self.base_model_dir / modality / f"loso_{modality}_{base_model_name}_oof_predictions.csv"
        if not oof_csv.exists():
            log.info(
                f"ℹ️ [{modality.upper()}] ベースモデル '{base_model_name}' の OOF 予測が未生成のため、"
                f"先に BaseModelPipeline で作成します..."
            )
            base_pipe = BaseModelPipeline(self.cfg)
            base_pipe.run(modalities=[modality], models=[base_model_name])

        if not oof_csv.exists():
            raise FileNotFoundError(
                f"ベースモデル ({modality} / {base_model_name}) の OOF 予測ファイルが見つかりません: {oof_csv}"
            )

        log.info(f"✓ [{modality.upper()}] ベースモデル ({base_model_name}) の予測値をロードしました: {oof_csv.name}")
        return pd.read_csv(oof_csv)

    def run(
        self,
        models: Optional[Sequence[str]] = None,
        visual_model: Optional[str] = None,
        vital_model: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        選択された Visual / Vital ベースモデルを使って全メタモデルを作成・評価し、
        report.md と最終推論結果 CSV を保存します。
        """
        sel_vis, sel_vital = self._resolve_selected_base_models(visual_model, vital_model)
        raw_meta_models = list(
            models
            or self.meta_model_cfg.get("models")
            or DEFAULT_META_MODELS
        )
        target_meta_models = [normalize_meta_model_name(m) for m in raw_meta_models]
        default_final_model = normalize_meta_model_name(
            self.meta_model_cfg.get("default_final_model", target_meta_models[0])
        )

        log.info("=" * 78)
        log.info(f"🚀 [Step 7-2: Meta Model LOSO-CV パイプライン開始] [Window: {self.window_size}]")
        log.info(f"   採用 Visual ベースモデル : {sel_vis}")
        log.info(f"   採用 Vital  ベースモデル : {sel_vital}")
        log.info(f"   対象メタモデル一覧       : {target_meta_models}")
        log.info(f"   保存先ディレクトリ       : {self.output_dir}")
        log.info("=" * 78)

        start_t = time.time()

        # 1. Visual / Vital ベースモデルの OOF 予測読み込み
        vis_oof_df = self._load_or_build_base_oof("visual", sel_vis)
        vit_oof_df = self._load_or_build_base_oof("vital", sel_vital)

        if len(vis_oof_df) != len(vit_oof_df):
            raise ValueError(
                f"Visual OOF ({len(vis_oof_df)}) と Vital OOF ({len(vit_oof_df)}) のサンプル数が一致しません。"
            )

        y_pred_vis = (
            vis_oof_df["y_pred_visual_oof"].values
            if "y_pred_visual_oof" in vis_oof_df.columns
            else vis_oof_df["y_pred_oof"].values
        ).astype(np.float64)
        y_pred_vit = (
            vit_oof_df["y_pred_vital_oof"].values
            if "y_pred_vital_oof" in vit_oof_df.columns
            else vit_oof_df["y_pred_oof"].values
        ).astype(np.float64)

        y_gmm_soft = (
            vis_oof_df["y_gmm_soft_oof"].values
            if "y_gmm_soft_oof" in vis_oof_df.columns
            else vis_oof_df["label"].values
        ).astype(np.float64)

        if "sample_weight_oof" in vis_oof_df.columns:
            sample_weight_oof = vis_oof_df["sample_weight_oof"].values.astype(np.float64)
        else:
            imbalance_penalty = float(self.downstream_cfg.get("imbalance_penalty", 4.0))
            sample_weight_oof = (y_gmm_soft * 1.0 + (1.0 - y_gmm_soft) * imbalance_penalty).astype(np.float64)

        # メタデータ列のみ抽出
        drop_pred_cols = {
            "y_gmm_soft_oof", "sample_weight_oof", "optimal_threshold", "threshold_criterion",
            "y_pred_visual_oof", "y_pred_vital_oof", "y_pred_oof", "y_pred_binary"
        }
        meta_cols = [c for c in vis_oof_df.columns if c not in drop_pred_cols]
        meta_df = vis_oof_df[meta_cols].copy()
        subject_col = "username" if "username" in meta_df.columns else "subject_id"
        label_col = "label" if "label" in meta_df.columns else "label_mean"
        labels = meta_df[label_col].values.astype(float)
        subjects = meta_df[subject_col].astype(str).values

        # 学習前決定の最適閾値 th*_k の取得 (Base OOF CSV または threshold_decision/fold_optimal_thresholds.csv)
        oof_thresholds: Optional[np.ndarray] = None
        if "optimal_threshold" in vis_oof_df.columns and vis_oof_df["optimal_threshold"].notna().all():
            oof_thresholds = vis_oof_df["optimal_threshold"].values.astype(np.float64)
        else:
            _, user_th_map = load_saved_fold_thresholds(self.output_dir.parent)
            if user_th_map:
                oof_thresholds = np.array(
                    [float(user_th_map.get(str(u), 0.5)) for u in subjects],
                    dtype=np.float64,
                )

        # ベースモデル単体の参照メトリクス算出 (学習前決定の最適閾値 th*_k を適用)
        ref_th = oof_thresholds if oof_thresholds is not None else 0.5
        vis_ref = compute_oof_and_fold_summary(labels, y_pred_vis, [], threshold=ref_th)
        vit_ref = compute_oof_and_fold_summary(labels, y_pred_vit, [], threshold=ref_th)
        base_metrics_summary = {
            "visual": {"model_name": sel_vis, **vis_ref},
            "vital": {"model_name": sel_vital, **vit_ref},
        }

        # 2. 各メタモデルの LOSO-CV 学習・評価・保存
        summary_rows: List[Dict] = []
        final_infer_df = meta_df.copy()
        final_infer_df["y_gmm_soft_oof"] = y_gmm_soft
        final_infer_df[f"y_pred_visual_{sel_vis}"] = y_pred_vis
        final_infer_df[f"y_pred_vital_{sel_vital}"] = y_pred_vit

        meta_preds_dict: Dict[str, np.ndarray] = {}
        final_thresholds_arr: Optional[np.ndarray] = oof_thresholds

        for m_name in target_meta_models:
            res = run_single_meta_model_loso(
                meta_model_name=m_name,
                selected_visual_model=sel_vis,
                selected_vital_model=sel_vital,
                y_pred_visual_oof=y_pred_vis,
                y_pred_vital_oof=y_pred_vit,
                y_gmm_soft_oof=y_gmm_soft,
                sample_weight_oof=sample_weight_oof,
                meta_df=meta_df,
                window_size=self.window_size,
                meta_output_dir=self.output_dir,
                cfg=self.cfg,
                log=log,
                oof_thresholds=oof_thresholds,
            )
            row = build_summary_row(
                model_name=res["model_name"],
                eval_res=res,
                extra_fields={
                    "visual_base_model": sel_vis,
                    "vital_base_model": sel_vital,
                    "visual_weight": round(res["visual_weight"], 4),
                    "vital_weight": round(res["vital_weight"], 4),
                    "intercept": round(res["intercept"], 4),
                    "model_path": res["model_path"],
                },
            )
            summary_rows.append(row)
            meta_preds_dict[res["model_name"]] = res["oof_preds"]
            final_infer_df[f"y_pred_meta_{res['model_name']}"] = res["oof_preds"]
            if final_thresholds_arr is None and res.get("oof_thresholds") is not None:
                final_thresholds_arr = res["oof_thresholds"]

        # 3. 最終推論結果列 (final_prediction & final_binary_prediction) の付与と保存 (最適閾値 th*_k 適用)
        chosen_final = (
            default_final_model
            if default_final_model in meta_preds_dict
            else next(iter(meta_preds_dict.keys()))
        )
        applied_th = final_thresholds_arr if final_thresholds_arr is not None else 0.5
        final_infer_df["optimal_threshold"] = applied_th
        final_infer_df["final_meta_model"] = chosen_final
        final_infer_df["final_prediction"] = meta_preds_dict[chosen_final]
        final_infer_df["final_binary_prediction"] = (meta_preds_dict[chosen_final] >= applied_th).astype(int)

        final_csv = self.output_dir / "final_inference_predictions.csv"
        final_infer_df.to_csv(final_csv, index=False)
        log.info(f"💾 最終推論結果 CSV を保存しました: {final_csv}")

        # 4. メタモデル比較 CSV・プロット・report.md 保存
        df_meta = pd.DataFrame(summary_rows)
        comp_csv = self.output_dir / "meta_models_comparison.csv"
        df_meta.to_csv(comp_csv, index=False)
        log.info(f"💾 メタモデル比較サマリー CSV を保存しました: {comp_csv}")

        if self.downstream_cfg.get("save_plots", True) and not df_meta.empty:
            plot_model_comparison(df_meta, self.output_dir, log)

        save_meta_model_comparison_report(
            meta_output_dir=self.output_dir,
            window_size=self.window_size,
            selected_visual_model=sel_vis,
            selected_vital_model=sel_vital,
            df_meta=df_meta,
            base_metrics_summary=base_metrics_summary,
            log=log,
        )

        elapsed_total = time.time() - start_t
        log.info("=" * 78)
        log.info(
            f"🏆 [Meta Model Pipeline] 全処理完了 ({elapsed_total:.2f}s) "
            f"[Base: Visual={sel_vis}, Vital={sel_vital}]"
        )
        for _, r in df_meta.iterrows():
            log.info(
                f"  - {r['model']:<20}: "
                f"ROC-AUC={r['roc_auc']:.4f} (±{r['roc_auc_std']:.3f}) | "
                f"PR-AUC={r['pr_auc']:.4f} | "
                f"Pearson r={r['pearson_r']:.4f} | "
                f"MAE={r['mae']:.4f} | "
                f"w_vis={r['visual_weight']:.3f}, w_vital={r['vital_weight']:.3f}"
            )
        log.info("=" * 78)

        return df_meta
