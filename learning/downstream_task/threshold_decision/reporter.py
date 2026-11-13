"""
閾値決定（Threshold Decision）レポート・可視化出力モジュール
(learning/downstream_task/threshold_decision/reporter.py)

生成する成果物:
1. 閾値感度曲線 & 混同行列比較プロット (`loso_{model}_threshold_{criterion}_plots.png`):
   - パネル1: 閾値 $th \\in [0, 1]$ に対する Youden's J / Balanced Acc / F1-Macro / Sensitivity / Specificity の推移曲線と最適閾値 $th^*$
   - パネル2: 被験者（LOSO Fold）別の最適閾値 $th^*_k$ と Balanced Accuracy (Fixed 0.50 vs Tuned $th^*_k$)
   - パネル3: 全体 OOF 指標の比較棒グラフ (Fixed 0.50 vs Tuned $th^*_k$)
   - パネル4: 集中/休憩クラスの予測確率ヒストグラムと最適カットオフ境界線
2. 個別モデル×基準の Markdown レポート (`loso_{model}_threshold_{criterion}_report.md`)
3. 全体比較サマリーレポート (`report.md`)
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_threshold_visualizations(
    output_dir: Path,
    model_tag: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tuning_res: Dict,
    log=None,
) -> Optional[Path]:
    """
    閾値最適化の4パネル総合可視化グラフを生成して保存します。
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        crit = tuning_res["criterion"]
        plot_path = output_dir / f"loso_{model_tag}_threshold_{crit}_plots.png"

        curve_df: pd.DataFrame = tuning_res["curve_df"]
        fold_df: pd.DataFrame = tuning_res["fold_df"]
        base_oof: Dict = tuning_res["baseline_oof_detail"]
        tuned_oof: Dict = tuning_res["tuned_oof_detail"]
        mean_th = float(tuning_res["mean_threshold"])

        fig, axes = plt.subplots(2, 2, figsize=(15, 11))
        fig.suptitle(
            f"Threshold Decision Analysis: {model_tag} | Criterion: {crit} "
            f"(Mean th* = {mean_th:.3f} ± {tuning_res['std_threshold']:.3f})",
            fontsize=13,
            fontweight="bold",
        )

        # ------------------------------------------------------------------
        # Panel 1: 閾値推移曲線 (Balanced Acc, F1-Macro, Sensitivity, Specificity, Youden's J)
        # ------------------------------------------------------------------
        ax1 = axes[0, 0]
        ths = curve_df["threshold"].values
        ax1.plot(ths, curve_df["balanced_acc"].values, label="Balanced Acc", lw=2.2, color="#1f77b4")
        ax1.plot(ths, curve_df["f1_macro"].values, label="F1-Macro", lw=2.2, color="#2ca02c")
        ax1.plot(ths, curve_df["sensitivity"].values, label="Sensitivity (Focus Recall)", lw=1.5, ls="--", color="#17becf")
        ax1.plot(ths, curve_df["specificity"].values, label="Specificity (Rest Recall)", lw=1.5, ls="--", color="#ff7f0e")
        ax1.plot(ths, curve_df["youden_j"].values, label="Youden's J", lw=1.8, ls=":", color="#9467bd")

        ax1.axvline(0.5, color="gray", ls="--", lw=1.5, label="Fixed th=0.50")
        ax1.axvline(mean_th, color="#d62728", ls="-", lw=2.2, label=f"Mean Tuned th*={mean_th:.3f}")
        ax1.set_title("1. Metric Curves vs Cutoff Threshold")
        ax1.set_xlabel("Cutoff Threshold (th)")
        ax1.set_ylabel("Score")
        ax1.set_xlim(0.02, 0.98)
        ax1.set_ylim(-0.05, 1.05)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="lower left", fontsize=8.5)

        # ------------------------------------------------------------------
        # Panel 2: Fold別の最適閾値 th*_k の分布 & Fold別 Balanced Acc 比較
        # ------------------------------------------------------------------
        ax2 = axes[0, 1]
        folds = fold_df["fold"].values
        users = [str(u)[:7] for u in fold_df["test_user"].values]
        x = np.arange(len(folds))
        w = 0.36
        ax2.bar(x - w / 2, fold_df["balanced_acc_05"].values, width=w, label="Fixed th=0.50", color="#aec7e8", edgecolor="#1f77b4")
        ax2.bar(x + w / 2, fold_df["balanced_acc_tuned"].values, width=w, label=f"Tuned th*_k ({crit})", color="#ffbb78", edgecolor="#ff7f0e")

        ax2_twin = ax2.twinx()
        ax2_twin.plot(x, fold_df["optimal_threshold"].values, color="#d62728", marker="o", lw=1.8, label="Fold Optimal th*_k")
        ax2_twin.axhline(0.5, color="gray", ls=":", lw=1.2)
        ax2_twin.set_ylabel("Optimal Threshold th*_k", color="#d62728")
        ax2_twin.set_ylim(0.0, 1.0)

        ax2.set_title("2. Per-Subject Balanced Accuracy & Selected Threshold th*_k")
        ax2.set_xticks(x)
        ax2.set_xticklabels(users, rotation=45, ha="right", fontsize=8)
        ax2.set_ylabel("Balanced Accuracy")
        ax2.set_ylim(0.0, 1.05)
        ax2.grid(True, axis="y", alpha=0.3)
        ax2.legend(loc="upper left", fontsize=8.5)
        ax2_twin.legend(loc="upper right", fontsize=8.5)

        # ------------------------------------------------------------------
        # Panel 3: 全体 OOF 主要二値分類指標の比較 (Fixed 0.50 vs Tuned th*)
        # ------------------------------------------------------------------
        ax3 = axes[1, 0]
        metric_keys = ["balanced_acc", "f1_macro", "f1_neg", "specificity", "sensitivity", "accuracy"]
        metric_labels = ["Balanced\nAcc", "F1-Macro", "F1-Rest\n(Class 0)", "Specificity\n(Rest Recall)", "Sensitivity\n(Focus Recall)", "Accuracy"]
        vals_05 = [float(base_oof.get(k, 0.0)) for k in metric_keys]
        vals_tuned = [float(tuned_oof.get(k, 0.0)) for k in metric_keys]

        xm = np.arange(len(metric_keys))
        wm = 0.36
        rects1 = ax3.bar(xm - wm / 2, vals_05, width=wm, label="Fixed th=0.50", color="#7f7f7f", alpha=0.85)
        rects2 = ax3.bar(xm + wm / 2, vals_tuned, width=wm, label=f"Tuned th* ({crit})", color="#2ca02c", alpha=0.9)

        ax3.set_title("3. OOF Classification Metrics: Fixed 0.50 vs Leak-Free Tuned th*")
        ax3.set_xticks(xm)
        ax3.set_xticklabels(metric_labels, fontsize=9)
        ax3.set_ylim(0.0, 1.12)
        ax3.set_ylabel("Score")
        ax3.grid(True, axis="y", alpha=0.3)
        ax3.legend(loc="upper right", fontsize=9)

        for r in rects1:
            h = r.get_height()
            if np.isfinite(h):
                ax3.text(r.get_x() + r.get_width() / 2.0, h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=7.5, color="#444444")
        for r in rects2:
            h = r.get_height()
            if np.isfinite(h):
                ax3.text(
                    r.get_x() + r.get_width() / 2.0,
                    h + 0.015,
                    f"{h:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

        # ------------------------------------------------------------------
        # Panel 4: 正解クラス別の予測スコア分布ヒストグラム & カットオフ閾値
        # ------------------------------------------------------------------
        ax4 = axes[1, 1]
        y_bin = (np.asarray(y_true, dtype=float) >= 0.5).astype(int)
        preds = np.asarray(y_pred, dtype=float)
        p_neg = preds[y_bin == 0]
        p_pos = preds[y_bin == 1]

        bins = np.linspace(0.0, 1.0, 35)
        if len(p_neg) > 0:
            ax4.hist(p_neg, bins=bins, density=True, alpha=0.55, color="#ff7f0e", label=f"Rest (Class 0, n={len(p_neg)})")
        if len(p_pos) > 0:
            ax4.hist(p_pos, bins=bins, density=True, alpha=0.55, color="#1f77b4", label=f"Focus (Class 1, n={len(p_pos)})")

        ax4.axvline(0.5, color="gray", ls="--", lw=1.5, label="Fixed th=0.50")
        ax4.axvline(mean_th, color="#d62728", ls="-", lw=2.2, label=f"Mean Tuned th*={mean_th:.3f}")
        ax4.set_title("4. OOF Prediction Distribution & Cutoff Threshold")
        ax4.set_xlabel("Predicted Score (y_pred)")
        ax4.set_ylabel("Density")
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc="upper left", fontsize=8.5)

        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)

        if log:
            log.info(f"📊 閾値決定プロットを保存しました: {plot_path}")
        return plot_path
    except Exception as e:
        if log:
            log.warning(f"⚠️ 閾値決定プロット生成中にエラーが発生しました: {e}")
        return None


def save_single_threshold_report(
    output_dir: Path,
    window_size: str,
    model_tag: str,
    tuning_res: Dict,
    log=None,
) -> Path:
    """
    単一モデル × 単一基準の閾値最適化 Markdown レポートを保存します。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    crit = tuning_res["criterion"]
    crit_label = tuning_res.get("criterion_label", crit)
    report_path = output_dir / f"loso_{model_tag}_threshold_{crit}_report.md"

    base_oof = tuning_res["baseline_oof_detail"]
    tuned_oof = tuning_res["tuned_oof_detail"]
    base_st = tuning_res["baseline_fold_stats"]
    tuned_st = tuning_res["tuned_fold_stats"]
    imp = tuning_res["improvement"]
    fold_df: pd.DataFrame = tuning_res["fold_df"]
    all_ths: Dict = tuning_res.get("global_all_criteria_thresholds", {})

    lines = [
        f"# 🎯 閾値決定 (Threshold Decision) 評価レポート: `{model_tag}` [Window: {window_size}]",
        "",
        "## 1. 閾値最適化の設定（完全データリークフリー LOSO-CV）",
        f"- **対象モデル / ターゲット**: `{model_tag}`",
        f"- **採用基準**: `{crit}` — **{crit_label}**",
        "- **リーク防止設計**: 各 LOSO Fold において、**テスト被験者（1名）のデータは一切参照せず**、残り被験者（Train/Val）の予測スコアと正解ラベルのみから最適閾値 $th^*_k$ を算出し、テスト被験者の二値化判定へ適用。",
        f"- **決定された Fold 最適閾値 $th^*_k$**: 平均 **{tuning_res['mean_threshold']:.4f} ± {tuning_res['std_threshold']:.4f}** (最小 `{tuning_res['min_threshold']:.4f}` 〜 最大 `{tuning_res['max_threshold']:.4f}`)",
        "- **【参考】全体 OOF 曲線上の3大基準の理論最適点**: "
        + ", ".join([f"`{k}` = **{v:.4f}**" for k, v in all_ths.items()]),
        "",
        "## 2. 固定閾値 ($th = 0.50$) vs 最適化閾値 ($th^*_k$) の性能比較",
        "| 評価指標 | 固定閾値 $th=0.50$ (OOF) | **最適化閾値 $th^*_k$ (OOF)** | **改善幅 ($\\Delta$)** | Fold Mean ± Std ($th=0.50$) | **Fold Mean ± Std ($th^*_k$)** |",
        "|---|---|---|---|---|---|",
        f"| **Balanced Accuracy (均衡正解率)** | {base_oof['balanced_acc']:.4f} | **{tuned_oof['balanced_acc']:.4f}** | **{imp['balanced_acc']:+.4f}** | {base_st['balanced_acc'][0]:.4f} ± {base_st['balanced_acc'][1]:.4f} | **{tuned_st['balanced_acc'][0]:.4f} ± {tuned_st['balanced_acc'][1]:.4f}** |",
        f"| **Youden's J ($\\text{{Sens}} + \\text{{Spec}} - 1$)** | {base_oof['youden_j']:.4f} | **{tuned_oof['youden_j']:.4f}** | **{imp['youden_j']:+.4f}** | {base_st['youden_j'][0]:.4f} ± {base_st['youden_j'][1]:.4f} | **{tuned_st['youden_j'][0]:.4f} ± {tuned_st['youden_j'][1]:.4f}** |",
        f"| **F1-Macro (両クラス調和平均)** | {base_oof['f1_macro']:.4f} | **{tuned_oof['f1_macro']:.4f}** | **{imp['f1_macro']:+.4f}** | {base_st['f1_macro'][0]:.4f} ± {base_st['f1_macro'][1]:.4f} | **{tuned_st['f1_macro'][0]:.4f} ± {tuned_st['f1_macro'][1]:.4f}** |",
        f"| **F1-Neg (休憩クラス=0 の F1)** | {base_oof['f1_neg']:.4f} | **{tuned_oof['f1_neg']:.4f}** | **{imp['f1_neg']:+.4f}** | {base_st['f1_neg'][0]:.4f} ± {base_st['f1_neg'][1]:.4f} | **{tuned_st['f1_neg'][0]:.4f} ± {tuned_st['f1_neg'][1]:.4f}** |",
        f"| **F1-Pos (集中クラス=1 の F1)** | {base_oof['f1_pos']:.4f} | **{tuned_oof['f1_pos']:.4f}** | {tuned_oof['f1_pos'] - base_oof['f1_pos']:+.4f} | {base_st['f1_pos'][0]:.4f} ± {base_st['f1_pos'][1]:.4f} | {tuned_st['f1_pos'][0]:.4f} ± {tuned_st['f1_pos'][1]:.4f} |",
        f"| **Specificity (休憩の検出率 / TNR)** | {base_oof['specificity']:.4f} | **{tuned_oof['specificity']:.4f}** | **{imp['specificity']:+.4f}** | {base_st['specificity'][0]:.4f} ± {base_st['specificity'][1]:.4f} | **{tuned_st['specificity'][0]:.4f} ± {tuned_st['specificity'][1]:.4f}** |",
        f"| **Sensitivity (集中の検出率 / TPR)** | {base_oof['sensitivity']:.4f} | **{tuned_oof['sensitivity']:.4f}** | {imp['sensitivity']:+.4f} | {base_st['sensitivity'][0]:.4f} ± {base_st['sensitivity'][1]:.4f} | {tuned_st['sensitivity'][0]:.4f} ± {tuned_st['sensitivity'][1]:.4f} |",
        f"| **Accuracy (全体正解率)** | {base_oof['accuracy']:.4f} | **{tuned_oof['accuracy']:.4f}** | {imp['accuracy']:+.4f} | {base_st['accuracy'][0]:.4f} ± {base_st['accuracy'][1]:.4f} | {tuned_st['accuracy'][0]:.4f} ± {tuned_st['accuracy'][1]:.4f} |",
        "",
        "## 3. 混同行列 (Confusion Matrix) の変化",
        f"- **固定閾値 ($th = 0.50$)**: TN(休憩正解)=`{base_oof['tn']}`, FP(休憩→集中誤判定)=`{base_oof['fp']}`, FN(集中→休憩誤判定)=`{base_oof['fn']}`, TP(集中正解)=`{base_oof['tp']}`",
        f"- **最適閾値 ($th^*_k$)**: TN(休憩正解)=`{tuned_oof['tn']}`, FP(休憩→集中誤判定)=`{tuned_oof['fp']}`, FN(集中→休憩誤判定)=`{tuned_oof['fn']}`, TP(集中正解)=`{tuned_oof['tp']}`",
        "",
        "## 4. 被験者 (LOSO Fold) 別の決定閾値 $th^*_k$ とテスト評価詳細",
        "| Fold | Test User | $N_{\\text{test}}$ | **決定閾値 $th^*_k$** | Val Score | BalAcc (0.5) | **BalAcc ($th^*_k$)** | $\\Delta$ BalAcc | F1-Macro (0.5) | **F1-Macro ($th^*_k$)** | Spec (0.5 ➔ $th^*_k$) | Sens (0.5 ➔ $th^*_k$) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for _, r in fold_df.iterrows():
        lines.append(
            f"| {int(r['fold'])} | `{r['test_user']}` | {int(r['n_test'])} | "
            f"**{r['optimal_threshold']:.4f}** | {r['val_criterion_score']:.4f} | "
            f"{r['balanced_acc_05']:.4f} | **{r['balanced_acc_tuned']:.4f}** | {r['delta_balanced_acc']:+.4f} | "
            f"{r['f1_macro_05']:.4f} | **{r['f1_macro_tuned']:.4f}** | "
            f"{r['specificity_05']:.3f} ➔ **{r['specificity_tuned']:.3f}** | "
            f"{r['sensitivity_05']:.3f} ➔ {r['sensitivity_tuned']:.3f} |"
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if log:
        log.info(f"📝 閾値決定レポートを保存しました: {report_path}")
    return report_path


def save_threshold_decision_summary_report(
    output_dir: Path,
    window_size: str,
    default_criterion: str,
    comparison_df: pd.DataFrame,
    log=None,
) -> Path:
    """
    全対象モデル × 全基準の閾値最適化結果を一覧比較するマスターレポート (`report.md`) を生成します。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.md"

    lines = [
        f"# 🎯 Step 7-0: 学習前 閾値決定 (Pre-Learning Threshold Decision) 総合レポート [Window: {window_size}]",
        "",
        "## 1. 概要とリークフリー設計",
        "- **実行タイミング**: Base Model / Meta Model の**学習前**に実行し、決定した各 LOSO Fold の最適閾値 $th^*_k$ を後続の Base Model・Meta Model の二値判定・評価・推論に適用します。",
        "- **鉄則（完全データリークフリー）**: テスト被験者（Test）のデータは一切参照せず、各 LOSO Fold 内の学習データ（Train）の GMM ソフト確率ラベル `y_soft_train` と正解ラベル `label` のみから最適カットオフ閾値 $th^*_k \\in [0, 1]$ を決定しています。",
        f"- **採用基準 (`config.yaml`)**: `{default_criterion}` (**Balanced Accuracy 最大化 / Youden's J Index**)",
        "- **3大基準**:",
        "  1. `youden_j` (**Youden's J index / Balanced Accuracy 最大化**): $J(th) = \\text{Recall}(th) + \\text{Specificity}(th) - 1 = 2 \\times \\text{Balanced Accuracy}(th) - 1$",
        "  2. `f1_macro` (**F1-Macro 最大化**): 集中（多数派）と休憩（少数派）の両クラスの調和平均を最大化",
        "  3. `pr_balance` (**Precision-Recall バランス**): 適合率と再現率の差が最小（または積が最大）となる均衡点",
        "",
        "## 2. ターゲット × 基準別の性能比較一覧 (Fixed 0.50 vs Tuned $th^*_k$)",
        "| スコープ | モデル / ターゲット | 最適化基準 | Mean $th^* \\pm$ Std | BalAcc (0.5) | **BalAcc ($th^*$)** | $\\Delta$ BalAcc | F1-Macro (0.5) | **F1-Macro ($th^*$)** | $\\Delta$ F1-Macro | F1-Rest (0.5) | **F1-Rest ($th^*$)** | Accuracy ($th^*$) | ROC-AUC |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for _, row in comparison_df.iterrows():
        lines.append(
            f"| `{row['scope']}` | `{row['model']}` | `{row['criterion']}` | "
            f"`{row['mean_th']:.4f} ± {row['std_th']:.4f}` | "
            f"{row['balanced_acc_05']:.4f} | **{row['balanced_acc_tuned']:.4f}** | **{row['delta_balanced_acc']:+.4f}** | "
            f"{row['f1_macro_05']:.4f} | **{row['f1_macro_tuned']:.4f}** | **{row['delta_f1_macro']:+.4f}** | "
            f"{row['f1_neg_05']:.4f} | **{row['f1_neg_tuned']:.4f}** | "
            f"{row['accuracy_tuned']:.4f} | {row['roc_auc']:.4f} |"
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if log:
        log.info(f"🏆 閾値決定 総合比較レポートを保存しました: {report_path}")
    return report_path


__all__ = [
    "generate_threshold_visualizations",
    "save_single_threshold_report",
    "save_threshold_decision_summary_report",
]

