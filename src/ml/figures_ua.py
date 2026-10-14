"""
Українські версії рисунків 3.1–3.13 магістерської, намальовані зі збережених
результатів. Загальних заголовків на рисунках немає, бо назву дає підпис під
рисунком. Назви метрик (R², MAE, SHAP) і моделей лишаються латиницею.

Оригінальні скрипти (evaluate.py, aggregation_curve.py, learning_curve.py,
xai_deep.py, validate_xai.py, synthetic_validation.py, calibration.py,
external_validation.py) тут не запускаються, бо вони перенавчають моделі й
перезаписують results/. Звідси беруться лише дані:
  3.1  results/model_comparison.csv
  3.2  results/aggregation_mixed.json, results/aggregation_curve.json
  3.3а results/pool_summary.csv
  3.3б results/aggregation_mixed_by_pool.json
  3.4  results/learning_curve.json
  3.5  results/feature_importance.csv
  3.6  кореляції Пірсона за data/dataset_blitz_2k.parquet (як у evaluate.py)
  3.7  внески окремих партій не збережено, тож модель перенавчається тим самим
       детермінованим xai_deep.fit_shap() (≈2 с)
  3.8  results/rules/xai_groundtruth_vs_pred_blitz_raw_oof.csv (протокол без
       витоку, src/ml/oof_pairs_raw.py)
  3.9  results/noleak/xai_reliability_vs_n_oof.csv (без витоку, src/ml/xai_reliability_oof.py)
  3.10 results/noleak/synthetic_validation_bands.json (модель і норми без цільових гравців,
       src/ml/synthetic_noleak.py)
  3.11 results/calibration_by_band.csv (calibration.py архівний, вхідного набору немає)
  3.12 results/{bullet,blitz,rapid}/external_validation_bias.csv
  3.13 results/blitz/external_validation_bias.csv; позаблоковий ізотонічний
       прогноз відтворюється тим самим KFold(5, shuffle, 42), що в external_validation.py

Usage:
    python -m src.ml.figures_ua --outdir results/fig_ua
"""
import argparse, json, os, re, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.features.extractor import FEATURE_NAMES
from src.ml import plotstyle

plotstyle.apply()
PRIMARY, SECONDARY, TERTIARY = plotstyle.PRIMARY, plotstyle.SECONDARY, plotstyle.TERTIARY
MINUS = "−"

# Назви ознак так, як їх ужито в тексті роботи (табл. 2.1, 3.11–3.15).
FULL = {
    "acpl": "Загальна ACPL", "inaccuracy_rate": "Частка неточностей", "mistake_rate": "Частка помилок",
    "blunder_rate": "Частка грубих помилок", "opening_acpl": "Точність у дебюті",
    "middlegame_acpl": "Точність у міттельшпілі", "endgame_acpl": "Точність в ендшпілі",
    "conversion_acpl": "Реалізація переваги", "defense_acpl": "Точність у захисті",
    "engine_match_rate": "Частка найкращих ходів (CPL = 0)", "capture_rate": "Частка взяттів",
    "check_given_rate": "Частка шахів", "hanging_rate": "Висячі фігури", "king_danger": "Небезпека для короля",
    "mobility_avg": "Рухливість фігур", "center_control_avg": "Контроль центру",
    "material_volatility": "Волатильність матеріалу", "pin_exposure": "Схильність до зв’язок",
    "think_time_consistency": "Стабільність часу", "avg_think_time": "Середній час на хід",
    "time_pressure_rate": "Частка ходів у цейтноті (< 15 с)",
    "wp_loss_mean": "Середня втрата ймовірності виграшу за хід",
    "wp_blunder_rate": "Частка ходів із втратою ймовірності ≥ 0,10",
    "cpl_std": "Стандартне відхилення втрати сантипішаків",
    "top1_agree": "Збіг із першим кандидатом рушія", "top3_agree": "Збіг із трійкою кандидатів рушія",
    "avg_move_rank": "Середній ранг зіграного ходу", "acpl_critical": "Точність у критичних позиціях",
    "critical_agree": "Збіг із найкращим ходом у критичних позиціях",
}
# Коротші форми лише для чотирьох найдовших назв і лише на матрицях 29 × 29.
SHORT = dict(FULL, wp_loss_mean="Втрата ймовірності виграшу", wp_blunder_rate="Частка ходів із втратою ≥ 0,10",
             cpl_std="Стандартне відхилення CPL", critical_agree="Збіг у критичних позиціях")
# Категорії синтетичної перевірки (табл. 3.11–3.14).
CATEGORY = dict(FULL, blunder_rate="Грубі помилки")
POOL = {"bullet": "Куля", "blitz": "Бліц", "rapid": "Рапід", "classical": "Класика"}
MODEL = {"Baseline (mean)": "Baseline\n(середнє)", "Ridge": "Ridge\n(гребенева)", "RandomForest": "Random\nForest",
         "MLP": "MLP\n(перцептрон)", "XGBoost": "XGBoost"}
X_N = "Кількість агрегованих партій гравця, N"
Y_MAE = "MAE, пунктів Elo"


class CommaFormatter(mticker.ScalarFormatter):
    """Десяткова кома; мінус U+2212 matplotlib ставить сам."""
    def __call__(self, x, pos=None):
        return super().__call__(x, pos).replace(".", ",")

    def format_ticks(self, values):
        return [s.replace(".", ",") for s in super().format_ticks(values)]


def thousands(x, pos=None):
    return f"{int(round(x)):,}".replace(",", " ").replace("-", MINUS)


def heat_text(ax, mat, fontsize, hide_zeros=False, white_above=None):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if hide_zeros and v == 0:
                continue
            dark = white_above is not None and v > white_above
            ax.text(j, i, v, ha="center", va="center", fontsize=fontsize, color="white" if dark else "black")


def strings_of(fig):
    fig.canvas.draw()
    out = [t.get_text() for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text()]
    return list(dict.fromkeys(out))


ALLOWED_LATIN = ("R²", "MAE", "SHAP", "ACPL", "CPL", "Elo", "XGBoost", "MLP", "Ridge", "Random", "Forest",
                 "Baseline", "Lichess", "Chess.com", "N")


def check(name, fig):
    """Жодного англійського слова, десяткової крапки, дефіса-мінуса чи підкреслення."""
    bad = []
    for s in strings_of(fig):
        rest = s
        for tok in ALLOWED_LATIN:
            rest = rest.replace(tok, "")
        if re.search(r"[A-Za-z]", rest):
            bad.append(("латиниця", s))
        if re.search(r"\d\.\d", s.replace("Chess.com", "")):
            bad.append(("десяткова крапка", s))
        if re.search(r"(^|[\s(])-\d", s):
            bad.append(("дефіс-мінус", s))
        if "_" in s:
            bad.append(("підкреслення", s))
    print(f"  {name}: {len(strings_of(fig))} написів" + ("" if not bad else f"; ПРОБЛЕМИ {bad}"))
    return bad


def save(fig, outdir, name, problems):
    problems += check(name, fig)
    fig.savefig(os.path.join(outdir, f"{name}.png"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results/fig_ua")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    R = "results"; out = args.outdir; problems = []

    # ── 3.1 Порівняння моделей за MAE ──
    comp = pd.read_csv(f"{R}/model_comparison.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([MODEL[m] for m in comp["model"]], comp["MAE"], yerr=comp["MAE_std"], color="#55A868", capsize=4)
    ax.set_ylabel(Y_MAE)
    fig.tight_layout(); save(fig, out, "fig_3_1", problems)

    # ── 3.2 Точність прогнозу залежно від кількості агрегованих партій ──
    mixed = json.load(open(f"{R}/aggregation_mixed.json"))
    ceil = {c["n_games"]: c for c in json.load(open(f"{R}/aggregation_curve.json"))}
    ns = [c["n_games"] for c in mixed]; cn = [n for n in ns if n in ceil]
    LA = "(а) окрема модель на серіях фіксованого N"
    LB = "(б) модель на одиничних партіях з усередненим вектором на вході"
    LC = "(в) змішане навчання (конфігурація застосунку)"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.6), layout="constrained")
    ax1.plot(cn, [ceil[n]["r2"] for n in cn], "^:", color="grey", label=LA)
    ax2.plot(cn, [ceil[n]["mae"] for n in cn], "^:", color="grey", label=LA)
    ax1.errorbar(ns, [c["r2_single"] for c in mixed], yerr=[c["r2_single_std"] for c in mixed],
                 marker="s", capsize=3, color="#d62728", ls="--", label=LB)
    ax1.errorbar(ns, [c["r2_mixed"] for c in mixed], yerr=[c["r2_mixed_std"] for c in mixed],
                 marker="o", capsize=3, color="#1f77b4", label=LC)
    ax2.plot(ns, [c["mae_single"] for c in mixed], "s--", color="#d62728", label=LB)
    ax2.plot(ns, [c["mae_mixed"] for c in mixed], "o-", color="#1f77b4", label=LC)
    ax1.set_xlabel(X_N); ax1.set_ylabel("R²"); ax2.set_xlabel(X_N); ax2.set_ylabel(Y_MAE)
    ax1.yaxis.set_major_formatter(CommaFormatter())
    h, l = ax1.get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=1, frameon=False)
    save(fig, out, "fig_3_2", problems)

    # ── 3.3а Точність за пулами, окрема партія ──
    tab = pd.read_csv(f"{R}/pool_summary.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    ax1.bar([POOL[p] for p in tab["pool"]], tab["MAE"], color="#C44E52"); ax1.set_ylabel(Y_MAE)
    ax2.bar([POOL[p] for p in tab["pool"]], tab["R2"], color="#4C72B0"); ax2.set_ylabel("R²")
    ax2.yaxis.set_major_formatter(CommaFormatter())
    fig.tight_layout(); save(fig, out, "fig_3_3a", problems)

    # ── 3.3б Криві агрегування за пулами ──
    curves = json.load(open(f"{R}/aggregation_mixed_by_pool.json"))
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for pool, curve in curves.items():
        ax.plot([c["n_games"] for c in curve], [c["r2_mixed"] for c in curve], "o-", label=POOL[pool])
    ax.set_xlabel(X_N); ax.set_ylabel("R²"); ax.legend()
    ax.yaxis.set_major_formatter(CommaFormatter())
    fig.tight_layout(); save(fig, out, "fig_3_3b", problems)

    # ── 3.4 Крива навчання ──
    lc = json.load(open(f"{R}/learning_curve.json"))["mixed"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    for ax, key, ylabel, comma in ((ax1, "r2", "R²", True), (ax2, "mae", Y_MAE, False)):
        ax.errorbar([r["n_players"] for r in lc], [r[key] for r in lc], yerr=[r[key + "_std"] for r in lc],
                    marker="o", capsize=3)
        ax.set_xlabel("Кількість гравців"); ax.set_ylabel(ylabel)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(thousands))
        if comma:
            ax.yaxis.set_major_formatter(CommaFormatter())
    fig.tight_layout(); save(fig, out, "fig_3_4", problems)

    # ── 3.5 Глобальна важливість ознак (SHAP) ──
    imp = pd.read_csv(f"{R}/feature_importance.csv").sort_values("shap_mean_abs")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([FULL[f] for f in imp["feature"]], imp["shap_mean_abs"], color="#4C72B0")
    ax.set_xlabel("Середній модуль внеску SHAP, пунктів Elo")
    fig.tight_layout(); save(fig, out, "fig_3_5", problems)

    # ── 3.6 Матриця кореляцій ознак (той самий X.corr(), що в evaluate.py) ──
    df = pd.read_parquet("data/dataset_blitz_2k.parquet")
    df = df[df["target_elo"].between(400, 3000)].reset_index(drop=True)
    corr = df[FEATURE_NAMES].corr()
    n = len(FEATURE_NAMES)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n)); ax.grid(False)
    ax.set_xticklabels([str(i + 1) for i in range(n)], fontsize=7)
    ax.set_yticklabels([f"{SHORT[f]}  {i + 1}" for i, f in enumerate(FEATURE_NAMES)], fontsize=8)
    cb = fig.colorbar(im, fraction=0.046); cb.ax.yaxis.set_major_formatter(CommaFormatter())
    fig.tight_layout(); save(fig, out, "fig_3_6", problems)

    # ── 3.7 Графіки залежності SHAP (xai_deep.fit_shap, детерміновано) ──
    from src.ml import xai_deep
    Xi, sv = xai_deep.fit_shap(df[FEATURE_NAMES], df["target_elo"], df["player"])
    feats = {"opening_acpl": "Точність у дебюті (ACPL), сантипішаків", "avg_think_time": "Середній час на хід, с",
             "hanging_rate": "Висячі фігури на хід"}
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), layout="constrained")
    for k, (ax, (f, xlabel)) in enumerate(zip(axes, feats.items())):
        j = FEATURE_NAMES.index(f)
        sc = ax.scatter(Xi[f], sv[:, j], c=df["target_elo"].values, cmap="viridis", s=8, alpha=0.5)
        ax.axhline(0, color="grey", lw=0.7)
        ax.set_xlabel(xlabel)
        if k == 0:
            ax.set_ylabel("Внесок SHAP, пунктів Elo")
        ax.xaxis.set_major_formatter(CommaFormatter()); ax.yaxis.set_major_formatter(CommaFormatter())
    fig.colorbar(sc, ax=axes, label="Рейтинг Elo", fraction=0.025)
    save(fig, out, "fig_3_7", problems)

    # ── 3.8 Слабкість за відхиленнями проти базового правила, без витоку ──
    pairs = pd.read_csv(f"{R}/rules/xai_groundtruth_vs_pred_blitz_raw_oof.csv")
    conf = pd.crosstab(pairs["gt_weak"], pairs["pred_weak"]).reindex(index=FEATURE_NAMES, columns=FEATURE_NAMES, fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 6.2))
    ax.imshow(conf.values, cmap="Blues"); ax.grid(False)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([str(i + 1) for i in range(n)], fontsize=6.5)
    ax.set_yticklabels([f"{SHORT[f]}  {i + 1}" for i, f in enumerate(FEATURE_NAMES)], fontsize=7)
    heat_text(ax, conf.values, 6, hide_zeros=True, white_above=0.6 * conf.values.max())
    ax.set_xlabel("Слабкість за базовим правилом"); ax.set_ylabel("Слабкість за відхиленнями")
    fig.colorbar(ax.images[0], fraction=0.046)
    fig.tight_layout(); save(fig, out, "fig_3_8", problems)
    print(f"     3.8: гравців {int(conf.values.sum())}, непорожніх стовпців {int((conf.sum(0) > 0).sum())}, "
          f"діагональ {np.trace(conf.values) / conf.values.sum():.3f}")

    # ── 3.9 Відтворення слабкості гравця за N партіями ──
    rel = pd.read_csv(f"{R}/noleak/xai_reliability_vs_n_oof.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(rel["n_games"], rel["agreement_single_shap"], "o-", label="SHAP за окремими партіями (вибір за більшістю)")
    ax.plot(rel["n_games"], rel["agreement_aggregated_z"], "s--", label="Правило відхилень на тих самих партіях (верхня межа)")
    ax.axhline(1 / 29, color="grey", ls=":", label="Випадковий рівень")
    ax.set_xlabel("Кількість партій для повторного визначення слабкості, N")
    ax.set_ylabel("Частка збігів із цільовою ознакою")
    ax.set_xticks([1, 2, 3, 4]); ax.yaxis.set_major_formatter(CommaFormatter())
    ax.legend(loc="center right"); fig.tight_layout(); save(fig, out, "fig_3_9", problems)

    # ── 3.10 Внесена слабкість × виявлена слабкість ──
    syn = json.load(open(f"{R}/noleak/synthetic_validation_bands.json"))
    cats_all = ["opening_acpl", "middlegame_acpl", "endgame_acpl", "conversion_acpl", "defense_acpl", "blunder_rate", "hanging_rate"]
    cats = syn["summary"]["injection"]["categories"]
    mat = np.array([[syn["confusion"][g][p] for p in cats_all] for g in cats])
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(mat, cmap="Greens"); ax.grid(False)
    ax.set_xticks(range(len(cats_all))); ax.set_yticks(range(len(cats)))
    ax.set_xticklabels([CATEGORY[c] for c in cats_all], rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels([CATEGORY[c] for c in cats], fontsize=9)
    heat_text(ax, mat, 8, white_above=0.6 * mat.max())
    ax.set_xlabel("Виявлена слабкість (базове правило)"); ax.set_ylabel("Внесена слабкість")
    fig.colorbar(im, fraction=0.046); fig.tight_layout(); save(fig, out, "fig_3_10", problems)

    # ── 3.11 Криві калібрування для трьох варіантів набору ──
    cal = pd.read_csv(f"{R}/calibration_by_band.csv")
    act, strat, iso = (cal[cal.dataset == k].reset_index(drop=True)
                       for k in ("legacy (skewed sample)", "stratified (400-2800)", "stratified + isotonic"))
    allv = np.concatenate([cal.actual_mean.values, cal.predicted_mean.values])
    lims = [allv.min() - 50, allv.max() + 50]
    XL, YL = "Фактичний рейтинг Elo (середнє за інтервалом)", "Прогнозований рейтинг Elo (середнє за інтервалом)"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    for ax, series, title in ((ax1, [(act, "Відбір за активністю", "#C44E52", "o"), (strat, "Стратифікований (без калібрування)", "#4C72B0", "s")],
                               "Стратегії відбору вибірки"),
                              (ax2, [(strat, "Стратифікований (без калібрування)", "#4C72B0", "s"), (iso, "Стратифікований + ізотонічний", "#55A868", "^")],
                               "Ізотонічне калібрування")):
        for t, label, color, marker in series:
            ax.plot(t["actual_mean"], t["predicted_mean"], marker=marker, color=color, label=label)
        ax.plot(lims, lims, "k--", lw=1, alpha=0.5, label="Ідеальне калібрування")
        ax.set_xlabel(XL); ax.set_ylabel(YL); ax.set_title(title)
        ax.legend(fontsize=8); ax.set_xlim(lims); ax.set_ylim(lims)
    fig.tight_layout(); save(fig, out, "fig_3_11", problems)

    # ── 3.12 Зміщення прогнозу залежно від рейтингу (як plot_bias_three_pools.py) ──
    def fit(d):
        yb = d["bias"].to_numpy(float); c = (d["platform"] != "lichess").to_numpy(float); e = d["actual_elo"].to_numpy(float)
        beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(yb)), c, e, c * e]), yb, rcond=None)
        return (lambda cc, ee: beta[0] + beta[1] * cc + beta[2] * ee + beta[3] * cc * ee), (lambda ee: beta[1] + beta[3] * ee)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)
    for ax, pool in zip(axes, ("bullet", "blitz", "rapid")):
        d = pd.read_csv(f"{R}/{pool}/external_validation_bias.csv")
        line, gap = fit(d)
        for plat, colour, label in (("lichess", PRIMARY, "Контрольна група Lichess"), ("chess.com", SECONDARY, "Вибірка Chess.com")):
            sub = d[d["platform"] == plat]
            ax.scatter(sub["actual_elo"], sub["bias"], s=14, alpha=0.45, color=colour, edgecolors="none", label=label)
        xs = np.linspace(d["actual_elo"].min(), d["actual_elo"].max(), 200)
        ax.plot(xs, line(0.0, xs), color=PRIMARY, lw=1.8); ax.plot(xs, line(1.0, xs), color=SECONDARY, lw=1.8)
        for e in (800, 2200):
            lo, hi = line(0.0, e), line(1.0, e)
            ax.annotate("", xy=(e, hi), xytext=(e, lo), arrowprops=dict(arrowstyle="|-|,widthA=0.4,widthB=0.4", color="black", lw=1.2))
            ax.text(e + 60, (lo + hi) / 2, f"{gap(e):+.0f}".replace("-", MINUS), va="center", ha="left", fontsize=9, fontweight="bold")
        ax.axhline(0, color="grey", ls="--", lw=0.9, alpha=0.7)
        ax.set_title(POOL[pool]); ax.set_xlabel("Фактичний рейтинг Elo")
    axes[0].set_ylabel("Зміщення (прогноз − факт)")
    h, l = axes[0].get_legend_handles_labels()
    h.append(Line2D([], [], color="grey", lw=1.8)); l.append("Лінії регресії та розрив між ними")
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.09))
    save(fig, out, "fig_3_12", problems)

    # ── 3.13 Прогноз до і після рекалібрування, пул бліц ──
    d = pd.read_csv(f"{R}/blitz/external_validation_bias.csv")
    cc = d[d.platform == "chess.com"].reset_index(drop=True)
    raw_pred, actual = cc["predicted_elo"].to_numpy(float), cc["actual_elo"].to_numpy(float)
    iso_pred = np.full(len(cc), np.nan)
    for tr, te in KFold(n_splits=5, shuffle=True, random_state=42).split(raw_pred):
        iso_pred[te] = IsotonicRegression(out_of_bounds="clip").fit(raw_pred[tr], actual[tr]).predict(raw_pred[te])
    summ = json.load(open(f"{R}/blitz/external_validation_summary.json"))
    print(f"     3.13: MAE після ізотонічного {np.abs(iso_pred - actual).mean():.3f}, "
          f"у збереженому підсумку {json.dumps(summ.get('part_b', {}), ensure_ascii=False)[:160]}")
    lims = [cc.actual_elo.min() - 100, cc.predicted_elo.max() + 100]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    for ax, yv, colour, ylabel, title in ((ax1, cc.predicted_elo, SECONDARY, "Прогноз у шкалі Lichess", "До рекалібрування"),
                                          (ax2, iso_pred, TERTIARY, "Прогноз у шкалі Chess.com", "Після ізотонічного рекалібрування")):
        ax.scatter(cc.actual_elo, yv, color=colour, s=40)
        ax.plot(lims, lims, "k--", lw=1, alpha=0.5)
        ax.set_xlabel("Фактичний рейтинг Chess.com"); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.set_xlim(lims); ax.set_ylim(lims)
    fig.tight_layout(); save(fig, out, "fig_3_13", problems)

    print(f"проблем у написах: {len(problems)}")
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
