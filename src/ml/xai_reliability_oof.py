"""
Відтворення слабкості гравця за N партіями (таблиця 3.10, рисунок 3.9) без витоку
даних за гравцями.

xai_deep.reliability_vs_n рахує обидва стовпці таблиці на моделі, навченій на
всіх гравцях (xai_deep.fit_shap), і з популяційними статистиками правила
відхилень за всією вибіркою, тобто й за гравцем, якого оцінюють. Тут той самий
розрахунок за GroupKFold(5) за гравцями: у кожному блоці імпутація медіанами,
змішана модель (ті самі параметри) і середні та стандартні відхилення ознак
будуються лише на навчальних гравцях, а внески SHAP і обидва визначення
слабкості рахуються для відкладених. Решта (розбиття партій, вибір за більшістю,
перелік кандидатів, зерна) – як у xai_deep.

Контроль: з моделлю на всіх гравцях і статистиками за всією вибіркою той самий
код мусить відтворити results/xai_reliability_vs_n.csv і рисунок 3.9.

Usage:
    python -m src.ml.xai_reliability_oof --outdir results/noleak
"""
import argparse, json, os, sys

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.features.extractor import FEATURE_NAMES, WEAKNESS_CANDIDATES, FEATURE_ORIENTATION
from src.ml.mixed_training import mixed_training_set
from src.ml.xai_deep import RNG, fit_shap, modal

NS, SEEDS = (1, 2, 3, 4, 5, 6), range(8)          # як у xai_deep.reliability_vs_n


def shap_weak(model, Xi):
    """Базове правило на кожній партії: ознака з найменшим внеском (як у xai_deep.main)."""
    sv = shap.TreeExplainer(model).shap_values(Xi)
    idx = [FEATURE_NAMES.index(c) for c in WEAKNESS_CANDIDATES]
    return [WEAKNESS_CANDIDATES[i] for i in np.argmin(sv[:, idx], axis=1)]


def agreements(df, stats):
    """Цикл xai_deep.reliability_vs_n; stats[player] = (pm, ps) популяції, проти якої
    нормується саме цей гравець."""
    cand = WEAKNESS_CANDIDATES
    orient = pd.Series(FEATURE_ORIENTATION)[cand]
    per_player = df.groupby("player").size()
    usable_ns = [n for n in NS if (per_player >= 2 * n).sum() >= 30]
    rows = []
    for n in usable_ns:
        a_single, a_agg = [], []
        for seed in SEEDS:
            for p, g in df.groupby("player"):
                if len(g) < 2 * n:
                    continue
                pm, ps = stats[p]
                sh = g.sample(frac=1.0, random_state=seed)
                half = len(sh) // 2
                A, B = sh.iloc[:half], sh.iloc[half:half + n]
                ref = (((A[cand].mean() - pm) / ps) * orient).idxmax()
                a_single.append(ref == modal(B["shap_weak"]))
                a_agg.append(ref == (((B[cand].mean() - pm) / ps) * orient).idxmax())
        rows.append({"n_games": n, "agreement_single_shap": float(np.mean(a_single)),
                     "agreement_aggregated_z": float(np.mean(a_agg)), "n_pairs": len(a_single)})
    return pd.DataFrame(rows)


def fig_39(tab, outdir):
    """Рисунок 3.9 тим самим кодом, що й у figures_ua.py."""
    from src.ml import figures_ua as F
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(tab["n_games"], tab["agreement_single_shap"], "o-", label="SHAP за окремими партіями (вибір за більшістю)")
    ax.plot(tab["n_games"], tab["agreement_aggregated_z"], "s--", label="Правило відхилень на тих самих партіях (верхня межа)")
    ax.axhline(1 / 29, color="grey", ls=":", label="Випадковий рівень")
    ax.set_xlabel("Кількість партій для повторного визначення слабкості, N")
    ax.set_ylabel("Частка збігів із цільовою ознакою")
    ax.set_xticks([1, 2, 3, 4]); ax.yaxis.set_major_formatter(F.CommaFormatter())
    ax.legend(loc="center right"); fig.tight_layout()
    problems = []
    F.save(fig, outdir, "fig_3_9", problems)
    assert not problems, problems
    return os.path.join(outdir, "fig_3_9.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/dataset_blitz_2k.parquet")
    ap.add_argument("--outdir", default="results/noleak")
    args = ap.parse_args()
    ctl = os.path.join(args.outdir, "control"); os.makedirs(ctl, exist_ok=True)
    df = pd.read_parquet(args.parquet)
    df = df[df["target_elo"].between(400, 3000)].reset_index(drop=True)
    X, y, players = df[FEATURE_NAMES], df["target_elo"], df["player"]
    cand = WEAKNESS_CANDIDATES
    out = {"protocol": "GroupKFold(5) за гравцями; імпутація, змішана модель і популяційні статистики "
                       "правила відхилень – лише на навчальних гравцях блоку", "n_players": int(players.nunique())}

    # ---- контроль: модель на всіх гравцях, статистики за всією вибіркою = збережена таблиця ----
    _, sv = fit_shap(X, y, players)
    idx = [FEATURE_NAMES.index(c) for c in cand]
    d0 = df.copy(); d0["shap_weak"] = [cand[i] for i in np.argmin(sv[:, idx], axis=1)]
    pm, ps = df[cand].mean(), df[cand].std(ddof=0).replace(0, 1)
    tab0 = agreements(d0, {p: (pm, ps) for p in df["player"].unique()})
    old = pd.read_csv("results/xai_reliability_vs_n.csv")
    out["control_max_abs_diff"] = float(np.abs(tab0[old.columns].to_numpy() - old.to_numpy()).max())
    out["control_fig_3_9_same_pixels"] = _same_pixels(fig_39(tab0, ctl), "results/fig_ua/fig_3_9.png")
    print(f"контроль: найбільша різниця зі збереженою таблицею {out['control_max_abs_diff']:.3g}; "
          f"рисунок 3.9 попіксельно: {out['control_fig_3_9_same_pixels']}")

    # ---- без витоку: усе, крім вимірювання, – на навчальних гравцях блоку ----
    d1 = df.copy(); d1["shap_weak"] = None; stats = {}
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, players), 1):
        imp = SimpleImputer(strategy="median").fit(X.iloc[tr])
        Xtr = pd.DataFrame(imp.transform(X.iloc[tr]), columns=FEATURE_NAMES)
        Xte = pd.DataFrame(imp.transform(X.iloc[te]), columns=FEATURE_NAMES)
        frame = Xtr.copy(); frame["target_elo"] = y.iloc[tr].values; frame["player"] = players.iloc[tr].values
        Xm, ym, _, _ = mixed_training_set(frame)
        model = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, random_state=RNG).fit(Xm, ym)
        d1.loc[d1.index[te], "shap_weak"] = shap_weak(model, Xte)
        tr_df = df.iloc[tr]
        pm_tr, ps_tr = tr_df[cand].mean(), tr_df[cand].std(ddof=0).replace(0, 1)
        for p in players.iloc[te].unique():
            stats[p] = (pm_tr, ps_tr)
        print(f"  блок {fold}: {players.iloc[te].nunique()} відкладених гравців")
    assert d1["shap_weak"].notna().all() and len(stats) == players.nunique()
    tab1 = agreements(d1, stats)
    tab1.to_csv(os.path.join(args.outdir, "xai_reliability_vs_n_oof.csv"), index=False)
    fig_39(tab1, args.outdir)
    out["in_sample"] = tab0.to_dict("records"); out["oof"] = tab1.to_dict("records")
    json.dump(out, open(os.path.join(args.outdir, "xai_reliability_vs_n_oof.json"), "w"), ensure_ascii=False, indent=1)
    print(pd.concat([tab0.add_suffix("_in"), tab1.add_suffix("_oof")], axis=1).round(4).to_string(index=False))
    print(f"-> {args.outdir}/xai_reliability_vs_n_oof.csv, fig_3_9.png")


def _same_pixels(a, b):
    from PIL import Image
    x, y = np.asarray(Image.open(a).convert("RGBA")), np.asarray(Image.open(b).convert("RGBA"))
    return x.shape == y.shape and bool((x == y).all())


if __name__ == "__main__":
    main()
