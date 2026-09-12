"""
Відтворюваність за кількістю партій — БЕЗ витоку даних.

Гравці набору з 16 партій здебільшого присутні й у навчальному 2k-наборі, тож
продакшн-модель їх бачила. Тут модель, норми внесків і популяційні статистики
еталона будуються на тих гравцях 2k-набору, яких у 16-партійному НЕМАЄ, а
вимірювання йде лише на 16-партійних. Жоден гравець не потрапляє в обидві ролі.

Для n = 2, 4, 8: 16 партій ділиться навпіл (8/8), з кожної половини беруться
перші n. Поруч із κ механізму рахується κ еталонного правила відхилень — стеля.
"""
import json, os, sys
from math import comb

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import cohen_kappa_score
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.features.extractor import (FEATURE_NAMES, TRAINABLE_CANDIDATES,
                                    FEATURE_ORIENTATION)

C = list(TRAINABLE_CANDIDATES)
SEEDS, RNG, NB = range(10), 42, 8


def band_ref(expl, agg):
    sv = expl.shap_values(agg)
    pred = np.asarray(expl.expected_value) + sv.sum(axis=1)
    sub = sv[:, [FEATURE_NAMES.index(c) for c in C]]
    e = np.quantile(pred, np.linspace(0, 1, NB + 1)); e[0], e[-1] = -np.inf, np.inf
    b = np.clip(np.searchsorted(e, pred, side="right") - 1, 0, NB - 1)
    f = [i for i in range(NB) if np.any(b == i)]
    return dict(centres=np.array([pred[b == i].mean() for i in f]),
                mean=np.vstack([sub[b == i].mean(0) for i in f]),
                std=np.vstack([sub[b == i].std(0) for i in f]))


def z_of(sub, pred, ref):
    x = ref["centres"]
    mu = np.column_stack([np.interp(pred, x, ref["mean"][:, j]) for j in range(sub.shape[1])])
    sd = np.column_stack([np.interp(pred, x, ref["std"][:, j])  for j in range(sub.shape[1])])
    sd = np.where(sd > 0, sd, np.nan)
    z = (sub - mu) / sd
    return np.where(np.isfinite(z), z, np.inf)


def main():

    te = pd.read_parquet("data/dataset_blitz_16g.parquet")
    te = te[te["target_elo"].between(400, 3000)]
    keep = te.groupby("player").size(); keep = keep[keep >= 16].index
    te = te[te["player"].isin(keep)].reset_index(drop=True)
    eval_players = set(te["player"].unique())

    tr = pd.read_parquet("data/dataset_blitz_2k.parquet")
    tr = tr[tr["target_elo"].between(400, 3000)]
    before = tr["player"].nunique()
    tr = tr[~tr["player"].isin(eval_players)].reset_index(drop=True)
    print(f"оцінюваних гравців (16 партій): {len(eval_players)}")
    print(f"навчальних: {tr['player'].nunique()} із {before} "
          f"({before - tr['player'].nunique()} виключено як спільні)")
    assert not (set(tr['player']) & eval_players), "перетин лишився"

    Xtr = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(tr[FEATURE_NAMES]),
                       columns=FEATURE_NAMES)
    m = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=RNG)
    m.fit(Xtr, tr["target_elo"])
    import shap
    expl = shap.TreeExplainer(m)
    ci = [FEATURE_NAMES.index(c) for c in C]

    ref = band_ref(expl, Xtr.groupby(tr["player"].values).mean()[FEATURE_NAMES])
    tr_pm = Xtr[C].groupby(tr["player"].values).mean()
    mu_p, sd_p = tr_pm.mean(), tr_pm.std(ddof=0)
    orient = pd.Series(FEATURE_ORIENTATION)[C]
    print("модель, норми внесків і популяція еталона — лише на навчальних\n")

    Xte = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(te[FEATURE_NAMES]),
                       columns=FEATURE_NAMES)
    splits = []
    for p, g in te.groupby("player"):
        idx = g.index.to_numpy()
        for s in SEEDS:
            sh = np.random.RandomState(s).permutation(idx)[:16]
            splits.append((sh[:8], sh[8:]))
    print(f"розбиттів: {len(splits)}")

    NS = (2, 4, 8)
    vecs, spans = [], {}
    for n in NS:
        st = len(vecs)
        for a, b in splits:
            vecs.append(Xte.loc[a[:n]].mean().values)
            vecs.append(Xte.loc[b[:n]].mean().values)
        spans[n] = (st, len(vecs))
    V = pd.DataFrame(vecs, columns=FEATURE_NAMES)
    print(f"векторів на пояснення: {len(V)} ...")
    sv = expl.shap_values(V)
    pr = np.asarray(expl.expected_value) + sv.sum(axis=1)
    Z = z_of(sv[:, ci], pr, ref)
    print("SHAP готово\n")

    def dev(rows):
        return ((Xte.loc[rows, C].mean() - mu_p) / sd_p).mul(orient).idxmax()

    out = {"n_players": len(eval_players), "n_splits": len(splits),
           "n_train_players": int(tr["player"].nunique()),
           "chance_ov3": 1 - comb(len(C)-3, 3)/comb(len(C), 3)}
    for n in NS:
        s, e = spans[n]
        R = np.array(C)[Z[s:e].argsort(1)]
        A, B = R[0::2], R[1::2]
        ga = [dev(a[:n]) for a, _ in splits]
        gb = [dev(b[:n]) for _, b in splits]
        out[str(n)] = dict(
            kappa=float(cohen_kappa_score(A[:, 0], B[:, 0], labels=C)),
            agreement=float(np.mean(A[:, 0] == B[:, 0])),
            ov3=float(np.mean([len(set(x[:3]) & set(y[:3])) > 0 for x, y in zip(A, B)])),
            ceiling_kappa=float(cohen_kappa_score(ga, gb, labels=C)))
        v = out[str(n)]
        print(f"  n={n}:  κ={v['kappa']:.4f}   стеля={v['ceiling_kappa']:.4f}   "
              f"перетин@3={v['ov3']:.3f}")
    json.dump(out, open("results/reliability_vs_n_oof.json", "w"), indent=1)
    print("\n->  results/reliability_vs_n_oof.json")


if __name__ == "__main__":
    main()
