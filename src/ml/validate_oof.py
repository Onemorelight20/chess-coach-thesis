"""
Валідація механізму БЕЗ витоку даних за гравцями.

validate_xai.py навчає модель на всіх даних і на них же міряє тест-ретест —
модель бачила тих самих гравців, тож відтворюваність може бути завищена.
Тут той самий протокол, що для точності рейтингу: GroupKFold(5) за гравцями.

Для кожного блоку:
  модель, норми внесків і популяційні статистики еталона будуються ЛИШЕ на
  навчальних гравцях, а κ тест-ретесту і збіг рахуються на відкладених.
Результати з п'яти блоків зводяться разом (out-of-fold).
"""
import json, os, sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GroupKFold
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

    df = pd.read_parquet("data/dataset_blitz_2k.parquet")
    df = df[df["target_elo"].between(400, 3000)].reset_index(drop=True)
    X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(df[FEATURE_NAMES]),
                     columns=FEATURE_NAMES)
    y, groups = df["target_elo"], df["player"]
    print(f"{len(df)} партій, {groups.nunique()} гравців; GroupKFold(5) за гравцями\n")
    import shap

    A_all, B_all, gt_all, pred_all, maj_all = [], [], [], [], []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, groups), 1):
        m = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, random_state=RNG)
        m.fit(X.iloc[tr], y.iloc[tr])
        expl = shap.TreeExplainer(m)
        ci = [FEATURE_NAMES.index(c) for c in C]

        # норми внесків — лише на навчальних гравцях
        tr_agg = X.iloc[tr].groupby(groups.iloc[tr].values).mean()[FEATURE_NAMES]
        ref = band_ref(expl, tr_agg)
        # популяція для еталонного правила — теж лише навчальна
        tr_pm = X.iloc[tr][C].groupby(groups.iloc[tr].values).mean()
        mu_p, sd_p = tr_pm.mean(), tr_pm.std(ddof=0)
        orient = pd.Series(FEATURE_ORIENTATION)[C]
        # тривіальний орієнтир: найпоширеніша категорія за НАВЧАЛЬНИМИ гравцями
        maj_cat = ((tr_pm - mu_p) / sd_p).mul(orient).idxmax(axis=1).mode().iloc[0]

        te_df = df.iloc[te]
        Xte = X.iloc[te]
        # ── тест-ретест на відкладених гравцях ──
        splits = []
        for p, g in te_df.groupby("player"):
            idx = g.index.to_numpy()
            if len(idx) < 4: continue
            for s in SEEDS:
                sh = np.random.RandomState(s).permutation(idx)
                splits.append((sh[:len(sh)//2], sh[len(sh)//2:]))
        vecs = [X.loc[h].mean().values for a, b in splits for h in (a, b)]
        sv = expl.shap_values(pd.DataFrame(vecs, columns=FEATURE_NAMES))
        pr = np.asarray(expl.expected_value) + sv.sum(axis=1)
        R = np.array(C)[z_of(sv[:, ci], pr, ref).argsort(1)]
        A_all.append(R[0::2]); B_all.append(R[1::2])

        # ── збіг із еталоном на відкладених гравцях ──
        te_pm = Xte[C].groupby(groups.iloc[te].values).mean()
        gt = ((te_pm - mu_p) / sd_p).mul(orient).idxmax(axis=1)
        agg_te = Xte.groupby(groups.iloc[te].values).mean()[FEATURE_NAMES].loc[gt.index]
        sv2 = expl.shap_values(agg_te)
        pr2 = np.asarray(expl.expected_value) + sv2.sum(axis=1)
        P = np.array(C)[z_of(sv2[:, ci], pr2, ref).argsort(1)]
        gt_all.append(gt.values); pred_all.append(P)
        maj_all.append(np.full(len(gt), maj_cat))
        print(f"  блок {fold}: {te_df['player'].nunique()} відкладених гравців, "
              f"{len(splits)} розбиттів")

    A, B = np.vstack(A_all), np.vstack(B_all)
    gt = np.concatenate(gt_all); P = np.vstack(pred_all); maj = np.concatenate(maj_all)
    out = {
        "kappa": float(cohen_kappa_score(A[:, 0], B[:, 0], labels=C)),
        "split_half_agreement": float(np.mean(A[:, 0] == B[:, 0])),
        "topk_overlap_3": float(np.mean([len(set(x[:3]) & set(y[:3])) > 0 for x, y in zip(A, B)])),
        "top1_accuracy": float(np.mean(P[:, 0] == gt)),
        "coverage_at_3": float(np.mean([g in r[:3] for g, r in zip(gt, P)])),
        "kappa_agreement": float(cohen_kappa_score(gt, P[:, 0], labels=C)),
        "trivial_baseline": float(np.mean(maj == gt)),
        "n_players": int(len(gt)), "n_splits": int(len(A))}
    print("\n── ПОЗА-ФОЛДОВІ (без витоку) ──")
    for k, v in out.items():
        print(f"  {k:<22} {v:.4f}" if isinstance(v, float) else f"  {k:<22} {v}")
    json.dump(out, open("results/rules/xai_validation_blitz_oof.json", "w"), indent=1)
    print("\n->  results/rules/xai_validation_blitz_oof.json")


if __name__ == "__main__":
    main()
