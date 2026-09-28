"""
Причина інверсії знака: модель, а не спосіб атрибуції.

Шість кандидатів мають внесок, протилежний педагогічному напряму
(src/ml/candidate_signs.py). Скрипт перевіряє дві гіпотези про причину.

  1. Артефакт умовного TreeSHAP. За замовчуванням TreeSHAP іде умовним
     сподіванням уздовж шляхів дерева, і на корельованих ознаках це може
     перевернути знак. Перевірка: інтервенційний SHAP із фоновою вибіркою,
     який рахує маргінальні значення Шеплі. Модель не змінюється.

  2. Властивість самої моделі. Перевірка: та сама архітектура з монотонними
     обмеженнями на кандидатах — гірше значення ознаки не може підвищувати
     прогноз. Змінюється модель, спосіб атрибуції той самий.

Результат: перша гіпотеза хибна (усі шість знаків лишаються протилежними), і
саме це є доказом, що інверсія лежить у моделі. Нуль протилежних знаків у
монотонній моделі — наслідок обмежень за побудовою, а не самостійний доказ;
цінність монотонної моделі в іншому: вона повертає категорії, які механізм не
виявляв, і показує ціну цього у точності прогнозу. Обидва варіанти правила тут
порівнюються з ОДНИМ 17-класовим еталоном (ALL_CANDIDATES), тому їхні збіги
зіставні між собою, але не з «власним» збігом 11-ознакового правила у
validate_xai (див. поле ref17 там).

Usage:
    python -m src.ml.monotone_analysis --pool blitz --outdir results
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.features.extractor import (FEATURE_NAMES, THEMED_CANDIDATES,
                                    SIGN_EXCLUDED, FEATURE_ORIENTATION)
from src.ml.model import ELOModel
from src.ml.mixed_training import mixed_training_set

ALL_CANDIDATES = list(THEMED_CANDIDATES)
PARAMS = dict(n_estimators=300, max_depth=5, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.8, random_state=42)
NB = 8


def wrong_signs(sv, agg, cols, orient):
    bad = []
    for c in cols:
        r = float(spearmanr(agg[c], sv[:, FEATURE_NAMES.index(c)]).statistic)
        ok = (r < 0) if orient[c] > 0 else (r > 0)
        if not ok:
            bad.append(c)
    return bad


def named_weakness(sv, pred, agg, cols):
    sub = sv[:, [FEATURE_NAMES.index(c) for c in cols]]
    e = np.quantile(pred, np.linspace(0, 1, NB + 1)); e[0], e[-1] = -np.inf, np.inf
    b = np.clip(np.searchsorted(e, pred, side="right") - 1, 0, NB - 1)
    filled = [i for i in range(NB) if np.any(b == i)]
    cen = np.array([pred[b == i].mean() for i in filled])
    mean = np.vstack([sub[b == i].mean(0) for i in filled])
    std = np.vstack([sub[b == i].std(0) for i in filled])
    mu = np.column_stack([np.interp(pred, cen, mean[:, j]) for j in range(len(cols))])
    sd = np.column_stack([np.interp(pred, cen, std[:, j]) for j in range(len(cols))])
    sd = np.where(sd > 0, sd, np.nan)
    z = (sub - mu) / sd
    return np.array(cols)[np.where(np.isfinite(z), z, np.inf).argmin(axis=1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="blitz")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    import shap

    df = pd.read_parquet(f"data/dataset_{args.pool}_2k.parquet")
    df = df[df["target_elo"].between(400, 3000)].reset_index(drop=True)
    X = df[FEATURE_NAMES].fillna(df[FEATURE_NAMES].median())
    y, groups = df["target_elo"], df["player"]
    agg = X.groupby(groups.values).mean()[FEATURE_NAMES]
    orient = pd.Series(FEATURE_ORIENTATION)
    out = {"pool": args.pool, "n_players": int(len(agg))}

    # еталонна слабкість: найбільше відхилення значення в несприятливий бік
    pm = X.groupby(groups.values)[ALL_CANDIDATES].mean()
    gt = ((pm - pm.mean()) / pm.std(ddof=0)).mul(orient[ALL_CANDIDATES]).idxmax(axis=1)

    prod = ELOModel(pool=args.pool).model
    sv_cond = shap.TreeExplainer(prod).shap_values(agg)
    pred = np.asarray(shap.TreeExplainer(prod).expected_value) + sv_cond.sum(axis=1)
    bad_cond = wrong_signs(sv_cond, agg, ALL_CANDIDATES, orient)
    print(f"умовний TreeSHAP (як є):      протилежних знаків {len(bad_cond)}")

    # ── гіпотеза 1: інтервенційний SHAP ─────────────────────────────────
    bg = shap.sample(X, 200, random_state=0)
    sv_int = shap.TreeExplainer(prod, data=bg, feature_perturbation="interventional") \
        .shap_values(agg, check_additivity=False)
    bad_int = wrong_signs(sv_int, agg, ALL_CANDIDATES, orient)
    print(f"інтервенційний SHAP:          протилежних знаків {len(bad_int)}"
          f"  -> гіпотеза {'хибна' if bad_int else 'правильна'}")

    # ── гіпотеза 2: монотонні обмеження ─────────────────────────────────
    mono = tuple(int(-orient[f]) if f in ALL_CANDIDATES else 0 for f in FEATURE_NAMES)
    # обидві моделі — на змішаному наборі (як train.py); точність поза блоками за довжиною серії
    frame = X.copy(); frame["target_elo"] = y.values; frame["player"] = groups.values
    Xm, ym, gm, nm = mixed_training_set(frame)
    oof = {"unconstrained": np.full(len(ym), np.nan), "monotone": np.full(len(ym), np.nan)}
    for tr, te in GroupKFold(n_splits=5).split(Xm, ym, gm):
        oof["unconstrained"][te] = XGBRegressor(**PARAMS, n_jobs=-1).fit(Xm.iloc[tr], ym.iloc[tr]).predict(Xm.iloc[te])
        oof["monotone"][te] = XGBRegressor(**PARAMS, monotone_constraints=mono, n_jobs=-1) \
            .fit(Xm.iloc[tr], ym.iloc[tr]).predict(Xm.iloc[te])
    accuracy = {k: {str(N): {"r2": float(r2_score(ym[nm == N], v[nm == N])),
                             "mae": float(mean_absolute_error(ym[nm == N], v[nm == N]))}
                    for N in (1, 8)} for k, v in oof.items()}
    r2s = [accuracy["monotone"]["1"]["r2"]]; maes = [accuracy["monotone"]["1"]["mae"]]
    mm = XGBRegressor(**PARAMS, monotone_constraints=mono, n_jobs=-1).fit(Xm, ym)
    sv_mono = shap.TreeExplainer(mm).shap_values(agg)
    pred_mono = np.asarray(shap.TreeExplainer(mm).expected_value) + sv_mono.sum(axis=1)
    bad_mono = wrong_signs(sv_mono, agg, ALL_CANDIDATES, orient)
    print(f"монотонні обмеження:          протилежних знаків {len(bad_mono)}"
          f"  -> гіпотеза {'правильна' if not bad_mono else 'хибна'}")
    for N in ("1", "8"):
        u, mo = accuracy["unconstrained"][N], accuracy["monotone"][N]
        print(f"  ціна точності (N={N}): R² {u['r2']:.3f} -> {mo['r2']:.3f}, MAE {u['mae']:.1f} -> {mo['mae']:.1f}")

    # ── чи повертає монотонна модель невиявлювані категорії ─────────────
    n_cond = named_weakness(sv_cond, pred, agg, ALL_CANDIDATES)
    n_mono = named_weakness(sv_mono, pred_mono, agg, ALL_CANDIDATES)
    per_cat = {}
    print(f"\n{'категорія':<22}{'гравців':>9}{'як є':>9}{'монотонна':>12}")
    for c in ALL_CANDIDATES:
        mask = (gt.values == c)
        if not mask.sum():
            continue
        a, b = float(np.mean(n_cond[mask] == c)), float(np.mean(n_mono[mask] == c))
        per_cat[c] = {"n": int(mask.sum()), "conditional": a, "monotone": b}
        print(f"  {c:<20}{mask.sum():>9}{a:>9.3f}{b:>12.3f}"
              + ("   <- поза T (критерій знака)" if c in SIGN_EXCLUDED else ""))
    ov_c, ov_m = float(np.mean(n_cond == gt.values)), float(np.mean(n_mono == gt.values))

    # ── відтворюваність (тест-ретест) над 17 кандидатами: та сама модель проти монотонної ──
    # протокол validate_xai.experiment_a: половини партій гравця, 10 перестановок,
    # норми внесків — на агрегованих векторах усіх гравців, застосовані до половин
    splits = []
    for _p, g in df.groupby("player"):
        idx = g.index.to_numpy()
        if len(idx) < 4:
            continue
        for s in range(10):
            sh = np.random.RandomState(s).permutation(idx)
            splits.append((sh[:len(sh) // 2], sh[len(sh) // 2:]))
    H = pd.DataFrame([X.loc[h].mean().values for a, b in splits for h in (a, b)], columns=FEATURE_NAMES)
    kappas = {}
    for name, mdl in (("conditional", prod), ("monotone", mm)):
        ex = shap.TreeExplainer(mdl)
        sv_a = ex.shap_values(agg); pr_a = np.asarray(ex.expected_value) + sv_a.sum(axis=1)
        sub_a = sv_a[:, [FEATURE_NAMES.index(c) for c in ALL_CANDIDATES]]
        e = np.quantile(pr_a, np.linspace(0, 1, NB + 1)); e[0], e[-1] = -np.inf, np.inf
        b = np.clip(np.searchsorted(e, pr_a, side="right") - 1, 0, NB - 1)
        filled = [i for i in range(NB) if np.any(b == i)]
        cen = np.array([pr_a[b == i].mean() for i in filled])
        mean = np.vstack([sub_a[b == i].mean(0) for i in filled]); std = np.vstack([sub_a[b == i].std(0) for i in filled])
        sv_h = ex.shap_values(H); pr_h = np.asarray(ex.expected_value) + sv_h.sum(axis=1)
        sub_h = sv_h[:, [FEATURE_NAMES.index(c) for c in ALL_CANDIDATES]]
        mu = np.column_stack([np.interp(pr_h, cen, mean[:, j]) for j in range(len(ALL_CANDIDATES))])
        sd = np.column_stack([np.interp(pr_h, cen, std[:, j]) for j in range(len(ALL_CANDIDATES))])
        sd = np.where(sd > 0, sd, np.nan)
        z = np.where(np.isfinite((sub_h - mu) / sd), (sub_h - mu) / sd, np.inf)
        named = np.array(ALL_CANDIDATES)[z.argmin(axis=1)]
        from sklearn.metrics import cohen_kappa_score
        kappas[name] = float(cohen_kappa_score(named[0::2], named[1::2], labels=ALL_CANDIDATES))
    print(f"\nκ тест-ретесту над 17 кандидатами: як є {kappas['conditional']:.3f}, монотонна {kappas['monotone']:.3f}")
    print(f"  {'РАЗОМ':<20}{len(gt):>9}{ov_c:>9.3f}{ov_m:>12.3f}")

    out.update(
        wrong_signs={"conditional": bad_cond, "interventional": bad_int, "monotone": bad_mono},
        monotone_accuracy={"r2": float(np.mean(r2s)), "mae": float(np.mean(maes))},
        accuracy_by_n=accuracy,
        kappa_testretest_17=kappas,
        per_category=per_cat,
        overall_agreement={"conditional": ov_c, "monotone": ov_m})
    path = os.path.join(args.outdir, f"monotone_analysis_{args.pool}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"\n->  {path}")


if __name__ == "__main__":
    main()
