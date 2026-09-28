"""
Критерій знака для кандидатів на слабкість — друга умова переліку T у (2.7).

Правило шукає ознаку з найменшим нормованим внеском і трактує її як слабкість.
Це коректно лише тоді, коли модель вивчила внесок у педагогічно правильному
напрямі: гірше значення ознаки -> нижчий внесок у прогноз. Критерій:

    sign(ρ_S(x̄_j, φ_j)) == −o_j,

де ρ_S — Спірмен на агрегованих векторах навчальних гравців пулу, o_j —
FEATURE_ORIENTATION (+1: більше = гірше). Він залежить від моделі пулу, тому
рахується для кожного пулу окремо (`sign_test` викликає build_shap_reference).

Скрипт виводить таблицю знаків для всіх 17 тематичних ознак і чотирьох пулів,
частку зворотних рекомендацій до і після фільтра, а також чутливість цієї
частки до орієнтації двох спірних ознак (material_volatility, king_danger),
для яких ρ(значення, Elo) слабка або протилежна оголошеній орієнтації.

Usage:
    python -m src.ml.candidate_signs --pool all --outdir results
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.features.extractor import FEATURE_NAMES, THEMED_CANDIDATES, FEATURE_ORIENTATION

POOLS = ["bullet", "blitz", "rapid", "classical"]
NB = 8
# Дві ознаки, чия оголошена орієнтація не підтверджується даними: ρ(x, Elo)
# слабка (king_danger ≈ +0,04 у бліці) або протилежна (material_volatility
# +0,20 при «більше = гірше»). Для них перевіряється чутливість.
DISPUTED = {"material_volatility": -1, "king_danger": -1}


def sign_test(agg, sv, feats, candidates, orient=None):
    """
    Кандидати, що проходять критерій знака, і таблиця кореляцій.

    agg  -- DataFrame агрегованих векторів (гравець × ознака);
    sv   -- масив SHAP тієї ж форми, стовпці у порядку `feats`;
    Повертає (passed, table), де table[c] = {"rho_shap", "correct_sign"}.
    """
    orient = pd.Series(FEATURE_ORIENTATION) if orient is None else pd.Series(orient)
    table, passed = {}, []
    for c in candidates:
        j = list(feats).index(c)
        r = float(spearmanr(agg[c], sv[:, j]).statistic)
        ok = bool((r < 0) if orient[c] > 0 else (r > 0))
        table[c] = {"rho_shap": r, "correct_sign": ok}
        if ok:
            passed.append(c)
    return passed, table


def partial_spearman(x, y, control):
    """ρ_S(x, y | control): Spearman between rank residuals after regressing on control."""
    rx, ry, rc = (pd.Series(v).rank().to_numpy(float) for v in (x, y, control))
    C = np.column_stack([np.ones(len(rc)), rc])
    res = lambda v: v - C @ np.linalg.lstsq(C, v, rcond=None)[0]
    return float(spearmanr(res(rx), res(ry)).statistic)


def zscores(sv, pred, cols):
    """Нормовані внески за (2.7), еталон будується на місці для заданого переліку."""
    sub = sv[:, [FEATURE_NAMES.index(c) for c in cols]]
    e = np.quantile(pred, np.linspace(0, 1, NB + 1))
    e[0], e[-1] = -np.inf, np.inf
    b = np.clip(np.searchsorted(e, pred, side="right") - 1, 0, NB - 1)
    filled = [i for i in range(NB) if np.any(b == i)]
    centres = np.array([pred[b == i].mean() for i in filled])
    mean = np.vstack([sub[b == i].mean(0) for i in filled])
    std = np.vstack([sub[b == i].std(0) for i in filled])
    mu = np.column_stack([np.interp(pred, centres, mean[:, j]) for j in range(len(cols))])
    sd = np.column_stack([np.interp(pred, centres, std[:, j]) for j in range(len(cols))])
    sd = np.where(sd > 0, sd, np.nan)
    z = (sub - mu) / sd
    return np.where(np.isfinite(z), z, np.inf)


def reversed_share(agg, sv, pred, cols, orient):
    """Частка відповідей правила, де гравець за названою ознакою КРАЩИЙ за
    медіану гравців того самого прогнозованого інтервалу."""
    band = pd.qcut(pred, NB, labels=False, duplicates="drop")
    named = np.array(cols)[zscores(sv, pred, cols).argmin(axis=1)]
    rev = sum(1 for i, c in enumerate(named)
              if (agg[c].values[i] - np.median(agg[c].values[band == band[i]]))
              * orient[c] < 0)
    return float(rev / len(named))


def run_pool(pool, outdir):
    from src.ml.model import ELOModel
    import shap
    df = pd.read_parquet(f"data/dataset_{pool}_2k.parquet")
    df = df[df["target_elo"].between(400, 3000)].reset_index(drop=True)
    agg = df.groupby("player")[FEATURE_NAMES].mean()          # nanmean, як у застосунку
    elo = df.groupby("player")["target_elo"].median().loc[agg.index]

    model = ELOModel(pool=pool)
    expl = shap.TreeExplainer(model.model)
    sv = expl.shap_values(agg)
    pred = np.asarray(expl.expected_value) + sv.sum(axis=1)
    orient = pd.Series(FEATURE_ORIENTATION)

    passed, table = sign_test(agg, sv, FEATURE_NAMES, THEMED_CANDIDATES, orient)
    print(f"\n== {pool}: {len(passed)} з {len(THEMED_CANDIDATES)} проходять критерій знака ==")
    print(f"{'ознака':<22}{'ρ(знач,Elo)':>13}{'ρ(знач,внесок)':>16}{'ρ(знач,Elo|acpl)':>18}   напрям")
    for c in THEMED_CANDIDATES:
        table[c]["rho_elo"] = float(spearmanr(agg[c], elo).statistic)
        # collinearity check: does the link with rating flip once overall ACPL is fixed?
        table[c]["rho_elo_given_acpl"] = partial_spearman(agg[c], elo, agg["acpl"])
        print(f"  {c:<20}{table[c]['rho_elo']:>13.2f}{table[c]['rho_shap']:>16.2f}"
              f"{table[c]['rho_elo_given_acpl']:>18.2f}   "
              f"{'правильний' if table[c]['correct_sign'] else 'ПРОТИЛЕЖНИЙ'}")

    stored = model._shap_reference["candidates"] if model._shap_reference else None
    if stored is not None and sorted(stored) != sorted(passed):
        print(f"  ! довідник пулу містить {len(stored)} кандидатів, критерій дає "
              f"{len(passed)} — перебудуйте build_shap_reference")

    out = {"pool": pool, "n_players": int(len(agg)), "signs": table,
           "passed": passed, "excluded": [c for c in THEMED_CANDIDATES if c not in passed],
           "reference_candidates": stored}
    for tag, cols in (("all", THEMED_CANDIDATES), ("sign_filtered", passed)):
        share = reversed_share(agg, sv, pred, cols, orient)
        out[tag] = {"n_candidates": len(cols), "reversed_share": share}
        print(f"  {tag}: {len(cols)} кандидатів, зворотних рекомендацій {share:.1%}")

    # чутливість до орієнтації спірних ознак: знак беремо з даних (ρ з Elo)
    emp = orient.copy()
    for c in DISPUTED:
        emp[c] = -1 if table[c]["rho_elo"] > 0 else 1
    out["sensitivity_empirical_orientation"] = {
        "disputed": {c: {"declared": int(orient[c]), "empirical": int(emp[c])} for c in DISPUTED},
        "all": reversed_share(agg, sv, pred, THEMED_CANDIDATES, emp),
        "sign_filtered": reversed_share(agg, sv, pred, passed, emp)}
    s = out["sensitivity_empirical_orientation"]
    print(f"  за емпіричною орієнтацією спірних ознак: зворотних "
          f"{s['all']:.1%} (17) / {s['sign_filtered']:.1%} (після фільтра)")

    path = os.path.join(outdir, f"candidate_signs_{pool}.json")
    json.dump(out, open(path, "w"), indent=1, ensure_ascii=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="blitz", choices=POOLS + ["all"])
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    pools = POOLS if args.pool == "all" else [args.pool]
    res = {p: run_pool(p, args.outdir) for p in pools}
    if len(pools) > 1:
        rows = []
        for c in THEMED_CANDIDATES:
            row = {"feature": c}
            for p in pools:
                row[p] = res[p]["signs"][c]["rho_shap"]
                row[f"{p}_ok"] = res[p]["signs"][c]["correct_sign"]
            rows.append(row)
        summary = {"table": rows,
                   "n_candidates": {p: len(res[p]["passed"]) for p in pools},
                   "excluded": {p: res[p]["excluded"] for p in pools}}
        path = os.path.join(args.outdir, "candidate_signs_all.json")
        json.dump(summary, open(path, "w"), indent=1, ensure_ascii=False)
        print(f"\n-> {path}")


if __name__ == "__main__":
    main()
