"""
Контрольний приклад для статті у «Віснику ВПІ» (рецензія 14.10.2026): повний розклад
таблиці 3.23 магістерської (`results/worked_examples.csv`) для гравців бліцу.

Для кожного гравця той самий шлях, що й у `src/ml/worked_examples.py`: перші вісім партій
гравця з `data/dataset_blitz_2k.parquet`, `aggregate_features(..., "mean")`, розгорнута модель
бліцу й довідник `shap_reference_blitz.pkl`. Зберігаються проміжні величини, які в CSV не
потрапили: середній прогноз φ0, внески φj, норми μj(r̂), σj(r̂), оцінки sj усіх кандидатів,
усереднені значення ознак гравця й типові значення для його прогнозованого рівня (середнє
усереднених ознак навчальних гравців за інтервалами прогнозу, лінійно інтерпольоване між
центрами інтервалів так само, як μj), відповідь базового правила (найменший φj із 29) і
відрив s2 − s1.

Розгорнута модель навчалася на всіх гравцях бліцу, зокрема на прикладових, тож ці числа
отримано у вибірці. Тому той самий приклад повторено за протоколом без витоку
(`src/ml/validate_oof.py`): GroupKFold(5) за гравцями, модель, критерій знака й норми будуються
на навчальних гравцях того блоку, де прикладовий гравець тестовий.

Usage:
    python -m src.ml.worked_example_detail --players AAPA23,AB9920,Aldo333 \
        --out results/worked_example_detail_blitz.json
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.features.extractor import FEATURE_NAMES, FEATURE_ORIENTATION, THEMED_CANDIDATES  # noqa: E402
from src.features.aggregate import aggregate_features  # noqa: E402
from src.ml.model import ELOModel  # noqa: E402
from src.ml import build_shap_reference  # noqa: E402
from src.ml.validate_oof import fit_fold, band_ref, z_of  # noqa: E402
from src.ml.candidate_signs import sign_test  # noqa: E402

T17 = list(THEMED_CANDIDATES)


def in_sample(df, players):
    """Розгорнута модель і збережений довідник – як у worked_examples.py."""
    model = ELOModel(pool="blitz")
    ref = model._shap_reference
    cands = list(ref["candidates"])
    feats = model.feature_names
    dfp = df[df["target_elo"].between(400, 3000)]
    agg_all = dfp.groupby("player")[feats].mean()
    pred_all = pd.Series(model.model.predict(agg_all), index=agg_all.index)
    band_all = pd.qcut(pred_all, ref["n_bands"], labels=False, duplicates="drop")
    centres = pred_all.groupby(band_all).mean().sort_index().values
    assert np.allclose(centres, ref["band_centres"]), "центри інтервалів не збігаються зі збереженим довідником"
    worked = pd.read_csv(os.path.join("results", "worked_examples.csv"))
    out = {}
    for p in players:
        g = df[df["player"] == p].head(8)
        fd = aggregate_features(g[FEATURE_NAMES].to_dict("records"), "mean")
        o = model.predict_and_explain(fd, k=3)
        x = pd.DataFrame([{f: fd.get(f, np.nan) for f in feats}], columns=feats)
        sv = model.explainer.shap_values(x)[0]
        phi0 = float(np.asarray(model.explainer.expected_value).ravel()[0])
        phi = dict(zip(feats, map(float, sv)))
        r = o["predicted_elo"]
        mu, sd = build_shap_reference.stats_at(ref, r)
        s = build_shap_reference.standardised_scores(phi, mu, sd, cands)
        rows = []
        for c in sorted(s, key=s.get):
            mean_b = agg_all.groupby(band_all)[c].mean().sort_index().values
            std_b = agg_all.groupby(band_all)[c].std(ddof=0).sort_index().values
            rows.append({"feature": c, "orientation": FEATURE_ORIENTATION[c], "x_player": float(fd[c]),
                         "typical_x": float(np.interp(r, centres, mean_b)), "sd_x": float(np.interp(r, centres, std_b)),
                         "phi": phi[c], "mu": float(mu[c]), "sigma": float(sd[c]), "s": float(s[c])})
        basic = min(phi, key=phi.get)
        w = worked[worked["гравець"] == p]
        top3 = [f for f, _ in sorted(s.items(), key=lambda kv: kv[1])[:3]]
        if len(w):  # контроль: ті самі прогноз, напрями й оцінки, що в таблиці 3.23
            assert int(w["прогноз"].iloc[0]) == int(round(r)), (p, r)
            assert list(w["напрям"]) == top3, (p, top3)
            assert np.allclose([float(v) for v in w["z"]], [s[c] for c in top3], atol=0.006), p
        out[p] = {"n_games": int(len(g)), "fact_median_elo": float(g["target_elo"].median()),
                  "elo_min": int(g["target_elo"].min()), "elo_max": int(g["target_elo"].max()),
                  "time_controls": sorted(set(g["time_control"].astype(str))),
                  "predicted_elo": r, "phi0": phi0, "phi0_plus_sum_phi": phi0 + float(np.sum(sv)),
                  "n_candidates": len(cands), "top3": top3,
                  "margin_s2_minus_s1": o["weakest_margin"],
                  "basic_rule_argmin29": basic, "basic_rule_phi": phi[basic],
                  "candidates_ranked": rows}
    return out


def out_of_fold(df, players):
    """Той самий приклад без витоку: протокол validate_oof.py, гравець у тестовому блоці."""
    df = df[df["target_elo"].between(400, 3000)].reset_index(drop=True)
    X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(df[FEATURE_NAMES]), columns=FEATURE_NAMES)
    y, groups = df["target_elo"], df["player"]
    out = {}
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, groups), 1):
        targets = [p for p in players if p in set(groups.iloc[te])]
        if not targets:
            continue
        m, expl = fit_fold(X, y, groups, tr)
        tr_agg = X.iloc[tr].groupby(groups.iloc[tr].values).mean()[FEATURE_NAMES]
        sv_tr = expl.shap_values(tr_agg)
        pred_tr = np.asarray(expl.expected_value) + sv_tr.sum(axis=1)
        C, _ = sign_test(tr_agg, sv_tr, FEATURE_NAMES, T17)
        ref_s = band_ref(sv_tr[:, [FEATURE_NAMES.index(c) for c in C]], pred_tr)
        for p in targets:
            idx = df.index[df["player"] == p]
            agg = X.loc[idx].mean()[FEATURE_NAMES].to_frame().T
            sv = expl.shap_values(agg)
            pr = float(np.asarray(expl.expected_value) + sv.sum(axis=1)[0])
            z = z_of(sv[:, [FEATURE_NAMES.index(c) for c in C]].astype(float), np.array([pr]), ref_s)[0]
            order = np.argsort(z)
            phi = dict(zip(FEATURE_NAMES, map(float, sv[0])))
            out[p] = {"fold": fold, "n_games": int(len(idx)), "predicted_elo_oof": pr, "n_candidates": len(C),
                      "candidates": list(C), "top3": [C[k] for k in order[:3]],
                      "scores": {C[k]: float(z[k]) for k in order},
                      "basic_rule_argmin29": min(phi, key=phi.get)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", default="AAPA23,AB9920,Aldo333")
    ap.add_argument("--out", default="results/worked_example_detail_blitz.json")
    args = ap.parse_args()
    players = args.players.split(",")
    df = pd.read_parquet(os.path.join("data", "dataset_blitz_2k.parquet"))
    res = {"note": "in_sample – розгорнута модель бліцу (гравці є в навчальному наборі); "
                   "oof – протокол validate_oof.py (GroupKFold(5), модель і норми без гравця)",
           "in_sample": in_sample(df, players), "oof": out_of_fold(df, players)}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    for p in players:
        a, b = res["in_sample"][p], res["oof"].get(p, {})
        print(f"{p}: факт {a['fact_median_elo']:.0f}, прогноз {a['predicted_elo']}, φ0 {a['phi0']:.1f}, "
              f"трійка {a['top3']}, базове {a['basic_rule_argmin29']} ({a['basic_rule_phi']:+.1f}); "
              f"без витоку: прогноз {b.get('predicted_elo_oof', float('nan')):.1f}, трійка {b.get('top3')}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
