"""
Базовий варіант без SHAP: правило (2.7), застосоване до сирих значень ознак.

Питання: чи потрібен крок пояснення взагалі? Якщо взяти той самий перелік
кандидатів T і те саме нормування на гравців того самого прогнозованого рівня,
але нормувати не внески SHAP, а самі значення ознак (з урахуванням орієнтації),
отримаємо рівнево-умовне правило відхилень. Воно не використовує модель
інакше, ніж для прогнозу r̂, який задає рівень порівняння.

Скрипт вимірює для обох правил на ОДНІЙ навченій моделі й тих самих розбиттях
(протокол validate_xai / таблиця 3.11):
  - тест-ретест κ і перетин трійок між половинами партій гравця;
  - збіг із правилом відхилень за популяцією (17-класовий еталон і еталон над T);
  - частку рекомендацій протилежного напряму;
  - узгодженість двох правил між собою.

Usage:
    python -m src.ml.raw_deviation_baseline --pool blitz --outdir results/rules
"""
import argparse
import json
import os
import sys
from math import comb

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.features.extractor import FEATURE_NAMES, THEMED_CANDIDATES, FEATURE_ORIENTATION
from src.ml import validate_xai as vx

NB, SEEDS, RNG = 8, range(10), 42


def raw_band_ref(pred, agg, cols):
    """Середнє та std сирих значень ознак за інтервалами прогнозованого рейтингу."""
    edges = np.quantile(pred, np.linspace(0, 1, NB + 1)); edges[0], edges[-1] = -np.inf, np.inf
    b = np.clip(np.searchsorted(edges, pred, side="right") - 1, 0, NB - 1)
    filled = [i for i in range(NB) if np.any(b == i)]
    vals = agg[cols].to_numpy(float)
    return {"cols": list(cols), "centres": np.array([pred[b == i].mean() for i in filled]),
            "mean": np.vstack([vals[b == i].mean(0) for i in filled]),
            "std": np.vstack([vals[b == i].std(0) for i in filled])}


def raw_rank(V, pred, ref, cols, k=3):
    """Ранжування ознак за орієнтованою z-оцінкою сирих значень відносно однолітків (найгірша перша)."""
    x = ref["centres"]
    jj = [ref["cols"].index(c) for c in cols]                 # довідник може містити більше ознак, ніж cols
    mu = np.column_stack([np.interp(pred, x, ref["mean"][:, j]) for j in jj])
    sd = np.column_stack([np.interp(pred, x, ref["std"][:, j]) for j in jj])
    sd = np.where(sd > 0, sd, np.nan)
    orient = np.array([FEATURE_ORIENTATION[c] for c in cols], dtype=float)
    z = (V[cols].to_numpy(float) - mu) / sd * orient           # більше = гірше
    z = np.where(np.isfinite(z), z, -np.inf)
    order = np.argsort(-z, axis=1)[:, :k]
    return np.array(cols)[order]


def pair_kappa(a, b, labels):
    return float(cohen_kappa_score(a, b, labels=labels))


def overlap3(A, B):
    return float(np.mean([len(set(x[:3]) & set(y[:3])) > 0 for x, y in zip(A, B)]))


def reversed_share(named, agg, pred, cols):
    """Частка відповідей, де гравець за названою ознакою КРАЩИЙ за медіану однолітків."""
    band = pd.qcut(pred, NB, labels=False, duplicates="drop")
    orient = pd.Series(FEATURE_ORIENTATION)
    rev = sum(1 for i, c in enumerate(named)
              if (agg[c].values[i] - np.median(agg[c].values[band == band[i]])) * orient[c] < 0)
    return float(rev / len(named))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="blitz")
    ap.add_argument("--outdir", default="results/rules")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_parquet(f"data/dataset_{args.pool}_2k.parquet")
    df = df[df["target_elo"].between(400, 3000)].reset_index(drop=True)
    model, explainer, Xi = vx.fit_model_and_explainer(df[FEATURE_NAMES], df["target_elo"], df["player"])
    T, source = vx.configure_rule("standardised", explainer, Xi, df, args.pool)
    T17 = list(THEMED_CANDIDATES)
    print(f"{args.pool}: {len(df):,} партій, {df['player'].nunique()} гравців; T = {len(T)} ({source})")

    agg = Xi.groupby(df["player"].values).mean()[FEATURE_NAMES]
    players = agg.index
    pred_agg = model.predict(agg)
    ref_raw = raw_band_ref(pred_agg, agg, T17)

    # ── розбиття навпіл, як у validate_xai.experiment_a ──
    splits = []
    for player, g in df.groupby("player"):
        if len(g) < 4: continue
        idx = g.index.to_numpy()
        for seed in SEEDS:
            sh = np.random.RandomState(seed).permutation(idx); h = len(sh) // 2
            splits.append((player, sh[:h], sh[h:]))
    halves = [h for _, a, b in splits for h in (a, b)]
    H = pd.DataFrame([Xi.loc[rows].mean().values for rows in halves], columns=FEATURE_NAMES)
    pred_h = model.predict(H)

    shap_ranked = vx.aggregate_weakness(explainer, Xi, halves, k=3)          # правило (2.7), SHAP
    raw_ranked = raw_rank(H, pred_h, ref_raw, T, k=3)                          # те саме на сирих ознаках
    labels = sorted(set(T17))
    out = {"pool": args.pool, "candidates": T, "n_splits": len(splits),
           "n_players": int(len({s[0] for s in splits}))}
    for name, Rk in (("shap_rule", shap_ranked), ("raw_rule", raw_ranked)):
        A, B = Rk[0::2], Rk[1::2]
        out[name] = {"kappa_testretest": pair_kappa(A[:, 0], B[:, 0], labels),
                     "agreement_testretest": float(np.mean(A[:, 0] == B[:, 0])),
                     "top3_overlap": overlap3(A, B)}
    out["chance_top3_overlap"] = 1 - comb(len(T) - 3, 3) / comb(len(T), 3)
    # узгодженість двох правил на тих самих половинах
    out["cross_rule"] = {"top1_agreement": float(np.mean(shap_ranked[:, 0] == raw_ranked[:, 0])),
                         "top3_overlap": overlap3(shap_ranked, raw_ranked),
                         "kappa": pair_kappa(shap_ranked[:, 0], raw_ranked[:, 0], labels)}

    # ── збіг із популяційним правилом відхилень (експеримент B) ──
    gt17 = vx._ground_truth(df, T17).loc[players].values
    gt_own = vx._ground_truth(df, T).loc[players].values
    shap_full = vx.aggregate_weakness(explainer, Xi, [df.index[df["player"] == p].to_numpy() for p in players], k=3)
    raw_full = raw_rank(agg, pred_agg, ref_raw, T, k=3)
    maj17 = pd.Series(gt17).value_counts(normalize=True).iloc[0]
    maj_own = pd.Series(gt_own).value_counts(normalize=True).iloc[0]
    for name, Rk in (("shap_rule", shap_full), ("raw_rule", raw_full)):
        out[name].update({
            "agree_ref17": float(np.mean(Rk[:, 0] == gt17)),
            "agree3_ref17": float(np.mean([g in r for g, r in zip(gt17, Rk)])),
            "agree_own": float(np.mean(Rk[:, 0] == gt_own)),
            "agree3_own": float(np.mean([g in r for g, r in zip(gt_own, Rk)])),
            "kappa_agree_ref17": pair_kappa(gt17, Rk[:, 0], labels),
            "kappa_agree_own": pair_kappa(gt_own, Rk[:, 0], labels),
            "reversed_share": reversed_share(Rk[:, 0], agg, pred_agg, T),
            "top_share": float(pd.Series(Rk[:, 0]).value_counts(normalize=True).iloc[0])})
    out["trivial_ref17"], out["trivial_own"] = float(maj17), float(maj_own)
    out["cross_rule"]["top1_agreement_full"] = float(np.mean(shap_full[:, 0] == raw_full[:, 0]))

    # ── кластерний бутстреп для різниці κ між правилами ──
    rec = pd.DataFrame({"player": [s[0] for s in splits],
                        "sa": shap_ranked[0::2, 0], "sb": shap_ranked[1::2, 0],
                        "ra": raw_ranked[0::2, 0], "rb": raw_ranked[1::2, 0]})
    groups = {p: g for p, g in rec.groupby("player")}; pl = np.array(list(groups))
    rng = np.random.RandomState(RNG); deltas, k_raw, k_shap = [], [], []
    for _ in range(args.boot):
        s = pd.concat([groups[p] for p in rng.choice(pl, len(pl), replace=True)])
        ks, kr = pair_kappa(s.sa, s.sb, labels), pair_kappa(s.ra, s.rb, labels)
        deltas.append(ks - kr); k_shap.append(ks); k_raw.append(kr)
    d = np.array(deltas)
    span = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    out["raw_rule"]["kappa_ci95"] = span(k_raw); out["shap_rule"]["kappa_ci95"] = span(k_shap)
    out["kappa_delta_shap_minus_raw"] = {"delta": out["shap_rule"]["kappa_testretest"] - out["raw_rule"]["kappa_testretest"],
                                         "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                                         "p_gt_0": float(np.mean(d > 0))}

    path = os.path.join(args.outdir, f"raw_baseline_{args.pool}.json")
    json.dump(out, open(path, "w"), indent=1, ensure_ascii=False)
    print(f"{'показник':<28}{'SHAP (2.7)':>12}{'сирі ознаки':>13}")
    for k in ("kappa_testretest", "agreement_testretest", "top3_overlap", "agree_ref17", "agree3_ref17",
              "kappa_agree_ref17", "agree_own", "agree3_own", "kappa_agree_own", "reversed_share", "top_share"):
        print(f"{k:<28}{out['shap_rule'][k]:>12.3f}{out['raw_rule'][k]:>13.3f}")
    print(f"{'тривіальний (17 / T)':<28}{out['trivial_ref17']:>12.3f}{out['trivial_own']:>13.3f}")
    print(f"{'випадковий перетин трійок':<28}{out['chance_top3_overlap']:>25.3f}")
    c = out["cross_rule"]
    print(f"узгодженість правил: top-1 {c['top1_agreement']:.3f} (половини) / {c['top1_agreement_full']:.3f} (усі партії), "
          f"перетин трійок {c['top3_overlap']:.3f}, κ {c['kappa']:.3f}")
    dd = out["kappa_delta_shap_minus_raw"]
    print(f"Δκ (SHAP − сирі) = {dd['delta']:+.3f} [{dd['ci95'][0]:+.3f}; {dd['ci95'][1]:+.3f}], P(Δ>0) = {dd['p_gt_0']:.3f}")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
