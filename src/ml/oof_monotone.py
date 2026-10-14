"""
Монотонна модель за протоколом без витоку даних за гравцями (підрозділ 3.3.5).

monotone_analysis.py рахує відтворюваність і збіг з 17-класовим еталоном на
моделях, навчених на всіх гравцях, і на нормах, побудованих на тих самих
гравцях. Тут те саме за протоколом validate_oof: GroupKFold(5) за гравцями; у
кожному блоці звичайна й монотонна моделі (змішаний набір, ті самі параметри),
норми внесків для 17 тематичних ознак і популяція правила відхилень будуються
лише на навчальних гравцях, а тест-ретест і збіг рахуються на відкладених.

Контроль: для звичайної моделі κ і збіг мають збігтися з кроком 2
(«standardised17») у results/rules/table_311_<pool>_oof.json.

Usage:
    python -m src.ml.oof_monotone --pool blitz --outdir results/rules
"""
import argparse, json, os, sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.features.extractor import FEATURE_NAMES, FEATURE_ORIENTATION
from src.ml.mixed_training import mixed_training_set
from src.ml.validate_oof import T17, ALL29, SEEDS, PARAMS, RNG, band_ref, rank_shap, kappa, fit_fold


def fit_fold_mono(X, y, groups, tr, mono):
    """Як validate_oof.fit_fold, але з монотонними обмеженнями на 17 тематичних ознаках."""
    import shap
    trf = X.iloc[tr].copy(); trf["target_elo"] = y.iloc[tr].values; trf["player"] = groups.iloc[tr].values
    Xm, ym, _, _ = mixed_training_set(trf)
    m = xgb.XGBRegressor(**PARAMS, monotone_constraints=mono).fit(Xm, ym)
    return m, shap.TreeExplainer(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="blitz")
    ap.add_argument("--outdir", default="results/rules")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_parquet(f"data/dataset_{args.pool}_2k.parquet")
    df = df[df["target_elo"].between(400, 3000)].reset_index(drop=True)
    X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(df[FEATURE_NAMES]), columns=FEATURE_NAMES)
    y, groups = df["target_elo"], df["player"]
    orient = pd.Series(FEATURE_ORIENTATION)
    mono = tuple(int(-orient[f]) if f in T17 else 0 for f in FEATURE_NAMES)   # як у monotone_analysis
    print(f"{args.pool}: {len(df)} партій, {groups.nunique()} гравців; GroupKFold(5), звичайна й монотонна моделі в кожному блоці\n")

    MODELS = ("conditional", "monotone")
    half_A = {k: [] for k in MODELS}; half_B = {k: [] for k in MODELS}; half_player = []
    named_full = {k: [] for k in MODELS}; g17_all = []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, groups), 1):
        fits = {"conditional": fit_fold(X, y, groups, tr), "monotone": fit_fold_mono(X, y, groups, tr, mono)}
        tr_agg = X.iloc[tr].groupby(groups.iloc[tr].values).mean()[FEATURE_NAMES]
        # популяція правила відхилень – лише навчальні гравці (як у validate_oof)
        tr_pm = X.iloc[tr].groupby(groups.iloc[tr].values)[ALL29].mean()
        mu_p, sd_p = tr_pm.mean(), tr_pm.std(ddof=0)
        te_df = df.iloc[te]; Xte = X.iloc[te]
        te_pm = Xte.groupby(groups.iloc[te].values)[ALL29].mean()
        z_te = ((te_pm - mu_p) / sd_p).mul(orient[ALL29])
        g17 = z_te[T17].idxmax(axis=1)
        agg_te = Xte.groupby(groups.iloc[te].values).mean()[FEATURE_NAMES].loc[te_pm.index]
        # половини партій відкладених гравців (ті самі розбиття, що в validate_oof)
        splits = []
        for p, g in te_df.groupby("player"):
            idx = g.index.to_numpy()
            if len(idx) < 4: continue
            for s_ in SEEDS:
                sh = np.random.RandomState(s_).permutation(idx); splits.append((p, sh[:len(sh) // 2], sh[len(sh) // 2:]))
        H = pd.DataFrame([X.loc[h].mean().values for _, a, b in splits for h in (a, b)], columns=FEATURE_NAMES)
        for k, (m, expl) in fits.items():
            sv_tr = expl.shap_values(tr_agg); pred_tr = np.asarray(expl.expected_value) + sv_tr.sum(axis=1)
            ref = band_ref(sv_tr[:, [FEATURE_NAMES.index(c) for c in T17]], pred_tr)
            svh = expl.shap_values(H); prh = np.asarray(expl.expected_value) + svh.sum(axis=1)
            R = rank_shap(svh, prh, T17, ref)
            half_A[k].append(R[0::2, 0]); half_B[k].append(R[1::2, 0])
            sva = expl.shap_values(agg_te); pra = np.asarray(expl.expected_value) + sva.sum(axis=1)
            named_full[k].append(rank_shap(sva, pra, T17, ref)[:, 0])
        half_player.append([s[0] for s in splits]); g17_all.append(g17.values)
        print(f"  блок {fold}: {te_df['player'].nunique()} відкладених гравців, {len(splits)} розбиттів")

    labels17 = sorted(T17); g17 = np.concatenate(g17_all)
    players_h = np.concatenate(half_player)
    rng = np.random.RandomState(RNG)
    gh = {p: np.where(players_h == p)[0] for p in np.unique(players_h)}; pl = np.array(list(gh))
    draws = [np.concatenate([gh[p] for p in rng.choice(pl, len(pl), replace=True)]) for _ in range(args.boot)]
    out = {"pool": args.pool, "protocol": "GroupKFold(5) за гравцями; обидві моделі, норми внесків (17 тематичних ознак) і популяція правила відхилень – лише на навчальних гравцях блоку",
           "n_players": int(len(g17)), "n_splits": int(len(players_h)), "models": {}}
    kd_all = {}
    for k in MODELS:
        A, B = np.concatenate(half_A[k]), np.concatenate(half_B[k]); N = np.concatenate(named_full[k])
        kd = kd_all[k] = np.array([kappa(A[d], B[d], labels17) for d in draws])
        per_cat = {c: {"n": int((g17 == c).sum()), "agree": float(np.mean(N[g17 == c] == c)) if (g17 == c).any() else None}
                   for c in labels17}
        out["models"][k] = {"kappa": kappa(A, B, labels17), "kappa_ci95": [float(np.percentile(kd, 2.5)), float(np.percentile(kd, 97.5))],
                            "agree_ref17": float(np.mean(N == g17)), "per_category_agree_ref17": per_cat}
        print(f"{k:<12} κ = {out['models'][k]['kappa']:.3f} [{np.percentile(kd, 2.5):.3f}; {np.percentile(kd, 97.5):.3f}]"
              f"   збіг з 17-класовим еталоном {out['models'][k]['agree_ref17']:.3f}")
    # парна різниця: ті самі бутстреп-вибірки гравців для обох моделей
    dd = kd_all["monotone"] - kd_all["conditional"]
    out["kappa_diff_monotone_minus_conditional"] = {
        "diff": out["models"]["monotone"]["kappa"] - out["models"]["conditional"]["kappa"],
        "ci95": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]}
    print(f"різниця κ (монотонна − звичайна) {out['kappa_diff_monotone_minus_conditional']['diff']:+.3f} "
          f"[{np.percentile(dd, 2.5):+.3f}; {np.percentile(dd, 97.5):+.3f}] (парний бутстреп)")
    print("\nзбіг за категоріями еталона (звичайна → монотонна):")
    for c in labels17:
        a, b = out["models"]["conditional"]["per_category_agree_ref17"][c], out["models"]["monotone"]["per_category_agree_ref17"][c]
        if a["n"]:
            print(f"  {c:<22} n={a['n']:<5} {a['agree']:.3f} → {b['agree']:.3f}")

    ref = os.path.join(args.outdir, f"table_311_{args.pool}_oof.json")
    if os.path.exists(ref):
        r = json.load(open(ref))["rules"]["standardised17"]
        out["control_conditional_vs_validate_oof"] = {"kappa": [r["kappa"], out["models"]["conditional"]["kappa"]],
                                                     "agree_ref17": [r["agree_ref17"], out["models"]["conditional"]["agree_ref17"]]}
        print(f"\nконтроль (звичайна модель = крок 2 у validate_oof): κ {r['kappa']:.6f} / {out['models']['conditional']['kappa']:.6f}, "
              f"збіг {r['agree_ref17']:.6f} / {out['models']['conditional']['agree_ref17']:.6f}")
    path = os.path.join(args.outdir, f"monotone_{args.pool}_oof.json")
    json.dump(out, open(path, "w"), indent=1, ensure_ascii=False)
    print(f"->  {path}")


if __name__ == "__main__":
    main()
