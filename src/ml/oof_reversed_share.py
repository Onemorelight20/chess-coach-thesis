"""
Частка рекомендацій протилежного напряму («зворотних порад») ДО і ПІСЛЯ критерію
знака за протоколом без витоку даних за гравцями.

Навіщо окремий скрипт. У validate_oof.py зворотна порада перевіряється за
довідником сирих значень, побудованим лише для ознак C, що пройшли критерій знака
в блоці; ознаки поза C там не зараховуються. Для правила (2.7) це байдуже (воно
називає лише ознаки з C), але для правила без критерію знака (standardised17)
пропускаються саме шість ознак, які критерій вилучає і які дають зворотні поради,
тож його частка там занижена. Тут довідник будується для всіх 17 тематичних
ознак, решта протоколу та сама: функції імпортовано з validate_oof.

Протокол: GroupKFold(5) за гравцями; у кожному блоці змішана модель, пояснювач,
норми внесків, критерій знака й медіани сирих значень за інтервалами прогнозу —
лише на навчальних гравцях; рекомендація й перевірка — на відкладених гравцях за
всіма їхніми партіями. Зворотна порада: за названою ознакою гравець кращий за
медіану навчальних гравців свого інтервалу прогнозованого рейтингу (з урахуванням
орієнтації ознаки) — те саме означення, що в validate_oof і candidate_signs.

Контроль: частка для правила (2.7) має збігтися з reversed_share «standardised»
у results/rules/table_311_<pool>_oof.json.

Usage:
    python -m src.ml.oof_reversed_share --pool blitz --outdir results/rules
"""
import argparse, json, os, sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.features.extractor import FEATURE_NAMES, FEATURE_ORIENTATION
from scipy.stats import spearmanr
from src.ml.candidate_signs import sign_test, DISPUTED
from src.ml.validate_oof import T17, RNG, fit_fold, band_ref, band_index, rank_shap

RULES = {"themed": "крок 1: 17 тематичних ознак, без нормування",
         "standardised17": "крок 2: 17 ознак, нормування на рівень (до критерію знака)",
         "standardised": "крок 3: правило (2.7), критерій знака в блоці"}


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
    print(f"{args.pool}: {len(df)} партій, {groups.nunique()} гравців; GroupKFold(5) за гравцями\n")

    hits = {r: [] for r in RULES}; named_all = {r: [] for r in RULES}; players = []; fold_cands = []
    hits_emp = {r: [] for r in ("standardised17", "standardised")}; emp_signs = []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, groups), 1):
        m, expl = fit_fold(X, y, groups, tr)
        tr_agg = X.iloc[tr].groupby(groups.iloc[tr].values).mean()[FEATURE_NAMES]
        sv_tr = expl.shap_values(tr_agg); pred_tr = np.asarray(expl.expected_value) + sv_tr.sum(axis=1)
        C, _ = sign_test(tr_agg, sv_tr, FEATURE_NAMES, T17); fold_cands.append(C)
        cands = {"themed": T17, "standardised17": T17, "standardised": C}
        refs = {"themed": None,
                "standardised17": band_ref(sv_tr[:, [FEATURE_NAMES.index(c) for c in T17]], pred_tr),
                "standardised": band_ref(sv_tr[:, [FEATURE_NAMES.index(c) for c in C]], pred_tr)}
        rref = band_ref(tr_agg, pred_tr, T17)          # медіани сирих значень для ВСІХ 17 ознак
        # чутливість (як у candidate_signs): напрям двох спірних ознак – за знаком ρ(значення, Elo) навчальних гравців
        elo_tr = y.iloc[tr].groupby(groups.iloc[tr].values).median().loc[tr_agg.index]
        emp = orient.copy()
        for c in DISPUTED:
            emp[c] = -1 if spearmanr(tr_agg[c], elo_tr).statistic > 0 else 1
        emp_signs.append({c: int(emp[c]) for c in DISPUTED})

        Xte = X.iloc[te]
        agg_te = Xte.groupby(groups.iloc[te].values).mean()[FEATURE_NAMES]
        sva = expl.shap_values(agg_te); pra = np.asarray(expl.expected_value) + sva.sum(axis=1)
        bi = band_index(pra, rref)
        for r in RULES:
            named = rank_shap(sva, pra, cands[r], refs[r])[:, 0]
            h = [(agg_te[c].values[i] - rref["median"][bi[i], T17.index(c)]) * orient[c] < 0 for i, c in enumerate(named)]
            hits[r].append(np.array(h)); named_all[r].append(named)
            if r in hits_emp:
                he = [(agg_te[c].values[i] - rref["median"][bi[i], T17.index(c)]) * emp[c] < 0 for i, c in enumerate(named)]
                hits_emp[r].append(np.array(he))
        players.extend(agg_te.index.tolist())
        print(f"  блок {fold}: {len(agg_te)} відкладених гравців, {len(C)} кандидатів після критерію знака")

    rng = np.random.RandomState(RNG)
    out = {"pool": args.pool,
           "protocol": "GroupKFold(5) за гравцями; модель, норми внесків, критерій знака й медіани сирих значень – лише на навчальних гравцях блоку",
           "definition": "зворотна порада: за названою ознакою гравець кращий за медіану навчальних гравців свого інтервалу прогнозованого рейтингу",
           "n_players": len(players), "candidates_per_fold": fold_cands, "rules": {}}
    for r, label in RULES.items():
        H = np.concatenate(hits[r]).astype(float); N = np.concatenate(named_all[r])
        boot = [H[rng.randint(0, len(H), len(H))].mean() for _ in range(args.boot)]
        by_feat = pd.Series(N[H.astype(bool)]).value_counts().to_dict()
        out["rules"][r] = {"label": label, "reversed_share": float(H.mean()),
                           "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                           "reversed_by_feature": {str(k): int(v) for k, v in by_feat.items()}}
        print(f"{r:<16} {H.mean():.4f}  95% ДІ [{np.percentile(boot, 2.5):.4f}; {np.percentile(boot, 97.5):.4f}]  {label}")

    out["sensitivity_empirical_orientation"] = {"disputed_signs_per_fold": emp_signs}
    for r in hits_emp:
        He = np.concatenate(hits_emp[r]).astype(float)
        out["sensitivity_empirical_orientation"][r] = float(He.mean())
        print(f"чутливість (емпірична орієнтація спірних ознак) {r:<16} {He.mean():.4f}")
    ref = os.path.join(args.outdir, f"table_311_{args.pool}_oof.json")
    if os.path.exists(ref):
        old = json.load(open(ref))["rules"]["standardised"]["reversed_share"]
        out["control_standardised_vs_validate_oof"] = {"validate_oof": old, "here": out["rules"]["standardised"]["reversed_share"]}
        print(f"\nконтроль: правило (2.7) тут {out['rules']['standardised']['reversed_share']:.6f}, у validate_oof {old:.6f}")
    path = os.path.join(args.outdir, f"reversed_share_{args.pool}_oof.json")
    json.dump(out, open(path, "w"), indent=1, ensure_ascii=False)
    print(f"->  {path}")


if __name__ == "__main__":
    main()
