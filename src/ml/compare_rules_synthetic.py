"""
Парне порівняння трьох правил на ОДНИХ І ТИХ САМИХ синтетично погіршених партіях
(підрозділ 3.3.4). Відповідає на питання, чи потрібен крок пояснення: правило
«rawfeat» застосовує той самий перелік кандидатів і те саме нормування на рівень
до сирих значень ознак, без внесків SHAP.

Приріст виявлення рахується парно: (виявлено в погіршеній партії) − (та сама
відповідь у чистій партії), тож обидва правила бачать однаковий матеріал, а
довірчі інтервали дає бутстреп по партіях.

Usage:
    python -m src.ml.compare_rules_synthetic --cache results/synthetic_features_bands.json
"""
import argparse, json, os, sys

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.ml import synthetic_validation as SV
from src.ml.model import ELOModel
from src.features.extractor import SIGN_EXCLUDED

NAME = {"opening_acpl": "дебютна точність", "middlegame_acpl": "точність у міттельшпілі",
        "endgame_acpl": "точність в ендшпілі", "conversion_acpl": "реалізація переваги",
        "defense_acpl": "стійкість у захисті", "blunder_rate": "грубі помилки",
        "hanging_rate": "висячі фігури"}
RULES = ("raw", "standardised", "rawfeat")


def deltas(model, recs, rule, shap_ref, feat_ref):
    """Парний приріст першої відповіді по кожній партії для одного правила."""
    SV.RULE = rule
    model._shap_reference = shap_ref if rule == "standardised" else None
    model._feature_reference = feat_ref if rule == "rawfeat" else None
    out = {}
    for c in SV.CATS: out[c] = []
    for r in recs:
        base = SV.rank_categories(model, r["base_feats"])[0]
        for c in SV.CATS:
            got = SV.rank_categories(model, r["treatments"][c]["feats"])[0]
            out[c].append(int(got == c) - int(base == c))
    return {c: np.array(v) for c, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="results/synthetic_features_bands.json")
    ap.add_argument("--pool", default="blitz")
    ap.add_argument("--shap-reference", default="src/ml/models/shap_reference_blitz_nosign.pkl",
                    help="довідник внесків БЕЗ критерію знака: у цьому експерименті всі сім категорій мають лишатися кандидатами")
    ap.add_argument("--feature-reference", default="results/feature_reference_blitz.pkl")
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--out", default="results/rules_synthetic_comparison.json")
    a = ap.parse_args()

    recs = json.load(open(a.cache))["records"]
    model = ELOModel(pool=a.pool)
    D = {r: deltas(model, recs, r, joblib.load(a.shap_reference), joblib.load(a.feature_reference)) for r in RULES}

    rng = np.random.default_rng(0); n = len(recs)
    idx = rng.integers(0, n, size=(a.boot, n))
    ci = lambda v: [float(np.quantile(v[idx].mean(1), .025)), float(np.quantile(v[idx].mean(1), .975))]

    res = {"n_games": n, "cache": a.cache, "per_category": {}, "overall": {}}
    print(f"{'категорія':25s}" + "".join(f"{r:>22s}" for r in RULES) + f"{'rawfeat − (2.7)':>26s}")
    for c, nm in NAME.items():
        row = f"{nm:25s}"; rec = {}
        for r in RULES:
            d = D[r][c]; lo, hi = ci(d); rec[r] = {"delta": float(d.mean()), "ci95": [lo, hi]}
            row += f"{d.mean():+8.2f} [{lo:+.2f};{hi:+.2f}]"
        diff = D["rawfeat"][c] - D["standardised"][c]; lo, hi = ci(diff)
        rec["rawfeat_minus_rule27"] = {"delta": float(diff.mean()), "ci95": [lo, hi], "significant": bool(lo > 0 or hi < 0)}
        row += f"{diff.mean():+9.2f} [{lo:+.2f};{hi:+.2f}]" + ("*" if lo > 0 or hi < 0 else " ")
        print(row); res["per_category"][c] = rec

    # Розклад за переліком кандидатів: три категорії проходять критерій знака в бліці,
    # чотири виключені. Правило (2.7) їх не називає за побудовою, тож саме там і лежить
    # уся перевага контрольного правила — це і є головний зріз для тлумачення.
    T = [c for c in SV.CATS if c not in SIGN_EXCLUDED]
    NT = [c for c in SV.CATS if c in SIGN_EXCLUDED]
    for lab, group in (("усі сім", SV.CATS), ("три в переліку T", T), ("чотири поза T", NT)):
        print(f"\n{lab} ({len(group)}): {', '.join(NAME[c] for c in group)}")
        block = {}
        for r in RULES:
            v = np.stack([D[r][c] for c in group]).mean(0); lo, hi = ci(v)
            block[r] = {"delta": float(v.mean()), "ci95": [lo, hi],
                        "categories_positive": int(sum(1 for c in group if D[r][c].mean() > 0))}
            print(f"  {r:14s} {v.mean():+.3f} [{lo:+.3f}; {hi:+.3f}]   додатних: {block[r]['categories_positive']}/{len(group)}")
        for lhs, rhs, key in (("rawfeat", "standardised", "rawfeat_minus_rule27"),
                              ("rawfeat", "raw", "rawfeat_minus_naive")):
            v = np.stack([D[lhs][c] - D[rhs][c] for c in group]).mean(0); lo, hi = ci(v)
            block[key] = {"delta": float(v.mean()), "ci95": [lo, hi], "significant": bool(lo > 0 or hi < 0)}
            print(f"  {lhs} − {rhs:12s} {v.mean():+.3f} [{lo:+.3f}; {hi:+.3f}] {'значуще' if lo > 0 or hi < 0 else 'НЕ значуще'}")
        res["overall"][lab] = block

    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=2)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
