"""
Чому правило на внесках виявляє одні категорії і пропускає інші (підрозділ 3.3.4).

Для кожної синтетично погіршеної партії міряється, як після погіршення змінюються
прогноз моделі та внесок САМЕ цільової ознаки. Якщо внесок після погіршення зростає,
модель читає це погіршення як ознаку сильнішої гри – та сама інверсія знака, що
діагностована в підрозділі 3.3.5. Правило на внесках такі категорії пропускає за
побудовою, правило на сирих значеннях від цього не залежить.

Usage:
    python -m src.ml.prediction_response --cache results/synthetic_features_bands.json
"""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.ml import synthetic_validation as SV
from src.ml.model import ELOModel

NAME = {"opening_acpl": "дебютна точність", "middlegame_acpl": "точність у міттельшпілі",
        "endgame_acpl": "точність в ендшпілі", "conversion_acpl": "реалізація переваги",
        "defense_acpl": "стійкість у захисті", "blunder_rate": "грубі помилки",
        "hanging_rate": "висячі фігури"}
# прирости виявлення з compare_rules_synthetic.py (те саме джерело даних)
GAIN27 = {"opening_acpl": .42, "middlegame_acpl": .21, "endgame_acpl": .01, "conversion_acpl": -.05,
          "defense_acpl": -.14, "blunder_rate": -.05, "hanging_rate": -.15}
GAINRAW = {"opening_acpl": .31, "middlegame_acpl": .07, "endgame_acpl": .09, "conversion_acpl": .21,
           "defense_acpl": .00, "blunder_rate": .12, "hanging_rate": .10}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="results/synthetic_features_bands.json")
    ap.add_argument("--pool", default="blitz")
    ap.add_argument("--out", default="results/prediction_response_synthetic.json")
    a = ap.parse_args()

    recs = json.load(open(a.cache))["records"]
    m = ELOModel(pool=a.pool)
    res = {"n_games": len(recs), "per_category": {}}
    print(f"{'категорія':26s}{'Δ прогнозу, Elo':>17s}{'Δ внеску цільової':>20s}{'(2.7)':>9s}{'сире':>8s}")
    for c in SV.CATS:
        dp, dphi = [], []
        for r in recs:
            ob, ot = m.predict_and_explain(r["base_feats"]), m.predict_and_explain(r["treatments"][c]["feats"])
            dp.append(ot["predicted_elo"] - ob["predicted_elo"])
            dphi.append(ot["shap_values"].get(c, np.nan) - ob["shap_values"].get(c, np.nan))
        dp, dphi = np.array(dp), np.array(dphi)
        res["per_category"][c] = {"d_pred": float(dp.mean()), "share_pred_down": float((dp < 0).mean()),
                                  "d_shap_target": float(dphi.mean()),
                                  "inverted": bool(dphi.mean() > 0),
                                  "gain_rule27": GAIN27[c], "gain_rawfeat": GAINRAW[c]}
        print(f"{NAME[c]:26s}{dp.mean():+17.1f}{dphi.mean():+20.1f}{GAIN27[c]:+9.2f}{GAINRAW[c]:+8.2f}")

    from scipy.stats import spearmanr
    v = res["per_category"]
    x = np.array([v[c]["d_shap_target"] for c in SV.CATS])
    for key, lab in (("gain_rule27", "правило на внесках (2.7)"), ("gain_rawfeat", "сире правило")):
        y = np.array([v[c][key] for c in SV.CATS]); r = spearmanr(x, y)
        res.setdefault("spearman", {})[key] = {"rho": float(r.statistic), "p": float(r.pvalue)}
        print(f"  Δ внеску проти приросту, {lab}: rho = {r.statistic:+.2f}, p = {r.pvalue:.3f}")

    inv = np.array([v[c]["inverted"] for c in SV.CATS])
    for lab, mask in (("знак коректний", ~inv), ("знак інвертований", inv)):
        res.setdefault("by_sign", {})[lab] = {
            "categories": [c for c, mm in zip(SV.CATS, mask) if mm],
            "mean_gain_rule27": float(np.array([v[c]["gain_rule27"] for c in SV.CATS])[mask].mean()),
            "mean_gain_rawfeat": float(np.array([v[c]["gain_rawfeat"] for c in SV.CATS])[mask].mean())}
        b = res["by_sign"][lab]
        print(f"  {lab} ({len(b['categories'])} кат.): (2.7) {b['mean_gain_rule27']:+.2f} | сире {b['mean_gain_rawfeat']:+.2f}")

    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=2)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
