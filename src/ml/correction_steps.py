"""
Таблиця 3.11: внесок кожного з трьох виправлень правила (2.7), одним прогоном.

Чотири варіанти правила рахуються на ОДНІЙ навченій моделі й тих самих
розбиттях (протокол validate_xai), тому стовпці таблиці порівнянні між собою і
з таблицями 3.17–3.18, які використовують ту саму модель:

    naive          — argmin сирого внеску серед усіх 29 ознак;
    themed         — лише 17 ознак із тренувальною темою;
    standardised17 — ті самі 17 + нормування внеску на рівень гравця;
    standardised   — розгорнуте правило: 17 → критерій знака → нормування
                     (перелік кандидатів читається з shap_reference_<pool>.pkl).

Збіг із розрахунком за відхиленнями подано двічі: з еталоном над власним
переліком кандидатів правила і з ФІКСОВАНИМ 17-класовим еталоном. Другий
потрібен, бо звуження переліку перевизначає «правильну відповідь» для гравців,
чия найслабша ознака з нього випала.

Usage:
    python -m src.ml.correction_steps --pool blitz --outdir results/rules
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.features.extractor import FEATURE_NAMES
from src.ml import validate_xai as vx

LABELS = {"raw": "наївне правило", "themed": "+ обмеження кандидатів",
          "standardised17": "+ нормування на рівень", "standardised": "+ фільтр знака"}


def row(res):
    a, b = res["experiment_a"], res["experiment_b"]
    r = b["ref17"]
    return {
        "label": LABELS[res["meta"]["rule"]],
        "n_cand": len(res["meta"]["candidates"]),
        "chance": 1.0 / len(res["meta"]["candidates"]),
        "trivial": b["majority_baseline"],
        "tr_agree": a["split_half_agreement"],
        "kappa": a["cohen_kappa"], "kappa_ci95": a["kappa_ci95"],
        "kappa_modal": a["modal_comparison"]["cohen_kappa"],
        "ov3": a["topk_overlap"]["3"], "ov3_chance": vx_ov3_chance(len(res["meta"]["candidates"])),
        "agree": b["top1_accuracy"], "agree3": b["coverage_at_k"]["3"],
        "kappa_agree": b["cohen_kappa"],
        "agree_ref17": r["top1_accuracy"], "agree3_ref17": r["coverage_at_k"]["3"],
        "trivial_ref17": r["majority_baseline"], "kappa_agree_ref17": r["cohen_kappa"],
        "share_outside": r["share_outside_candidates"],
    }


def vx_ov3_chance(n):
    from math import comb
    return 1 - comb(n - 3, 3) / comb(n, 3) if n >= 6 else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="blitz", choices=["bullet", "blitz", "rapid", "classical"])
    ap.add_argument("--outdir", default="results/rules")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--rules", default=",".join(vx.RULES))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_parquet(f"data/dataset_{args.pool}_2k.parquet")
    df = df[df["target_elo"].between(400, 3000)].reset_index(drop=True)
    print(f"{args.pool}: {len(df):,} партій, {df['player'].nunique()} гравців\n")
    model, explainer, Xi = vx.fit_model_and_explainer(df[FEATURE_NAMES], df["target_elo"], df["player"])

    table = {}
    for rule in args.rules.split(","):
        res = vx.run_rule(df, Xi, explainer, rule, args.outdir, f"_{args.pool}_{rule}",
                          args.pool, args.boot)
        table[rule] = row(res)

    # top_share: найчастіша відповідь механізму (атрактори), з CSV кожного правила
    for rule in table:
        csv = os.path.join(args.outdir, f"xai_groundtruth_vs_pred_{args.pool}_{rule}.csv")
        pred = pd.read_csv(csv)["pred_weak"]
        table[rule]["top_share"] = float(pred.value_counts(normalize=True).iloc[0])
        table[rule]["top_name"] = str(pred.value_counts().index[0])

    path = os.path.join(args.outdir, f"table_311_{args.pool}.json")
    json.dump(table, open(path, "w"), indent=1, ensure_ascii=False)
    cols = ["n_cand", "chance", "trivial", "tr_agree", "kappa", "ov3", "ov3_chance",
            "agree", "agree3", "kappa_agree", "agree_ref17", "agree3_ref17", "kappa_agree_ref17",
            "trivial_ref17", "share_outside", "top_share"]
    print(f"{'показник':<20}" + "".join(f"{LABELS[r][:22]:>24}" for r in table))
    for c in cols:
        vals = [table[r][c] for r in table]
        fmt = (lambda v: f"{v:>24d}") if c == "n_cand" else (lambda v: f"{v:>24.3f}")
        print(f"{c:<20}" + "".join(fmt(v) for v in vals))
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
