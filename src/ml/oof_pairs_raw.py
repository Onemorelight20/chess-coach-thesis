"""
Позаблокові пари «еталон – відповідь» базового правила (raw) для матриці
невідповідностей (рисунок 3.8).

validate_oof.py зберігає для кожного правила лише зведення (частки відповідей,
збіг, κ), а матриці потрібні самі пари по гравцях. Цей скрипт повторює рівно
ту частину протоколу validate_oof.py, що стосується правила raw: GroupKFold(5)
за гравцями, у кожному блоці змішана модель і популяційні статистики правила
відхилень лише на навчальних гравцях, а еталон і відповідь – на відкладених.
Результат звіряється з answer_shares і agree правила raw у
results/rules/table_<pool>_oof.json, тож пари гарантовано ті самі.

Usage:
    python -m src.ml.oof_pairs_raw --pool blitz
"""
import argparse, json, os, sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.features.extractor import FEATURE_NAMES, FEATURE_ORIENTATION
from src.ml.validate_oof import ALL29, fit_fold, rank_shap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="blitz")
    ap.add_argument("--outdir", default="results/rules")
    args = ap.parse_args()

    df = pd.read_parquet(f"data/dataset_{args.pool}_2k.parquet")
    df = df[df["target_elo"].between(400, 3000)].reset_index(drop=True)
    X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(df[FEATURE_NAMES]), columns=FEATURE_NAMES)
    y, groups = df["target_elo"], df["player"]
    orient = pd.Series(FEATURE_ORIENTATION)

    rows = []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, groups), 1):
        m, expl = fit_fold(X, y, groups, tr)
        tr_pm = X.iloc[tr].groupby(groups.iloc[tr].values)[ALL29].mean()
        mu_p, sd_p = tr_pm.mean(), tr_pm.std(ddof=0)
        Xte = X.iloc[te]
        te_pm = Xte.groupby(groups.iloc[te].values)[ALL29].mean()
        z_te = ((te_pm - mu_p) / sd_p).mul(orient[ALL29])
        agg_te = Xte.groupby(groups.iloc[te].values).mean()[FEATURE_NAMES].loc[te_pm.index]
        sva = expl.shap_values(agg_te); pra = np.asarray(expl.expected_value) + sva.sum(axis=1)
        gt = z_te[ALL29].idxmax(axis=1)
        pred = rank_shap(sva, pra, ALL29, None)[:, 0]
        rows += [{"player": p, "fold": fold, "gt_weak": g, "pred_weak": q} for p, g, q in zip(te_pm.index, gt.values, pred)]
        print(f"  блок {fold}: {len(te_pm)} відкладених гравців")

    out = pd.DataFrame(rows)
    shares = out["pred_weak"].value_counts(normalize=True)
    agree = float((out["gt_weak"] == out["pred_weak"]).mean())
    ref = json.load(open(os.path.join(args.outdir, f"table_311_{args.pool}_oof.json")))["rules"]["raw"]
    ref_shares = pd.Series(ref["answer_shares"])
    diff = (shares.reindex(ref_shares.index, fill_value=0) - ref_shares).abs().max()
    print(f"гравців {len(out)}; збіг {agree:.4f} (у JSON {ref['agree']:.4f}); категорій у відповідях {shares.size} "
          f"(у JSON {ref_shares.size}); найбільша розбіжність часток {diff:.2e}")
    assert len(shares) == len(ref_shares) and diff < 1e-12 and abs(agree - ref["agree"]) < 1e-12, \
        "пари не збігаються з table_311_*_oof.json"
    path = os.path.join(args.outdir, f"xai_groundtruth_vs_pred_{args.pool}_raw_oof.csv")
    out.to_csv(path, index=False)
    print(f"->  {path}")


if __name__ == "__main__":
    main()
