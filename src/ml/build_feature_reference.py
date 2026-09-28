"""
Довідник норм СИРИХ значень ознак за інтервалами прогнозованого рейтингу.

Аналог build_shap_reference.py, але статистики рахуються не для внесків SHAP,
а для самих значень ознак. Потрібен для контрольного правила «rawfeat»: те саме
нормування на рівень і той самий перелік кандидатів, лише без кроку пояснення
(підрозділ 3.3.3). Формат збігається з shap_reference_<pool>.pkl, тож
build_shap_reference.stats_at() читає обидва однаково.

Usage:
    python -m src.ml.build_feature_reference --pool blitz --parquet data/dataset_blitz_2k.parquet
"""
import argparse, os, sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.features.extractor import THEMED_CANDIDATES

REF_PATH_TMPL = "results/feature_reference_{pool}.pkl"


def build(parquet, pool, n_bands=8, candidates=None, out_path=None):
    from src.ml.model import ELOModel
    df = pd.read_parquet(parquet)
    df = df[df["target_elo"].between(400, 3000)]
    model = ELOModel(pool=pool)
    feats = model.feature_names

    agg = df.groupby("player")[feats].mean()          # той самий вектор, що будує застосунок
    pred = pd.Series(model.model.predict(agg), index=agg.index)
    cands = [c for c in (candidates or THEMED_CANDIDATES) if c in agg.columns]
    values = agg[cands]

    band = pd.qcut(pred, n_bands, labels=False, duplicates="drop")
    edges = [float(x) for x in pd.qcut(pred, n_bands, retbins=True, duplicates="drop")[1]]
    mean = values.groupby(band).mean()
    std = values.groupby(band).std(ddof=0).replace(0, np.nan)
    centres = pred.groupby(band).mean().reindex(mean.index)

    ref = {"pool": pool, "kind": "raw_features", "candidates": list(cands), "sign_excluded": [], "sign_test": {},
           "band_edges": edges, "band_centres": [float(x) for x in centres],
           "mean": mean, "std": std, "n_players": int(len(agg)), "n_bands": int(mean.shape[0])}
    path = out_path or REF_PATH_TMPL.format(pool=pool)
    joblib.dump(ref, path)
    print(f"{pool}: {len(agg)} гравців, {mean.shape[0]} інтервалів "
          f"[{edges[0]:.0f}…{edges[-1]:.0f}], {len(cands)} кандидатів -> {path}")
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="blitz")
    ap.add_argument("--parquet", default=None)
    ap.add_argument("--bands", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    build(a.parquet or f"data/dataset_{a.pool}_2k.parquet", a.pool, a.bands, out_path=a.out)


if __name__ == "__main__":
    main()
