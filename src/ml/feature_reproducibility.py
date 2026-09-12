"""
Наскільки відтворювані самі ознаки — і скільки в них лишається після зняття рівня гри.

Два виміри:
  1) кореляція значення ознаки між двома половинами партій того самого гравця
     (набір 2k, половини по 4 партії — той самий обсяг, на якому міряється κ);
  2) те саме після вилучення лінійного впливу рейтингу (набір із 16 партій,
     половини по 8), тобто чи лишається індивідуальна структура, коли прибрати
     те, що правило (2.7) і так нормує.

Друге питання ключове: правило порівнює гравця з гравцями його рівня, тож усе,
що ознака знає лише про рівень, для нього марне.

Usage:
    python -m src.ml.feature_reproducibility --outdir results
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.features.extractor import FEATURE_NAMES, TRAINABLE_CANDIDATES

C = list(TRAINABLE_CANDIDATES)
SEED = 0


def halves(df, per_half):
    """Дві неперетинні половини по per_half партій на гравця."""
    X = df[FEATURE_NAMES].fillna(df[FEATURE_NAMES].median())
    A, B, elo = [], [], []
    need = per_half * 2
    for _, g in df.groupby("player"):
        idx = g.index.to_numpy()
        if len(idx) < need:
            continue
        sh = np.random.RandomState(SEED).permutation(idx)[:need]
        A.append(X.loc[sh[:per_half], C].mean())
        B.append(X.loc[sh[per_half:], C].mean())
        elo.append(float(g["target_elo"].iloc[0]))
    return pd.DataFrame(A), pd.DataFrame(B), np.asarray(elo)


def drop_level(M, elo):
    """Прибрати лінійний вплив рейтингу з кожної ознаки."""
    out = M.copy()
    for c in C:
        out[c] = M[c].values - np.polyval(np.polyfit(elo, M[c].values, 1), elo)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--base", default="data/dataset_blitz_2k.parquet")
    ap.add_argument("--wide", default="data/dataset_blitz_16g.parquet")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    out = {}

    for tag, path, per_half in (("n4", args.base, 4), ("n8", args.wide, 8)):
        if not os.path.exists(path):
            print(f"пропускаю {tag}: немає {path}")
            continue
        df = pd.read_parquet(path)
        df = df[df["target_elo"].between(400, 3000)].reset_index(drop=True)
        A, B, elo = halves(df, per_half)
        RA, RB = drop_level(A, elo), drop_level(B, elo)
        raw = {c: float(np.corrcoef(A[c], B[c])[0, 1]) for c in C}
        res = {c: float(np.corrcoef(RA[c], RB[c])[0, 1]) for c in C}
        out[tag] = {"n_players": len(A), "games_per_half": per_half,
                    "raw": raw, "residual": res,
                    "raw_range": [min(raw.values()), max(raw.values())],
                    "raw_mean": float(np.mean(list(raw.values()))),
                    "residual_mean": float(np.mean(list(res.values())))}
        print(f"\n=== {path}, половини по {per_half} партій, {len(A)} гравців ===")
        print(f"{'ознака':<24}{'сира r':>9}{'без рівня':>11}{'втрата':>9}")
        for c in sorted(C, key=lambda x: -(raw[x] - res[x])):
            print(f"  {c:<22}{raw[c]:9.3f}{res[c]:11.3f}{raw[c]-res[c]:9.3f}")
        print(f"  {'СЕРЕДНЄ':<22}{out[tag]['raw_mean']:9.3f}{out[tag]['residual_mean']:11.3f}")
        print(f"  діапазон сирих: {out[tag]['raw_range'][0]:.3f}–{out[tag]['raw_range'][1]:.3f}")

    path = os.path.join(args.outdir, "feature_reproducibility.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"\n->  {path}")


if __name__ == "__main__":
    main()
