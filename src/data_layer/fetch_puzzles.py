"""
Download the public Lichess puzzle database and build a compact local index for
absolute-rating puzzle selection (RecommenderEngine). The live
/api/puzzle/next endpoint only offers a difficulty relative to 1500; the
downloadable database carries an exact Rating per puzzle.

Usage:
    python -m src.data_layer.fetch_puzzles            # -> data/puzzles.parquet
"""

import argparse
import os
import subprocess

import pandas as pd

URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
COLS = ["PuzzleId", "Rating", "Themes", "Popularity", "NbPlays"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/puzzles.parquet")
    ap.add_argument("--keep-raw", action="store_true",
                    help="keep the downloaded .zst instead of deleting it")
    args = ap.parse_args()

    zst = "lichess_db_puzzle.csv.zst"
    if not os.path.exists(zst):
        print(f"Downloading {URL} (~300 MB)...", flush=True)
        subprocess.run(["curl", "-L", "--fail", "-o", zst, URL], check=True)

    print("Streaming + decompressing with zstd, reading needed columns...", flush=True)
    proc = subprocess.Popen(["zstd", "-dc", zst], stdout=subprocess.PIPE)
    df = pd.read_csv(proc.stdout, usecols=COLS)
    proc.wait()

    df = df.dropna(subset=["Rating", "Themes"])
    df["Rating"] = df["Rating"].astype("int32")
    df["Popularity"] = df["Popularity"].fillna(0).astype("int16")
    df.to_parquet(args.out, index=False)
    print(f"Wrote {len(df):,} puzzles -> {args.out} "
          f"(rating {int(df.Rating.min())}-{int(df.Rating.max())})", flush=True)

    if not args.keep_raw:
        os.remove(zst)
        print(f"Removed {zst}", flush=True)


if __name__ == "__main__":
    main()
