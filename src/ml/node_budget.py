"""
Підбір фіксованого бюджету вузлів рушія для застосунку замість ліміту часу 0,05 с.

Навчальні ознаки пораховано з лімітом 0,05 с на позицію на cpu_count() − 1 паралельних
процесах (src/data_layer/collector.py), тож кількість вузлів на позицію там залежала від
ядра й навантаження. Ліміт часу робить аналіз неповторюваним: ті самі партії дають інші
ознаки й інший прогноз. Фіксована кількість вузлів на одному потоці з очищеним хешем дає
той самий результат щоразу.

Тут для стратифікованої за рейтингом вибірки навчальних партій (game_id = зміщення партії
в дампі PGN; гравець і рейтинг звіряються із заголовком) ознаки перераховуються тим самим
кодом, що й у колекторі: спершу з лімітом часу 0,05 с за тих самих умов (стільки ж
паралельних процесів), потім із кількома бюджетами вузлів на очищеному хеші. Кожен
варіант зіставляється зі збереженими в наборі ознаками: зсув кожної ознаки в її
стандартних відхиленнях і зсув прогнозу моделі відповідного формату. Оптимальний бюджет –
той, за якого прогноз і ознаки найменше відхиляються від навчальних.

Usage:
    python -m src.ml.node_budget --pools blitz rapid --games 120 --outdir results/node_budget
"""
import argparse, json, os, sys, time
from multiprocessing import Pool, cpu_count

import chess
import chess.engine
import chess.pgn
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.engine.analyzer import EngineAnalyzer
from src.features.extractor import FeatureExtractor, FEATURE_NAMES
from src.data_layer.collector import parse_increment

PGN = "lichess_db_standard_rated_2026-01.pgn"
ENGINE_INDEPENDENT = ("avg_think_time", "think_time_consistency", "time_pressure_rate")   # з годинника, не з рушія
_W = {}


def _init(pgn_path, budgets):
    _W["fh"] = open(pgn_path, "r", encoding="utf-8")
    _W["a"] = EngineAnalyzer(time_limit=0.05)          # як у колекторі: MultiPV 5, обрізання оцінки 1000
    _W["budgets"] = budgets


def _features(res, color, inc):
    return FeatureExtractor(target_player_color=color).extract(res, increment=inc)


def _task(t):
    pool, offset, color, player, elo, tc = t
    fh, a = _W["fh"], _W["a"]
    key = dict(pool=pool, game_id=offset, target_color=color)
    try:
        fh.seek(offset)
        g = chess.pgn.read_game(fh)
        h = g.headers if g is not None else {}
        side = "White" if color == "white" else "Black"
        if g is None or h.get(side) != player or str(h.get(side + "Elo")) != str(int(elo)):
            return [dict(key, config="mismatch")]
        inc = parse_increment(tc)
        rows = []
        # 1) ліміт часу, як у колекторі (хеш між партіями не очищується)
        a._limit = lambda: chess.engine.Limit(time=0.05)
        t0 = time.time(); res = a.analyze_game(g)
        rows.append(dict(key, **_features(res, color, inc), config="time", seconds=time.time() - t0))
        # 2) бюджети вузлів, кожен на очищеному хеші, як робитиме застосунок
        for n in _W["budgets"]:
            a.engine.configure({"Clear Hash": None})
            a._limit = (lambda n=n: chess.engine.Limit(nodes=n))
            t0 = time.time(); res = a.analyze_game(g)
            rows.append(dict(key, **_features(res, color, inc), config=str(n), seconds=time.time() - t0))
        return rows
    except Exception as e:  # noqa: BLE001 – одна погана партія не має зупиняти пул
        print(f"[worker] {pool} {offset}: {e}", file=sys.stderr, flush=True)
        return [dict(key, config="error")]


def sample(pools, n_games, seed=0):
    tasks, stored = [], []
    for p in pools:
        df = pd.read_parquet(f"data/dataset_{p}_2k.parquet")
        df = df[df["target_elo"].between(400, 3000)].reset_index(drop=True)
        bins = pd.cut(df["target_elo"], bins=list(range(400, 2801, 200)) + [3001], right=False)
        per = max(1, n_games // bins.nunique())
        s = (df.groupby(bins, observed=True, group_keys=False)
               .apply(lambda g: g.sample(min(per, len(g)), random_state=seed)))
        s = s.assign(pool=p)
        stored.append(s)
        tasks += [(p, int(r.game_id), r.target_color, r.player, int(r.target_elo), r.time_control) for r in s.itertuples()]
    return tasks, pd.concat(stored, ignore_index=True)


def summarise(rec, stored, pools):
    from src.ml.model import ELOModel
    eng_feats = [f for f in FEATURE_NAMES if f not in ENGINE_INDEPENDENT]
    out = {}
    key = ["pool", "game_id", "target_color"]
    for p in pools + ["усі"]:
        S = stored if p == "усі" else stored[stored["pool"] == p]
        R = rec if p == "усі" else rec[rec["pool"] == p]
        res = {}
        for cfg in R["config"].unique():
            r = R[R["config"] == cfg].merge(S[key + FEATURE_NAMES], on=key, suffixes=("", "_st"))
            preds_r, preds_s = [], []
            for q in r["pool"].unique():
                m = ELOModel(pool=q).model
                rq = r[r["pool"] == q]
                preds_r.append(pd.Series(m.predict(rq[FEATURE_NAMES]), index=rq.index))
                preds_s.append(pd.Series(m.predict(rq[[f + "_st" for f in FEATURE_NAMES]].set_axis(FEATURE_NAMES, axis=1)), index=rq.index))
            dp = (pd.concat(preds_r) - pd.concat(preds_s)).loc[r.index]
            sd = pd.concat([pd.read_parquet(f"data/dataset_{q}_2k.parquet")[FEATURE_NAMES] for q in r["pool"].unique()]).std()
            z = pd.DataFrame({f: (r[f] - r[f + "_st"]) / sd[f] for f in eng_feats})
            rng = np.random.default_rng(0); v = dp.to_numpy()
            boot = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)]
            res[cfg] = {"n": int(len(r)), "seconds_per_game": float(r["seconds"].mean()),
                        "pred_shift_mean": float(dp.mean()), "pred_shift_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                        "pred_shift_mad": float(dp.abs().mean()),
                        "feat_bias_mean_abs_sd": float(z.mean().abs().mean()), "feat_mad_sd": float(z.abs().mean().mean()),
                        "feat_bias_sd": {f: float(z[f].mean()) for f in eng_feats}}
        out[p] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", nargs="+", default=["blitz", "rapid"])
    ap.add_argument("--games", type=int, default=120, help="партій на формат (порівну за інтервалами рейтингу по 200)")
    ap.add_argument("--budgets", nargs="+", type=int, default=[15000, 25000, 40000, 60000, 90000])
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 1), help="як у колекторі: cpu_count() − 1")
    ap.add_argument("--outdir", default="results/node_budget")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    tasks, stored = sample(args.pools, args.games)
    print(f"{len(tasks)} партій ({', '.join(args.pools)}), {args.workers} процесів, бюджети {args.budgets}", flush=True)
    rows, t0 = [], time.time()
    with Pool(args.workers, initializer=_init, initargs=(PGN, args.budgets)) as pool:
        for i, rr in enumerate(pool.imap_unordered(_task, tasks), 1):
            rows += rr
            if i % 20 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} партій, {time.time() - t0:.0f} с", flush=True)
    rec = pd.DataFrame(rows)
    rec.to_parquet(os.path.join(args.outdir, "node_budget_features.parquet"))
    stored.to_parquet(os.path.join(args.outdir, "node_budget_stored.parquet"))
    bad = rec[rec["config"].isin(["mismatch", "error"])]
    rec = rec[~rec["config"].isin(["mismatch", "error"])]
    summary = summarise(rec, stored, args.pools)
    summary["_meta"] = {"pgn": PGN, "workers": args.workers, "budgets": args.budgets, "n_tasks": len(tasks),
                        "n_mismatch_or_error": int(len(bad)), "engine": "Stockfish, Threads 1, Hash 16, MultiPV 5"}
    json.dump(summary, open(os.path.join(args.outdir, "node_budget_summary.json"), "w"), ensure_ascii=False, indent=1)
    print(f"\nне знайдено / помилки: {len(bad)}")
    for p, res in summary.items():
        if p.startswith("_"): continue
        print(f"\n== {p} ==\n{'варіант':>8} {'n':>4} {'с/партію':>9} {'зсув прогнозу':>14} {'95% ДІ':>18} {'|зсув| сер.':>11} {'|зсув ознак|, σ':>15} {'розкид ознак, σ':>15}")
        for cfg in ["time"] + [str(b) for b in args.budgets]:
            if cfg not in res: continue
            v = res[cfg]
            print(f"{cfg:>8} {v['n']:>4} {v['seconds_per_game']:>9.1f} {v['pred_shift_mean']:>+14.1f} "
                  f"[{v['pred_shift_ci95'][0]:+.1f}; {v['pred_shift_ci95'][1]:+.1f}] {v['pred_shift_mad']:>11.1f} "
                  f"{v['feat_bias_mean_abs_sd']:>15.3f} {v['feat_mad_sd']:>15.3f}")
    print(f"-> {args.outdir}/node_budget_summary.json")


if __name__ == "__main__":
    main()
