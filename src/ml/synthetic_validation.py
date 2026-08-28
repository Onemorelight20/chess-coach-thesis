"""
Synthetic-weakness validation: a positive control for the SHAP weakness detector.
No public dataset labels a player's true weakness, so one is manufactured and the
detector is asked to recover it.

Controlled, paired and localised: the baseline is an untouched high-Elo game, the
treatment is the SAME game with the player's moves replaced by engine-verified
inferior ones in positions of ONE category only. One treatment per category gives
an injected x detected confusion matrix.

Usage:
    python -m src.ml.synthetic_validation lichess_db_standard_rated_2026-01.pgn \
        --games 60 --min-elo 2000 --event blitz --depth 12 --outdir results
"""

import argparse
import json
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import chess
import chess.pgn

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.engine.analyzer import EngineAnalyzer
from src.features.extractor import (
    FeatureExtractor, DISPLAY_NAMES,
    OPENING_MIN_PIECES, ENDGAME_MAX_PIECES, WINNING_CP, LOSING_CP,
)
from src.ml.model import ELOModel
from src.ml import plotstyle
plotstyle.apply()

# The categories we can inject and therefore test (the trainable-weakness subset).
CATS = ["opening_acpl", "middlegame_acpl", "endgame_acpl",
        "conversion_acpl", "defense_acpl", "blunder_rate", "hanging_rate"]


def _eligible(cat: str, pieces: int, peval: float) -> bool:
    """Does a target-player ply belong to the category we want to degrade?"""
    if cat == "opening_acpl":
        return pieces >= OPENING_MIN_PIECES
    if cat == "middlegame_acpl":
        return ENDGAME_MAX_PIECES < pieces < OPENING_MIN_PIECES
    if cat == "endgame_acpl":
        return pieces <= ENDGAME_MAX_PIECES
    if cat == "conversion_acpl":
        return peval > WINNING_CP
    if cat == "defense_acpl":
        return peval < LOSING_CP
    # blunder_rate / hanging_rate: any move can be made to hang / blunder
    return True


def _pick_substitute(cands, board, mover, cat, rng, mode, cp_lo, cp_hi, wp_lo, wp_hi):
    """
    Choose an engine-verified inferior move appropriate to the category, from the
    already-computed candidates `cands` -- (move, cp_loss, wp_loss) triples
    ordered best-to-worst, so cands[0] is engine-rank #2. Returns the whole
    triple, so the caller gets rank and cp_loss without a second engine call.
    """
    if not cands:
        return None

    if cat == "hanging_rate":
        # Prefer a move that actually leaves one of the mover's pieces en prise.
        best_c, best_h = None, 0
        for c in cands:
            tmp = board.copy(stack=False)
            tmp.push(c[0])
            hanging = EngineAnalyzer._positional_signals(tmp, mover)[0]
            if hanging > best_h:
                best_c, best_h = c, hanging
        return best_c  # may be None if no move increases hanging

    if cat == "blunder_rate":
        # Genuine blunder: worst available in winprob mode, large cp drop otherwise.
        if mode == "winprob":
            pool = [c for c in cands if c[2] >= wp_lo]
        else:
            pool = [c for c in cands if c[1] >= 300]
        pool = pool or list(cands)
        return max(pool, key=lambda c: c[1]) if pool else None

    if mode == "winprob":
        band = [c for c in cands if wp_lo <= c[2] <= wp_hi]
    else:
        band = [c for c in cands if cp_lo <= c[1] <= cp_hi]
    pool = band if band else list(cands)
    return rng.choice(pool) if pool else None


def build_treatment(analyzer, game, color, clean, initial_eval, cat, rate, rng,
                    mode, cp_lo, cp_hi, wp_lo, wp_hi):
    """
    Return (modified per-move dicts, n_injected). Overrides ONLY the substituted
    plies, so non-target categories stay clean. cpl / move_rank / is_critical for
    an injected ply come from the same candidate list used to pick the
    substitute, exactly how analyze_game scores every unmodified ply.
    """
    out = [dict(d) for d in clean]
    board = game.board()
    target_white = color == "white"
    n_injected = 0
    idx = 0
    for move_node in game.mainline():
        move = move_node.move
        mover = board.turn
        prev_eval = initial_eval if idx == 0 else clean[idx - 1]["eval_cp"]
        is_target = (mover == chess.WHITE) == target_white
        peval = prev_eval if mover == chess.WHITE else -prev_eval
        pieces = len(board.piece_map())

        if (is_target and _eligible(cat, pieces, peval) and rng.random() < rate):
            cands = analyzer.worse_move_candidates(board, mover)
            # cands[0]'s loss is the rank-1-to-rank-2 gap: same test as analyze_game.
            is_critical = bool(cands) and cands[0][1] >= 100
            chosen = _pick_substitute(cands, board, mover, cat, rng,
                                      mode, cp_lo, cp_hi, wp_lo, wp_hi)
            if chosen is not None:
                sub, cp_loss, _wp_loss = chosen
                rank = cands.index(chosen) + 2  # +2: cands starts at engine-rank #2
                tmp = board.copy(stack=False)
                tmp.push(sub)
                inj_eval = analyzer._capped_white_cp(
                    analyzer.engine.analyse(tmp, analyzer._limit()))
                h, p, kd, cc, mat = analyzer._positional_signals(tmp, mover)
                out[idx].update(eval_cp=inj_eval, cpl=cp_loss, move_rank=rank,
                                is_critical=is_critical,
                                is_capture=board.is_capture(sub),
                                is_check=tmp.is_check(),
                                piece_count=len(tmp.piece_map()),
                                hanging=h, pins=p, king_danger=kd,
                                center_control=cc, material=mat)
                n_injected += 1
        board.push(move)
        idx += 1
    return out, n_injected


def rank_categories(model, features):
    """Injectable categories ordered from most to least weakening (argmin first)."""
    shap = model.predict_and_explain(features)["shap_values"]
    cands = [c for c in CATS if c in shap]
    return sorted(cands, key=lambda c: shap[c])


def detect(model, features):
    """Detector's chosen weakness, argmin SHAP restricted to the injectable set."""
    return rank_categories(model, features)[0]


def parse_bands(spec):
    """'1000-1400,1400-1800' -> [(1000, 1400), (1400, 1800)]."""
    out = []
    for part in spec.split(","):
        lo, hi = part.split("-")
        out.append((int(lo), int(hi)))
    return out


def stream_games(path, n, min_elo, event, bands=None):
    """Yield (game, target_colour, target_elo, band). The target is the stronger
    side, so its Elo decides the band. With `bands`, fill an equal quota per band
    instead of taking the first n games above min_elo."""
    quota = None if bands is None else max(1, n // len(bands))
    taken = {b: 0 for b in (bands or [])}
    games = []
    with open(path, encoding="utf-8") as f:
        while len(games) < n:
            g = chess.pgn.read_game(f)
            if g is None:
                break
            h = g.headers
            if event and event.lower() not in h.get("Event", "").lower():
                continue
            we, be = h.get("WhiteElo", "?"), h.get("BlackElo", "?")
            if not (we.isdigit() and be.isdigit()):
                continue
            we, be = int(we), int(be)
            elo = max(we, be)                      # the target player's rating
            if sum(1 for _ in g.mainline_moves()) < 24:
                continue
            if bands is None:
                if elo < min_elo:
                    continue
                band = None
            else:
                band = next((b for b in bands if b[0] <= elo < b[1]), None)
                if band is None or taken[band] >= quota:
                    continue
                taken[band] += 1
            games.append((g, "white" if we >= be else "black", elo, band))
    return games


def bootstrap_ci(hits_by_game, iters=2000, seed=0):
    rng = np.random.default_rng(seed)
    games = list(hits_by_game)
    if not games:
        return (0.0, 0.0)
    means = []
    for _ in range(iters):
        sample = rng.choice(len(games), len(games), replace=True)
        flat = [h for i in sample for h in games[i]]
        if flat:
            means.append(np.mean(flat))
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn")
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--min-elo", type=int, default=2000)
    ap.add_argument("--event", default="blitz")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--rate", type=float, default=0.7,
                    help="fraction of eligible plies degraded per treatment")
    ap.add_argument("--magnitude", choices=["winprob", "cp"], default="winprob",
                    help="how to size a planted error: Maia win-prob loss, or centipawn band")
    ap.add_argument("--wp-lo", type=float, default=0.10,
                    help="min win-prob loss for a planted error (Maia blunder = 0.10)")
    ap.add_argument("--wp-hi", type=float, default=0.30,
                    help="max win-prob loss for non-blunder categories (keeps them sub-blunder)")
    ap.add_argument("--loss-lo", type=int, default=120, help="cp-mode min centipawn loss")
    ap.add_argument("--loss-hi", type=int, default=400, help="cp-mode max centipawn loss")
    ap.add_argument("--elo-bands", default=None,
                    help="stratify the substrate, e.g. '1000-1400,1400-1800,1800-2200,2200-3000'; "
                         "an equal quota of games is taken per band and sensitivity is also "
                         "reported per band. Overrides --min-elo.")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--cats", default=None,
                    help="comma-separated subset of CATS to inject (default: all seven)")
    ap.add_argument("--tag", default="", help="suffix for the output files")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rng = random.Random(args.seed)
    bands = parse_bands(args.elo_bands) if args.elo_bands else None
    cats = [c.strip() for c in args.cats.split(",")] if args.cats else list(CATS)
    unknown = [c for c in cats if c not in CATS]
    assert not unknown, f"unknown categories: {unknown}"

    scope = (f"bands {args.elo_bands}" if bands else f"Elo >= {args.min_elo}")
    print(f"Streaming up to {args.games} clean '{args.event}' games ({scope})...")
    games = stream_games(args.pgn, args.games, args.min_elo, args.event, bands)
    print(f"  collected {len(games)} games.")
    if bands:
        for b in bands:
            got = sum(1 for g in games if g[3] == b)
            print(f"    {b[0]}-{b[1]}: {got}")
    print()
    if not games:
        print("No games matched. Lower --min-elo or change --event.")
        return

    analyzer = EngineAnalyzer(depth=args.depth)
    model = ELOModel(pool=args.event)  # pool-specific model, matching the injected games' time control

    # rows: injected subset; columns: everything the detector may answer (all seven)
    confusion = {gt: {pred: 0 for pred in CATS} for gt in cats}
    hits_by_game = {c: [] for c in cats}          # for bootstrap CI
    top2_hits = {c: [] for c in cats}
    manip = {c: [] for c in cats}                  # injected-feature delta
    n_inj = {c: [] for c in cats}                  # plies actually substituted
    baseline_detect = {c: 0 for c in CATS}         # specificity / null profile

    by_band = {} if bands is None else {b: {c: [] for c in cats} for b in bands}
    # rank of the injected category, in the treated game and in the untouched one:
    # together they give coverage@k against the detector's own prior.
    rank_treated = {c: [] for c in cats}
    rank_clean = {c: [] for c in cats}

    for gi, (game, color, elo, band) in enumerate(games, 1):
        clean = analyzer.analyze_game(game)
        initial_eval = analyzer._capped_white_cp(
            analyzer.engine.analyse(game.board(), analyzer._limit()))
        base_feats = FeatureExtractor(color).extract(clean)
        base_rank = rank_categories(model, base_feats)
        baseline_detect[base_rank[0]] += 1

        for cat in cats:
            treat, k = build_treatment(analyzer, game, color, clean, initial_eval,
                                       cat, args.rate, rng, args.magnitude,
                                       args.loss_lo, args.loss_hi, args.wp_lo, args.wp_hi)
            feats = FeatureExtractor(color).extract(treat)
            ranked = rank_categories(model, feats)
            pred = ranked[0]
            rank_treated[cat].append(ranked.index(cat) + 1)
            rank_clean[cat].append(base_rank.index(cat) + 1)
            confusion[cat][pred] += 1
            hits_by_game[cat].append(int(pred == cat))
            # top-2: re-rank candidates by SHAP, check membership
            shap = model.predict_and_explain(feats)["shap_values"]
            top2 = sorted([c for c in CATS if c in shap], key=lambda c: shap[c])[:2]
            top2_hits[cat].append(int(cat in top2))
            manip[cat].append(feats.get(cat, np.nan) - base_feats.get(cat, np.nan))
            n_inj[cat].append(k)
            if band is not None:
                by_band[band][cat].append(int(pred == cat))
        print(f"  [{gi}/{len(games)}] {color:5s} Elo {elo} done", flush=True)

    analyzer.close()

    # ---- aggregate ----
    summary = {"n_games": len(games), "chance_baseline": 1.0 / len(CATS),
               "injection": {"categories": cats, "magnitude": args.magnitude, "rate": args.rate,
                             "wp_lo": args.wp_lo, "wp_hi": args.wp_hi,
                             "cp_lo": args.loss_lo, "cp_hi": args.loss_hi,
                             "depth": args.depth, "min_elo": args.min_elo},
               "per_category": {}}
    for c in cats:
        sens = float(np.mean(hits_by_game[c])) if hits_by_game[c] else 0.0
        summary["per_category"][c] = {
            "sensitivity_top1": sens,
            "sensitivity_top2": float(np.mean(top2_hits[c])) if top2_hits[c] else 0.0,
            "ci95": bootstrap_ci([[h] for h in hits_by_game[c]]),
            "mean_feature_delta": float(np.nanmean(manip[c])) if manip[c] else 0.0,
            "mean_injected_plies": float(np.mean(n_inj[c])) if n_inj[c] else 0.0,
        }
    overall = [h for c in cats for h in hits_by_game[c]]
    summary["overall_top1"] = float(np.mean(overall)) if overall else 0.0
    summary["overall_ci95"] = bootstrap_ci(list(hits_by_game.values()))
    summary["baseline_detect_profile"] = baseline_detect
    summary["coverage_at_k"] = {
        c: {"treated": [float(np.mean([r <= k for r in rank_treated[c]])) for k in (1, 2, 3)],
            "clean":   [float(np.mean([r <= k for r in rank_clean[c]])) for k in (1, 2, 3)]}
        for c in cats}
    summary["coverage_at_k_overall"] = {
        "treated": [float(np.mean([r <= k for c in cats for r in rank_treated[c]])) for k in (1, 2, 3)],
        "clean":   [float(np.mean([r <= k for c in cats for r in rank_clean[c]])) for k in (1, 2, 3)],
        "chance":  [k / len(CATS) for k in (1, 2, 3)]}
    if bands:
        summary["elo_bands"] = {
            f"{lo}-{hi}": {
                "n_games": len(by_band[(lo, hi)][cats[0]]),
                "overall_top1": float(np.mean([h for c in cats for h in by_band[(lo, hi)][c]]))
                                if by_band[(lo, hi)][cats[0]] else 0.0,
                "per_category": {c: float(np.mean(by_band[(lo, hi)][c]))
                                 if by_band[(lo, hi)][c] else 0.0 for c in cats},
            } for lo, hi in bands}

    name = f"synthetic_validation{args.tag}.json"
    with open(os.path.join(args.outdir, name), "w") as f:
        json.dump({"summary": summary, "confusion": confusion}, f, indent=2)

    # ---- console report ----
    crit = (f"Maia win-prob loss >= {args.wp_lo:.2f}" if args.magnitude == "winprob"
            else f"centipawn loss in [{args.loss_lo}, {args.loss_hi}]")
    print(f"\n=== Injection criterion: {crit} ===")
    print("=== Manipulation check (did the injected feature actually rise?) ===")
    print(f"{'category':22s}{'feat delta':>12}{'inj plies':>11}{'sens@1':>9}{'sens@2':>9}")
    for c in cats:
        s = summary["per_category"][c]
        print(f"{DISPLAY_NAMES.get(c, c)[:21]:22s}{s['mean_feature_delta']:>12.1f}"
              f"{s['mean_injected_plies']:>11.1f}{s['sensitivity_top1']:>9.2f}{s['sensitivity_top2']:>9.2f}")
    lo, hi = summary["overall_ci95"]
    print(f"\nOverall top-1 sensitivity: {summary['overall_top1']:.2f} "
          f"(95% CI {lo:.2f}-{hi:.2f}) vs chance {summary['chance_baseline']:.2f}")
    print(f"Baseline (no injection) detector picks: {baseline_detect}")

    cov = summary["coverage_at_k_overall"]
    print("\n=== Coverage@k: injected category among the k weakest ===")
    print(f"{'k':<4}{'treated':>10}{'clean':>10}{'lift':>9}{'chance':>9}")
    for i, k in enumerate((1, 2, 3)):
        print(f"{k:<4}{cov['treated'][i]:>10.3f}{cov['clean'][i]:>10.3f}"
              f"{cov['treated'][i] - cov['clean'][i]:>+9.3f}{cov['chance'][i]:>9.3f}")
    print(f"\n{'category':<22}{'@1':>7}{'@2':>7}{'@3':>7}{'  (clean: @1/@2/@3)':>22}")
    for c in cats:
        b = summary["coverage_at_k"][c]
        print(f"{DISPLAY_NAMES.get(c, c)[:21]:<22}"
              f"{b['treated'][0]:>7.2f}{b['treated'][1]:>7.2f}{b['treated'][2]:>7.2f}"
              f"   {b['clean'][0]:.2f} / {b['clean'][1]:.2f} / {b['clean'][2]:.2f}")

    if bands:
        print(f"\n=== Sensitivity by Elo band ===")
        print(f"{'band':<12}{'n':>4}" + "".join(f"{DISPLAY_NAMES.get(c, c)[:9]:>11}" for c in CATS)
              + f"{'ALL':>9}")
        for key, blk in summary["elo_bands"].items():
            print(f"{key:<12}{blk['n_games']:>4}"
                  + "".join(f"{blk['per_category'][c]:>11.2f}" for c in CATS)
                  + f"{blk['overall_top1']:>9.2f}")

    # ---- confusion figure ----
    mat = np.array([[confusion[gt][pr] for pr in CATS] for gt in cats])
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(mat, cmap="Greens")
    xlabs = [DISPLAY_NAMES.get(c, c) for c in CATS]
    ylabs = [DISPLAY_NAMES.get(c, c) for c in cats]
    ax.set_xticks(range(len(CATS))); ax.set_yticks(range(len(cats)))
    ax.set_xticklabels(xlabs, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(ylabs, fontsize=7)
    for i in range(len(cats)):
        for j in range(len(CATS)):
            ax.text(j, i, mat[i, j], ha="center", va="center", fontsize=8)
    ax.set_xlabel("Detected (single-game SHAP)")
    ax.set_ylabel("Injected (known ground truth)")
    ax.set_title("Synthetic weakness: injected vs detected")
    fig.colorbar(im, fraction=0.046); fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, f"fig_synth_confusion{args.tag}.png"))
    plt.close(fig)
    print(f"\nSaved results + figure to '{args.outdir}/'.")


if __name__ == "__main__":
    main()
