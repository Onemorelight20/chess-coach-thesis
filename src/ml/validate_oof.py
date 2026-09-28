"""
Валідація механізму виявлення слабких місць БЕЗ витоку даних за гравцями —
основний протокол таблиці 3.11 (усі варіанти правила) і базового варіанта на
сирих ознаках.

Той самий протокол, що й для точності рейтингу: GroupKFold(5) за гравцями. У
кожному блоці модель (змішаний набір, mixed_training.py), норми внесків,
довідник сирих ознак, популяційні статистики правила відхилень, тривіальний
орієнтир І перелік кандидатів за критерієм знака будуються ЛИШЕ на навчальних
гравцях; усі показники рахуються на відкладених гравцях і зводяться з п'яти
блоків разом (out-of-fold). Довірчі інтервали — кластерний бутстреп за гравцями.

Варіанти правила (як у validate_xai / correction_steps):
  raw            — argmin внеску серед усіх 29 ознак;
  themed         — серед 17 ознак із тренувальною темою;
  standardised17 — 17 ознак, внесок нормований на рівень (μ, σ за інтервалом r̂);
  standardised   — правило (2.7): 17 ознак → критерій знака (у блоці) → нормування;
  rawfeat        — те саме нормування на рівень для сирих значень ознак замість
                   внесків SHAP (базовий варіант без пояснення).

Usage:
    python -m src.ml.validate_oof --pool blitz --outdir results/rules
"""
import argparse, json, os, sys
from math import comb

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GroupKFold
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.features.extractor import (FEATURE_NAMES, WEAKNESS_CANDIDATES, THEMED_CANDIDATES,
                                    FEATURE_ORIENTATION)
from src.ml.candidate_signs import sign_test
from src.ml.mixed_training import mixed_training_set

T17 = list(THEMED_CANDIDATES)
ALL29 = list(WEAKNESS_CANDIDATES)
SEEDS, RNG, NB, K = range(10), 42, 8, 3
PARAMS = dict(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=RNG)
RULES = ("raw", "themed", "standardised17", "standardised", "rawfeat")
LABELS = {"raw": "наївне правило", "themed": "обмеження кандидатів", "standardised17": "обмеження і нормування",
          "standardised": "правило (2.7)", "rawfeat": "сирі ознаки, те саме нормування"}


def fit_fold(X, y, groups, tr):
    """Змішана модель лише на навчальних гравцях блоку + її пояснювач."""
    import shap
    trf = X.iloc[tr].copy(); trf["target_elo"] = y.iloc[tr].values; trf["player"] = groups.iloc[tr].values
    Xm, ym, _, _ = mixed_training_set(trf)
    m = xgb.XGBRegressor(**PARAMS).fit(Xm, ym)
    return m, shap.TreeExplainer(m)


def band_ref(values, pred, cols=None):
    """Середнє та std стовпців `values` (масив або DataFrame) за інтервалами прогнозованого рейтингу."""
    vals = values if isinstance(values, np.ndarray) else values[cols].to_numpy(float)
    e = np.quantile(pred, np.linspace(0, 1, NB + 1)); e[0], e[-1] = -np.inf, np.inf
    b = np.clip(np.searchsorted(e, pred, side="right") - 1, 0, NB - 1)
    f = [i for i in range(NB) if np.any(b == i)]
    return dict(edges=e, centres=np.array([pred[b == i].mean() for i in f]),
                mean=np.vstack([vals[b == i].mean(0) for i in f]), std=np.vstack([vals[b == i].std(0) for i in f]),
                median=np.vstack([np.median(vals[b == i], axis=0) for i in f]))


def z_of(sub, pred, ref):
    x = ref["centres"]
    mu = np.column_stack([np.interp(pred, x, ref["mean"][:, j]) for j in range(sub.shape[1])])
    sd = np.column_stack([np.interp(pred, x, ref["std"][:, j]) for j in range(sub.shape[1])])
    sd = np.where(sd > 0, sd, np.nan)
    z = (sub - mu) / sd
    return np.where(np.isfinite(z), z, np.inf)


def rank_shap(sv, pred, cands, ref=None, k=K):
    """Ранжування кандидатів за внеском (сирим або нормованим), найгірший перший."""
    sub = sv[:, [FEATURE_NAMES.index(c) for c in cands]].astype(float)
    score = z_of(sub, pred, ref) if ref is not None else sub
    return np.array(cands)[np.argsort(score, axis=1)[:, :k]]


def rank_rawfeat(V, pred, cands, ref, k=K):
    """Те саме нормування на рівень для сирих значень ознак (орієнтованих: більше = гірше)."""
    sub = V[cands].to_numpy(float)
    z = -z_of(sub, pred, ref) * np.array([FEATURE_ORIENTATION[c] for c in cands])   # мінус: найгірший = найменша оцінка
    z = np.where(np.isfinite(z), z, np.inf)
    return np.array(cands)[np.argsort(z, axis=1)[:, :k]]


def band_index(pred, ref):
    return np.clip(np.searchsorted(ref["edges"], pred, side="right") - 1, 0, len(ref["centres"]) - 1)


def kappa(a, b, labels):
    return float(cohen_kappa_score(a, b, labels=labels))


def overlap3(A, B):
    return float(np.mean([len(set(x[:3]) & set(y[:3])) > 0 for x, y in zip(A, B)]))


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
    print(f"{args.pool}: {len(df)} партій, {groups.nunique()} гравців; GroupKFold(5) за гравцями, змішана модель у кожному блоці\n")

    # накопичувачі за блоками
    half_player, half_A, half_B, modal_A, modal_B = [], {r: [] for r in RULES}, {r: [] for r in RULES}, {r: [] for r in RULES}, {r: [] for r in RULES}
    full_player, full_pred, gt_own, gt17_all, triv_hit, triv17_hit, rev_hit = [], {r: [] for r in RULES}, {r: [] for r in RULES}, [], {r: [] for r in RULES}, [], {r: [] for r in RULES}
    outside = []; fold_cands = []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, groups), 1):
        m, expl = fit_fold(X, y, groups, tr)
        tr_agg = X.iloc[tr].groupby(groups.iloc[tr].values).mean()[FEATURE_NAMES]
        sv_tr = expl.shap_values(tr_agg); pred_tr = np.asarray(expl.expected_value) + sv_tr.sum(axis=1)
        C, _ = sign_test(tr_agg, sv_tr, FEATURE_NAMES, T17); fold_cands.append(C)
        cands = {"raw": ALL29, "themed": T17, "standardised17": T17, "standardised": C, "rawfeat": C}
        refs = {"raw": None, "themed": None,
                "standardised17": band_ref(sv_tr[:, [FEATURE_NAMES.index(c) for c in T17]], pred_tr),
                "standardised": band_ref(sv_tr[:, [FEATURE_NAMES.index(c) for c in C]], pred_tr),
                "rawfeat": band_ref(tr_agg, pred_tr, C)}
        # популяція правила відхилень і тривіальні орієнтири — лише навчальні гравці
        tr_pm = X.iloc[tr].groupby(groups.iloc[tr].values)[ALL29].mean()
        mu_p, sd_p = tr_pm.mean(), tr_pm.std(ddof=0)
        z_tr = ((tr_pm - mu_p) / sd_p).mul(orient[ALL29])
        maj = {r: z_tr[cands[r]].idxmax(axis=1).mode().iloc[0] for r in RULES}
        maj17 = z_tr[T17].idxmax(axis=1).mode().iloc[0]

        te_df = df.iloc[te]; Xte = X.iloc[te]
        # ── тест-ретест на відкладених гравцях ──
        splits = []
        for p, g in te_df.groupby("player"):
            idx = g.index.to_numpy()
            if len(idx) < 4: continue
            for s_ in SEEDS:
                sh = np.random.RandomState(s_).permutation(idx); splits.append((p, sh[:len(sh) // 2], sh[len(sh) // 2:]))
        halves = [h for _, a, b in splits for h in (a, b)]
        H = pd.DataFrame([X.loc[h].mean().values for h in halves], columns=FEATURE_NAMES)
        svh = expl.shap_values(H); prh = np.asarray(expl.expected_value) + svh.sum(axis=1)
        svg = expl.shap_values(Xte); prg = np.asarray(expl.expected_value) + svg.sum(axis=1)   # окремі партії — для модального порядку
        pos = {i: k for k, i in enumerate(Xte.index)}
        for r in RULES:
            R = rank_rawfeat(H, prh, cands[r], refs[r]) if r == "rawfeat" else rank_shap(svh, prh, cands[r], refs[r])
            half_A[r].append(R[0::2]); half_B[r].append(R[1::2])
            if r != "rawfeat":
                pg = rank_shap(svg, prg, cands[r], refs[r], k=1)[:, 0]
                modal_A[r].append([pd.Series(pg[[pos[i] for i in a]]).mode().iloc[0] for _, a, _ in splits])
                modal_B[r].append([pd.Series(pg[[pos[i] for i in b]]).mode().iloc[0] for _, _, b in splits])
        half_player.append([s[0] for s in splits])

        # ── збіг із правилом відхилень на відкладених гравцях (усі партії) ──
        te_pm = Xte.groupby(groups.iloc[te].values)[ALL29].mean()
        z_te = ((te_pm - mu_p) / sd_p).mul(orient[ALL29])
        agg_te = Xte.groupby(groups.iloc[te].values).mean()[FEATURE_NAMES].loc[te_pm.index]
        sva = expl.shap_values(agg_te); pra = np.asarray(expl.expected_value) + sva.sum(axis=1)
        g17 = z_te[T17].idxmax(axis=1)
        for r in RULES:
            g_own = z_te[cands[r]].idxmax(axis=1)
            P = rank_rawfeat(agg_te, pra, cands[r], refs[r]) if r == "rawfeat" else rank_shap(sva, pra, cands[r], refs[r])
            full_pred[r].append(P); gt_own[r].append(g_own.values); triv_hit[r].append((g_own.values == maj[r]))
            # зворотна порада: за названою ознакою гравець кращий за медіану навчальних гравців свого інтервалу r̂
            rref = refs["rawfeat"]; bi = band_index(pra, rref)
            named = P[:, 0]; ci = [C.index(c) if c in C else None for c in named]
            hits = []
            for i, (c, j) in enumerate(zip(named, ci)):
                if j is None: hits.append(False); continue   # поза C (raw/themed можуть назвати ознаку без сирого довідника) — не рахуємо
                hits.append((agg_te[c].values[i] - rref["median"][bi[i], j]) * orient[c] < 0)
            rev_hit[r].append(np.array(hits))
        gt17_all.append(g17.values); triv17_hit.append((g17.values == maj17)); full_player.append(list(te_pm.index))
        outside.append(~g17.isin(C).values)
        print(f"  блок {fold}: {te_df['player'].nunique()} відкладених гравців, {len(splits)} розбиттів, {len(C)} кандидатів"
              + (f" (відмінність від блоку 1: {sorted(set(C) ^ set(fold_cands[0]))})" if fold > 1 and set(C) != set(fold_cands[0]) else ""))

    # ── зведення ──
    players_h = np.concatenate(half_player); gt17 = np.concatenate(gt17_all); outside = np.concatenate(outside)
    labels29 = sorted(ALL29); labels17 = sorted(T17)
    out = {"pool": args.pool, "protocol": "GroupKFold(5) за гравцями; модель, довідники, критерій знака й популяція правила відхилень — лише на навчальних гравцях блоку",
           "candidates_per_fold": fold_cands, "n_players": int(len(gt17)), "n_splits": int(len(players_h)), "rules": {}}
    rng = np.random.RandomState(RNG)
    groups_h = {p: np.where(players_h == p)[0] for p in np.unique(players_h)}; pl = np.array(list(groups_h))
    draws = [np.concatenate([groups_h[p] for p in rng.choice(pl, len(pl), replace=True)]) for _ in range(args.boot)]
    span = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    kap_draws = {}
    for r in RULES:
        A, B = np.vstack(half_A[r]), np.vstack(half_B[r]); lab = labels29 if r == "raw" else labels17
        P = np.vstack(full_pred[r]); g_own = np.concatenate(gt_own[r]); n_cand = len(ALL29) if r == "raw" else (len(T17) if r in ("themed", "standardised17") else len(fold_cands[0]))
        res = {"label": LABELS[r], "n_cand": n_cand, "chance": 1 / n_cand,
               "trivial": float(np.concatenate(triv_hit[r]).mean()),
               "tr_agree": float(np.mean(A[:, 0] == B[:, 0])), "kappa": kappa(A[:, 0], B[:, 0], lab),
               "ov3": overlap3(A, B), "ov3_chance": 1 - comb(n_cand - 3, 3) / comb(n_cand, 3),
               "agree": float(np.mean(P[:, 0] == g_own)), "agree3": float(np.mean([g in p[:3] for g, p in zip(g_own, P)])),
               "kappa_agree": kappa(g_own, P[:, 0], lab),
               "agree_ref17": float(np.mean(P[:, 0] == gt17)), "agree3_ref17": float(np.mean([g in p[:3] for g, p in zip(gt17, P)])),
               "kappa_agree_ref17": kappa(gt17, P[:, 0], labels29 if r == "raw" else labels17),
               "trivial_ref17": float(np.concatenate(triv17_hit).mean()),
               "share_outside": float(outside.mean()) if r in ("standardised", "rawfeat") else 0.0,
               "top_share": float(pd.Series(P[:, 0]).value_counts(normalize=True).iloc[0]),
               "top_name": str(pd.Series(P[:, 0]).value_counts().index[0]),
               "reversed_share": float(np.concatenate(rev_hit[r]).mean()),
               # випадковий збіг половин за тими самими частками відповідей (pₑ у формулі каппи) і сам розподіл відповідей
               "chance_agreement": float((pd.Series(np.concatenate([A[:, 0], B[:, 0]])).value_counts(normalize=True) ** 2).sum()),
               "answer_shares": {str(k): float(v) for k, v in pd.Series(P[:, 0]).value_counts(normalize=True).items()}}
        kd = np.array([kappa(A[d, 0], B[d, 0], lab) for d in draws]); kap_draws[r] = kd
        res["kappa_ci95"] = span(kd)
        if r != "rawfeat":
            MA, MB = np.concatenate(modal_A[r]), np.concatenate(modal_B[r])
            res["kappa_modal"] = kappa(MA, MB, lab)
            md = np.array([kappa(MA[d], MB[d], lab) for d in draws])
            res["kappa_gain_over_modal"] = {"delta": res["kappa"] - res["kappa_modal"], "ci95": span(kd - md)}
        out["rules"][r] = res
    # узгодженість правила (2.7) і його сирого аналога
    A_s, A_r = np.vstack(half_A["standardised"]), np.vstack(half_A["rawfeat"]); B_s, B_r = np.vstack(half_B["standardised"]), np.vstack(half_B["rawfeat"])
    Ps, Pr = np.vstack(full_pred["standardised"]), np.vstack(full_pred["rawfeat"])
    both = np.vstack([A_s, B_s]); bothr = np.vstack([A_r, B_r])
    out["cross_rule"] = {"top1_agreement": float(np.mean(both[:, 0] == bothr[:, 0])), "top3_overlap": overlap3(both, bothr),
                         "top1_agreement_full": float(np.mean(Ps[:, 0] == Pr[:, 0]))}
    d = kap_draws["standardised"] - kap_draws["rawfeat"]
    out["kappa_delta_shap_minus_raw"] = {"delta": out["rules"]["standardised"]["kappa"] - out["rules"]["rawfeat"]["kappa"],
                                         "ci95": span(d), "p_gt_0": float(np.mean(d > 0))}
    path = os.path.join(args.outdir, f"table_311_{args.pool}_oof.json")
    json.dump(out, open(path, "w"), indent=1, ensure_ascii=False)
    # сумісність зі старим виходом для правила (2.7)
    if args.pool == "blitz":
        s = out["rules"]["standardised"]
        legacy = {"kappa": s["kappa"], "split_half_agreement": s["tr_agree"], "topk_overlap_3": s["ov3"],
                  "top1_accuracy": s["agree"], "coverage_at_3": s["agree3"], "kappa_agreement": s["kappa_agree"],
                  "trivial_baseline": s["trivial"],
                  "ref17": {"top1_accuracy": s["agree_ref17"], "coverage_at_3": s["agree3_ref17"],
                            "kappa_agreement": s["kappa_agree_ref17"], "trivial_baseline": s["trivial_ref17"]},
                  "candidates_per_fold": fold_cands, "n_players": out["n_players"], "n_splits": out["n_splits"]}
        json.dump(legacy, open(os.path.join(args.outdir, "xai_validation_blitz_oof.json"), "w"), indent=1)

    print(f"\n{'показник':<22}" + "".join(f"{LABELS[r][:22]:>24}" for r in RULES))
    for k in ("n_cand", "chance", "trivial", "tr_agree", "kappa", "kappa_modal", "ov3", "ov3_chance", "agree", "agree3", "kappa_agree",
              "agree_ref17", "agree3_ref17", "kappa_agree_ref17", "trivial_ref17", "share_outside", "top_share", "reversed_share"):
        row = f"{k:<22}"
        for r in RULES:
            v = out["rules"][r].get(k); row += f"{v:>24}" if isinstance(v, int) else (f"{v:>24.3f}" if isinstance(v, float) else f"{'—':>24}")
        print(row)
    c = out["cross_rule"]; dd = out["kappa_delta_shap_minus_raw"]
    print(f"\nузгодженість (2.7) і сирого аналога: top-1 {c['top1_agreement']:.3f} (половини) / {c['top1_agreement_full']:.3f} (усі партії), перетин трійок {c['top3_overlap']:.3f}; "
          f"Δκ = {dd['delta']:+.3f} [{dd['ci95'][0]:+.3f}; {dd['ci95'][1]:+.3f}]")
    print(f"κ 95% ДІ: " + ", ".join(f"{r} [{out['rules'][r]['kappa_ci95'][0]:.3f}; {out['rules'][r]['kappa_ci95'][1]:.3f}]" for r in RULES))
    print(f"->  {path}")


if __name__ == "__main__":
    main()
