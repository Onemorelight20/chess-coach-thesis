"""
Чи можна підняти надійність правила (2.7), не втративши валідності — БЕЗ витоку
за гравцями (протокол validate_oof: GroupKFold(5), модель/довідник/критерій знака/
популяція правила відхилень — лише на навчальних гравцях блоку, вимірювання — на
відкладених). Позаблоковий аналог improve_rule.py; ті самі варіанти:
  base       — правило (2.7): z за прогнозованим рейтингом половини, argmin
  stability  — leave-one-out підвибірки половини, голосування за трійками
  grouped    — z усереднюється в межах 5 тем, ранжуються теми
  fixedelo   — r̂ береться з усіх партій гравця, а не з половини
  noopening  — opening_acpl виключено з кандидатів
  abstain    — крива «надійність проти покриття» за порогом відриву
  ceiling    — те саме правило відхилень (еталон) на половинах
  n_games    — n = 3 (перший LOO-вектор половини) проти n = 4 (половина)

Usage:
    python -m src.ml.improve_rule_oof --pool blitz --outdir results
"""
import argparse, json, os, sys
from math import comb

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.features.extractor import FEATURE_NAMES, THEMED_CANDIDATES, FEATURE_ORIENTATION
from src.ml.candidate_signs import sign_test
from src.ml.validate_oof import fit_fold, band_ref, z_of, kappa, overlap3, SEEDS, RNG

T17 = list(THEMED_CANDIDATES)
GROUPS = {  # теми над 17 тематичними ознаками; у блоці обмежуються переліком T
    "точність за фазами й ситуаціями": ["opening_acpl", "middlegame_acpl", "endgame_acpl", "conversion_acpl", "defense_acpl", "acpl_critical"],
    "тактичні помилки": ["blunder_rate", "mistake_rate", "inaccuracy_rate"],
    "тактичні мотиви": ["hanging_rate", "pin_exposure", "king_danger"],
    "позиційна гра": ["center_control_avg", "mobility_avg"],
    "стиль гри": ["material_volatility", "capture_rate", "check_given_rate"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="blitz")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    df = pd.read_parquet(f"data/dataset_{args.pool}_2k.parquet")
    df = df[df["target_elo"].between(400, 3000)].reset_index(drop=True)
    X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(df[FEATURE_NAMES]), columns=FEATURE_NAMES)
    y, groups = df["target_elo"], df["player"]
    orient = pd.Series(FEATURE_ORIENTATION)
    labels = sorted(T17); gnames = list(GROUPS)
    acc = {k: [] for k in ("base_a", "base_b", "stab_a", "stab_b", "grp_a", "grp_b", "fix_a", "fix_b", "noop_a", "noop_b", "n3_a", "n3_b",
                           "ceil_a", "ceil_b", "margin", "full_p1", "full_p3", "gt", "triv", "grp_p1", "grp_gt", "grp_triv")}
    fold_cands = []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, groups), 1):
        m, expl = fit_fold(X, y, groups, tr)
        tr_agg = X.iloc[tr].groupby(groups.iloc[tr].values).mean()[FEATURE_NAMES]
        sv_tr = expl.shap_values(tr_agg); pred_tr = np.asarray(expl.expected_value) + sv_tr.sum(axis=1)
        C, _ = sign_test(tr_agg, sv_tr, FEATURE_NAMES, T17); fold_cands.append(C)
        ci = [FEATURE_NAMES.index(c) for c in C]
        ref = band_ref(sv_tr[:, ci], pred_tr)
        tr_pm = X.iloc[tr].groupby(groups.iloc[tr].values)[T17].mean(); mu_p, sd_p = tr_pm.mean(), tr_pm.std(ddof=0)
        z_tr = ((tr_pm - mu_p) / sd_p).mul(orient[T17])
        maj = z_tr[C].idxmax(axis=1).mode().iloc[0]
        groups_f = {g: [f for f in fs if f in C] for g, fs in GROUPS.items()}; groups_f = {g: fs for g, fs in groups_f.items() if fs}
        gmap = {f: g for g, fs in groups_f.items() for f in fs}
        maj_g = z_tr[C].idxmax(axis=1).map(gmap).mode().iloc[0]

        te_df = df.iloc[te]; Xte = X.iloc[te]
        splits = []
        for p, g in te_df.groupby("player"):
            idx = g.index.to_numpy()
            if len(idx) < 4: continue
            for s_ in SEEDS:
                sh = np.random.RandomState(s_).permutation(idx); splits.append((p, sh[:len(sh) // 2], sh[len(sh) // 2:]))
        halves = [h for _, a, b in splits for h in (a, b)]
        vecs, loo_span = [], []
        for h in halves:
            vecs.append(X.loc[h].mean().values); st = len(vecs)
            for drop in range(len(h)): vecs.append(X.loc[np.delete(h, drop)].mean().values)
            loo_span.append((st, len(vecs)))
        full_idx = {p_: g.index.to_numpy() for p_, g in te_df.groupby("player")}; full_players = list(full_idx)   # мітки рядків, не позиції
        vecs += [X.loc[full_idx[p]].mean().values for p in full_players]
        V = pd.DataFrame(vecs, columns=FEATURE_NAMES)
        sv = expl.shap_values(V); pred = np.asarray(expl.expected_value) + sv.sum(axis=1); sub = sv[:, ci]
        Z = z_of(sub, pred, ref)
        mean_idx = np.array([st - 1 for st, _ in loo_span]); n_full = len(full_players)
        full_rows = np.arange(len(V) - n_full, len(V)); full_pred = dict(zip(full_players, pred[full_rows]))
        Ca = np.array(C)
        rank = lambda rows, cols=C: np.array(cols)[Z[rows][:, [C.index(c) for c in cols]].argsort(1)]
        R = rank(mean_idx); acc["base_a"].append(R[0::2, :3]); acc["base_b"].append(R[1::2, :3])
        stab = []
        for st, en in loo_span:
            r = Ca[Z[st:en].argsort(1)][:, :3]; cnt = pd.Series(r.ravel()).value_counts(); stab.append(list(cnt.index[:3]))
        S = np.array(stab); acc["stab_a"].append(S[0::2, :3]); acc["stab_b"].append(S[1::2, :3])
        GZ = np.column_stack([Z[mean_idx][:, [C.index(f) for f in fs]].mean(1) for fs in groups_f.values()])
        GR = np.array(list(groups_f))[GZ.argsort(1)]; acc["grp_a"].append(GR[0::2, :1]); acc["grp_b"].append(GR[1::2, :1])
        pl = np.repeat([s[0] for s in splits], 2); fixed_pred = np.array([full_pred[p] for p in pl])
        RF = Ca[z_of(sub[mean_idx], fixed_pred, ref).argsort(1)]; acc["fix_a"].append(RF[0::2, :3]); acc["fix_b"].append(RF[1::2, :3])
        C2 = [c for c in C if c != "opening_acpl"]; RN = rank(mean_idx, C2); acc["noop_a"].append(RN[0::2, :3]); acc["noop_b"].append(RN[1::2, :3])
        loo_first = np.array([st for st, _ in loo_span]); R3 = rank(loo_first); acc["n3_a"].append(R3[0::2, :3]); acc["n3_b"].append(R3[1::2, :3])
        Zm = np.sort(Z[mean_idx], axis=1); margin = Zm[:, 1] - Zm[:, 0]; acc["margin"].append(np.minimum(margin[0::2], margin[1::2]))
        def dev_weak(rows):
            mm = X.loc[rows, C].mean(); return ((mm - mu_p[C]) / sd_p[C]).mul(orient[C]).idxmax()
        acc["ceil_a"].append(np.array([dev_weak(a) for _, a, _ in splits])); acc["ceil_b"].append(np.array([dev_weak(b) for _, _, b in splits]))
        # валідність на всіх партіях відкладених гравців
        te_pm = Xte.groupby(groups.iloc[te].values)[T17].mean().loc[full_players]
        gt = ((te_pm - mu_p) / sd_p).mul(orient[T17])[C].idxmax(axis=1).values
        P = Ca[Z[full_rows].argsort(1)]; acc["full_p1"].append(P[:, 0]); acc["full_p3"].append(P[:, :3]); acc["gt"].append(gt); acc["triv"].append(gt == maj)
        GP = np.array(list(groups_f))[np.column_stack([Z[full_rows][:, [C.index(f) for f in fs]].mean(1) for fs in groups_f.values()]).argsort(1)][:, 0]
        gtg = pd.Series(gt).map(gmap).values; acc["grp_p1"].append(GP); acc["grp_gt"].append(gtg); acc["grp_triv"].append(gtg == maj_g)
        print(f"  блок {fold}: {len(full_players)} гравців, {len(splits)} розбиттів, {len(C)} кандидатів, {len(groups_f)} тем")

    cat = lambda k: np.concatenate(acc[k]) if np.asarray(acc[k][0]).ndim == 1 else np.vstack(acc[k])
    res = {"pool": args.pool, "candidates_per_fold": fold_cands, "n_splits": int(len(cat("margin")))}
    def pair(ka, kb, lab=labels, n_cand=len(fold_cands[0])):
        A, B = cat(ka), cat(kb)
        return dict(kappa=kappa(A[:, 0], B[:, 0], lab), ov3=overlap3(A, B), chance_ov3=1 - comb(n_cand - 3, 3) / comb(n_cand, 3), n_cand=n_cand)
    res["base"] = pair("base_a", "base_b"); res["stability"] = pair("stab_a", "stab_b")
    GA, GB = cat("grp_a"), cat("grp_b"); res["grouped"] = dict(kappa=kappa(GA[:, 0], GB[:, 0], gnames), ov3=1.0, chance_ov3=1.0, n_cand=len(gnames))
    res["fixedelo"] = pair("fix_a", "fix_b"); res["noopening"] = pair("noop_a", "noop_b", n_cand=len(fold_cands[0]) - 1)
    A, B, mg = cat("base_a"), cat("base_b"), cat("margin"); curve = []
    for q in (0.0, 0.25, 0.5, 0.75, 0.9):
        thr = np.quantile(mg, q); keep = mg >= thr
        curve.append(dict(quantile=q, threshold=float(thr), coverage=float(keep.mean()), kappa=kappa(A[keep, 0], B[keep, 0], labels), ov3=overlap3(A[keep], B[keep])))
    res["abstain_curve"] = curve
    ga, gb = cat("ceil_a"), cat("ceil_b")
    res["ceiling_deviation_rule"] = dict(kappa=kappa(ga, gb, labels), agreement=float(np.mean(ga == gb)))
    p1, p3, gt = cat("full_p1"), cat("full_p3"), cat("gt")
    res["validity"] = {"base_top1": float(np.mean(p1 == gt)), "base_top3": float(np.mean([g in r for g, r in zip(gt, p3)])),
                       "trivial": float(cat("triv").mean()), "grouped_top1": float(np.mean(cat("grp_p1") == cat("grp_gt"))),
                       "grouped_trivial": float(cat("grp_triv").mean())}
    res["n_games_curve"] = {"3": pair("n3_a", "n3_b"), "4": res["base"]}
    for k in ("3", "4"): res["n_games_curve"][k] = {kk: res["n_games_curve"][k][kk] for kk in ("kappa", "ov3")}
    path = os.path.join(args.outdir, f"rule_improvements_oof{'' if args.pool == 'blitz' else '_' + args.pool}.json")
    json.dump(res, open(path, "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("candidates_per_fold",)}, ensure_ascii=False, indent=1))
    print("->", path)


if __name__ == "__main__":
    main()
