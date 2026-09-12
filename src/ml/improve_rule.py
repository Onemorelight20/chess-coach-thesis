"""
Чи можна підняти надійність правила (2.7), не втративши валідності.

П'ять ідей на спільній основі (пул бліц, той самий протокол, що у validate_xai):
  base       — поточне правило: z за прогнозованим рейтингом половини, argmin
  stability  — відбір за стабільністю: leave-one-out підвибірки половини, голосування
  grouped    — z усереднюється в межах 6 тем, ранжуються теми
  fixedelo   — r̂ береться з усіх партій гравця, а не з половини (діагностика)
  noopening  — opening_acpl виключено з кандидатів
  abstain    — крива «надійність проти покриття» за порогом відриву δ
"""
import json, os, sys
from math import comb

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import cohen_kappa_score
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.features.extractor import (FEATURE_NAMES, TRAINABLE_CANDIDATES,
                                    FEATURE_ORIENTATION)

RNG, SEEDS, PARQUET = 42, range(10), "data/dataset_blitz_2k.parquet"
C = list(TRAINABLE_CANDIDATES)
GROUPS = {
    "фази гри":        ["opening_acpl", "middlegame_acpl", "endgame_acpl"],
    "ситуативна гра":  ["conversion_acpl", "defense_acpl", "acpl_critical"],
    "тактичні помилки":["blunder_rate", "mistake_rate", "inaccuracy_rate"],
    "тактичні патерни":["hanging_rate", "pin_exposure", "king_danger"],
    "позиційна гра":   ["center_control_avg", "mobility_avg"],
    "матеріал і стиль":["material_volatility", "capture_rate", "check_given_rate"],
}
assert sorted(sum(GROUPS.values(), [])) == sorted(C)


def fit(X, y):
    Xi = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=X.columns)
    m = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=RNG)
    m.fit(Xi, y)
    import shap
    return shap.TreeExplainer(m), Xi


def band_ref(expl, Xi, n=8):
    sv = expl.shap_values(Xi)
    pred = np.asarray(expl.expected_value) + sv.sum(axis=1)
    sub = sv[:, [FEATURE_NAMES.index(c) for c in C]]
    e = np.quantile(pred, np.linspace(0, 1, n + 1)); e[0], e[-1] = -np.inf, np.inf
    b = np.clip(np.searchsorted(e, pred, side="right") - 1, 0, n - 1)
    f = [i for i in range(n) if np.any(b == i)]
    return {"centres": np.array([pred[b == i].mean() for i in f]),
            "mean": np.vstack([sub[b == i].mean(0) for i in f]),
            "std":  np.vstack([sub[b == i].std(0)  for i in f])}


def zscores(sub, pred, ref):
    """Нормовані внески: (φ − μ(r̂)) / σ(r̂), інтерпольовані між центрами."""
    x = ref["centres"]
    mu = np.column_stack([np.interp(pred, x, ref["mean"][:, j]) for j in range(sub.shape[1])])
    sd = np.column_stack([np.interp(pred, x, ref["std"][:, j])  for j in range(sub.shape[1])])
    sd = np.where(sd > 0, sd, np.nan)
    z = (sub - mu) / sd
    return np.where(np.isfinite(z), z, np.inf)      # непридатні — у кінець ранжування


def kappa(a, b, labels):
    return float(cohen_kappa_score(a, b, labels=labels))


def overlap(A, B, k):
    return float(np.mean([len(set(x[:k]) & set(y[:k])) > 0 for x, y in zip(A, B)]))


def main():

    df = pd.read_parquet(PARQUET)
    df = df[df["target_elo"].between(400, 3000)].reset_index(drop=True)
    expl, Xi = fit(df[FEATURE_NAMES], df["target_elo"])
    # еталон будується на агрегованих векторах гравців — саме до таких
    # застосовується правило; на окремих партіях розкид інший (див. validate_xai)
    agg = Xi.groupby(df['player'].values).mean()[FEATURE_NAMES]
    ref = band_ref(expl, agg)
    ci = [FEATURE_NAMES.index(c) for c in C]
    print(f"модель навчена; {len(df)} партій, {df['player'].nunique()} гравців\n")

    # ── розбиття навпіл ──
    splits = []
    for p, g in df.groupby("player"):
        if len(g) < 4:
            continue
        idx = g.index.to_numpy()
        for s in SEEDS:
            sh = np.random.RandomState(s).permutation(idx)
            splits.append((p, sh[:len(sh)//2], sh[len(sh)//2:]))
    print(f"розбиттів: {len(splits)}")

    # ── усі потрібні вектори одним проходом SHAP ──
    halves = [h for _, a, b in splits for h in (a, b)]
    vecs, loo_span = [], []
    for h in halves:
        vecs.append(Xi.loc[h].mean().values)               # середнє половини
        start = len(vecs)
        for drop in range(len(h)):                          # leave-one-out
            keep = np.delete(h, drop)
            vecs.append(Xi.loc[keep].mean().values)
        loo_span.append((start, len(vecs)))
    full = {p: Xi.loc[g.index].mean().values for p, g in df.groupby("player")}
    order = list(full)
    vecs += [full[p] for p in order]
    V = pd.DataFrame(vecs, columns=Xi.columns)
    print(f"векторів на пояснення: {len(V)}  ...")
    sv = expl.shap_values(V)
    pred = np.asarray(expl.expected_value) + sv.sum(axis=1)
    sub = sv[:, ci]
    Z = zscores(sub, pred, ref)
    print("SHAP готово\n")

    n_half = len(halves)
    # індекси: для кожної половини рядок середнього — це loo_span[i][0]-1
    mean_idx = np.array([st - 1 for st, _ in loo_span])
    full_idx = np.arange(len(V) - len(order), len(V))
    full_pred = dict(zip(order, pred[full_idx]))

    def rank_rows(rows, names=C):
        return np.array(names)[Z[rows][:, [C.index(n) for n in names]].argsort(1)]

    results = {}

    # ── base ──
    R = rank_rows(mean_idx)
    A, B = R[0::2], R[1::2]
    results["base"] = dict(kappa=kappa(A[:, 0], B[:, 0], C), ov3=overlap(A, B, 3),
                           chance_ov3=1 - comb(len(C)-3, 3)/comb(len(C), 3), n_cand=len(C))

    # ── stability: голосування по leave-one-out підвибірках ──
    stab = []
    for st, en in loo_span:
        r = np.array(C)[Z[st:en].argsort(1)][:, :3]
        cnt = pd.Series(r.ravel()).value_counts()
        stab.append(list(cnt.index[:3]))
    S = np.array(stab)
    results["stability"] = dict(kappa=kappa(S[0::2, 0], S[1::2, 0], C),
                                ov3=overlap(S[0::2], S[1::2], 3),
                                chance_ov3=results["base"]["chance_ov3"], n_cand=len(C))

    # ── grouped ──
    gnames = list(GROUPS)
    GZ = np.column_stack([Z[mean_idx][:, [C.index(f) for f in fs]].mean(1) for fs in GROUPS.values()])
    GR = np.array(gnames)[GZ.argsort(1)]
    results["grouped"] = dict(kappa=kappa(GR[0::2, 0], GR[1::2, 0], gnames),
                              ov3=overlap(GR[0::2], GR[1::2], 3),
                              chance_ov3=1 - comb(len(gnames)-3, 3)/comb(len(gnames), 3),
                              n_cand=len(gnames))

    # ── fixedelo: r̂ з усіх партій гравця ──
    pl = np.repeat([s[0] for s in splits], 2)
    fixed_pred = np.array([full_pred[p] for p in pl])
    ZF = zscores(sub[mean_idx], fixed_pred, ref)
    RF = np.array(C)[ZF.argsort(1)]
    results["fixedelo"] = dict(kappa=kappa(RF[0::2, 0], RF[1::2, 0], C),
                               ov3=overlap(RF[0::2], RF[1::2], 3),
                               chance_ov3=results["base"]["chance_ov3"], n_cand=len(C))

    # ── noopening ──
    C2 = [c for c in C if c != "opening_acpl"]
    RN = rank_rows(mean_idx, C2)
    results["noopening"] = dict(kappa=kappa(RN[0::2, 0], RN[1::2, 0], C2),
                                ov3=overlap(RN[0::2], RN[1::2], 3),
                                chance_ov3=1 - comb(len(C2)-3, 3)/comb(len(C2), 3), n_cand=len(C2))

    # ── abstain: надійність проти покриття ──
    Zm = np.sort(Z[mean_idx], axis=1)
    margin = Zm[:, 1] - Zm[:, 0]
    ma, mb = margin[0::2], margin[1::2]
    curve = []
    for q in (0.0, 0.25, 0.5, 0.75, 0.9):
        thr = np.quantile(np.minimum(ma, mb), q)
        keep = np.minimum(ma, mb) >= thr
        if keep.sum() < 50: continue
        curve.append(dict(quantile=q, threshold=float(thr), coverage=float(keep.mean()),
                          kappa=kappa(A[keep, 0], B[keep, 0], C),
                          ov3=overlap(A[keep], B[keep], 3)))
    results["abstain_curve"] = curve

    # ── валідність: збіг із розрахунком за відхиленнями ──
    pm = df.groupby("player")[C].mean()
    z_gt = ((pm - pm.mean()) / pm.std(ddof=0)).mul(pd.Series(FEATURE_ORIENTATION)[C], axis=1)
    gt = z_gt.idxmax(axis=1).reindex(order)
    ZP = Z[full_idx]
    P1 = np.array(C)[ZP.argsort(1)]
    maj = gt.value_counts(normalize=True).iloc[0]
    results["validity"] = {
        "base_top1": float((P1[:, 0] == gt.values).mean()),
        "base_top3": float(np.mean([g in r[:3] for g, r in zip(gt.values, P1)])),
        "trivial": float(maj)}
    gt_g = gt.map({f: g for g, fs in GROUPS.items() for f in fs})
    GP = np.array(gnames)[np.column_stack(
        [ZP[:, [C.index(f) for f in fs]].mean(1) for fs in GROUPS.values()]).argsort(1)]
    results["validity"]["grouped_top1"] = float((GP[:, 0] == gt_g.values).mean())
    results["validity"]["grouped_trivial"] = float(gt_g.value_counts(normalize=True).iloc[0])

    # ── СТЕЛЯ: чи відтворюване саме еталонне правило відхилень на половинах? ──
    # Якщо ні, то жоден спосіб атрибуції не може бути надійнішим за нього.
    mu_p, sd_p = pm.mean(), pm.std(ddof=0)
    orient = pd.Series(FEATURE_ORIENTATION)[C]
    def dev_weak(rows):
        m = Xi.loc[rows, C].mean()
        return ((m - mu_p) / sd_p).mul(orient).idxmax()
    ga = [dev_weak(a) for _, a, _ in splits]
    gb = [dev_weak(b) for _, _, b in splits]
    results["ceiling_deviation_rule"] = dict(
        kappa=kappa(ga, gb, C),
        agreement=float(np.mean(np.asarray(ga) == np.asarray(gb))))

    # ── залежність від кількості партій у половині (n=3 з leave-one-out, n=4 — сама половина) ──
    loo_first = np.array([st for st, _ in loo_span])          # один LOO-вектор на половину
    R3 = np.array(C)[Z[loo_first].argsort(1)]
    results["n_games_curve"] = {
        "3": dict(kappa=kappa(R3[0::2, 0], R3[1::2, 0], C), ov3=overlap(R3[0::2], R3[1::2], 3)),
        "4": dict(kappa=results["base"]["kappa"], ov3=results["base"]["ov3"])}

    print(json.dumps(results, ensure_ascii=False, indent=1))
    json.dump(results, open("results/rule_improvements.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
