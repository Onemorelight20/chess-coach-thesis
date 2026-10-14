"""
Синтетична перевірка (підрозділ 3.3.4) без витоку даних за гравцями.

Розгорнута модель бліцу й три довідники норм (внески з критерієм знака, внески
всіх 17 тематичних ознак, сирі значення ознак) побудовані на всіх 2 160 гравцях
навчального набору. Самі синтетичні партії до набору не входять (за game_id
жодного збігу), але частина цільових гравців має в ньому інші партії
(results/rules/synthetic_games_players.csv). Тут модель навчається тією самою
функцією train.train_model (змішаний набір, ті самі параметри) на наборі без цих
гравців, довідники будуються тими самими функціями на тих самих відфільтрованих
гравцях, і всі прогони синтетичного експерименту, на які спирається текст,
повторюються з кешу рушія без Stockfish. Скрипти прогонів не змінено: їм лише
підставляється інша модель (src.ml.model.ELOModel та імена, імпортовані з нього).

Контроль: модель, навчена тим самим кодом на повному наборі, має давати ті самі
прогнози, що й розгорнута; довідники, перебудовані розгорнутою моделлю, мають
збігтися зі збереженими .pkl; той самий конвеєр із розгорнутою моделлю мусить
відтворити збережені results/*.json і рисунок 3.10.

Усе пишеться в окрему теку (за замовчуванням results/noleak/); наявні файли не
змінюються.

Usage:
    python -m src.ml.synthetic_noleak --outdir results/noleak
"""
import argparse, contextlib, json, os, sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.features.extractor import FEATURE_NAMES, SIGN_EXCLUDED
import src.ml.model as M
import src.ml.synthetic_validation as SV
import src.ml.synthetic_aggregate as SA
import src.ml.compare_rules_synthetic as CR
from src.ml import build_shap_reference as BSR, build_feature_reference as BFR
from src.ml.train import train_model

ELOModel = M.ELOModel                     # справжній клас, збережений до підміни
PARQUET = "data/dataset_blitz_2k.parquet"
PLAYERS = "results/rules/synthetic_games_players.csv"
DEPLOYED = {"model": "src/ml/models/xgboost_elo_blitz.json",
            "signfilter": "src/ml/models/shap_reference_blitz.pkl",
            "nosign": "src/ml/models/shap_reference_blitz_nosign.pkl",
            "rawfeat": "results/feature_reference_blitz.pkl"}
BANDS = "results/synthetic_features_bands.json"
HIGHDOSE = "results/synthetic_features_highdose.json"
WIDE = "results/synthetic_aggregate_features_wide.json"
FIG_UA = "results/fig_ua/fig_3_10.png"


def runs(refs, outdir):
    """Прогони, на які спирається текст: команди з README з іншою текою й іншими
    довідниками. Кожен елемент – (модуль, argv, файл результату)."""
    sv = ["--event", "blitz", "--outdir", outdir]
    return [
        (SV, ["--from-cache", BANDS, *sv, "--tag", "_bands"], "synthetic_validation_bands.json"),
        (SV, ["--from-cache", BANDS, *sv, "--rule", "standardised", "--reference", refs["nosign"],
              "--tag", "_bands_std"], "synthetic_validation_bands_std.json"),
        (SV, ["--from-cache", BANDS, *sv, "--rule", "standardised", "--tag", "_bands_signfilter"],
         "synthetic_validation_bands_signfilter.json"),
        (SV, ["--from-cache", BANDS, *sv, "--rule", "rawfeat", "--feature-reference", refs["rawfeat"],
              "--tag", "_rawfeat"], "synthetic_validation_rawfeat.json"),
        (SV, ["--from-cache", HIGHDOSE, *sv, "--tag", "_highdose"], "synthetic_validation_highdose.json"),
        (SA, ["--from-cache", WIDE, "--outdir", outdir, "--tag", "_wide"], "synthetic_aggregate_wide.json"),
        # збережений _wide_std рахувався з довідником усіх 17 тематичних ознак (інакше
        # чотири категорії поза T не могли б бути названі), тож і тут --reference nosign
        (SA, ["--from-cache", WIDE, "--outdir", outdir, "--rule", "standardised", "--reference", refs["nosign"],
              "--tag", "_wide_std"], "synthetic_aggregate_wide_std.json"),
        (CR, ["--cache", BANDS, "--pool", "blitz", "--shap-reference", refs["nosign"],
              "--feature-reference", refs["rawfeat"], "--out", os.path.join(outdir, "rules_synthetic_comparison.json")],
         "rules_synthetic_comparison.json"),
    ]


@contextlib.contextmanager
def substituted(model_path, shap_ref, loaded):
    """Підміняє ELOModel у модулях прогонів і побудови довідників: замість моделі
    пулу завантажується model_path, а довідником для правила (2.7) стає shap_ref."""
    def make(pool=None, **_):
        m = ELOModel(model_path=model_path)
        assert os.path.abspath(m.model_path) == os.path.abspath(model_path) and not m.is_mock, m.model_path
        m._shap_reference = shap_ref
        loaded.append(m.model_path)
        return m
    mods = (M, SV, SA, CR)
    saved = [m.ELOModel for m in mods]
    for m in mods:
        m.ELOModel = make
    try:
        yield
    finally:
        for m, s in zip(mods, saved):
            m.ELOModel = s


def logged(path, fn, *a, **kw):
    """Виклик із повним журналом консолі у файлі."""
    with open(path, "w") as f, contextlib.redirect_stdout(f):
        return fn(*a, **kw)


def run_all(refs, model_path, outdir, logdir):
    """Усі прогони з runs(): модель model_path, довідник правила (2.7) refs['signfilter']."""
    os.makedirs(outdir, exist_ok=True)
    loaded = []
    with substituted(model_path, joblib.load(refs["signfilter"]), loaded):
        for mod, argv, out in runs(refs, outdir):
            sys.argv = [mod.__name__] + argv
            logged(os.path.join(logdir, out.replace(".json", ".log")), mod.main)
    assert set(loaded) == {model_path}, loaded
    return len(loaded)


def build_refs(model_path, parquet, outdir, suffix):
    """Три довідники тими самими функціями, що й розгорнуті, на моделі model_path."""
    p = {"signfilter": os.path.join(outdir, f"shap_reference_blitz{suffix}.pkl"),
         "nosign": os.path.join(outdir, f"shap_reference_blitz_nosign{suffix}.pkl"),
         "rawfeat": os.path.join(outdir, f"feature_reference_blitz{suffix}.pkl")}
    loaded = []
    with substituted(model_path, None, loaded):
        log = lambda k: os.path.join(outdir, f"build_{k}{suffix}.log")
        logged(log("signfilter"), BSR.build, parquet, "blitz", out_path=p["signfilter"])
        logged(log("nosign"), BSR.build, parquet, "blitz", all_themed=True, out_path=p["nosign"])
        logged(log("rawfeat"), BFR.build, parquet, "blitz", out_path=p["rawfeat"])
    assert set(loaded) == {model_path}, loaded
    return p


def diff_json(a, b, path="", tol=1e-12):
    """Розбіжності двох JSON-структур (числа з допуском tol)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f"{path}: ключі {sorted(set(a) ^ set(b))}"] if set(a) != set(b) else []
        return out + [d for k in a if k in b for d in diff_json(a[k], b[k], f"{path}/{k}", tol)]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: довжина {len(a)} / {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in diff_json(x, y, f"{path}[{i}]", tol)]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return [] if abs(a - b) <= tol else [f"{path}: {a} / {b}"]
    return [] if a == b else [f"{path}: {a} / {b}"]


def diff_ref(a, b):
    """Розбіжності двох довідників (кадри mean/std – з допуском 1e-9)."""
    out = [f"{k}: {a.get(k)} / {b.get(k)}" for k in ("candidates", "sign_excluded", "n_players", "n_bands")
           if a.get(k) != b.get(k)]
    for k in ("band_edges", "band_centres"):
        if not np.allclose(a[k], b[k], rtol=0, atol=1e-9):
            out.append(f"{k}: {a[k]} / {b[k]}")
    for k in ("mean", "std"):
        x, y = a[k], b[k]
        if list(x.columns) != list(y.columns) or x.shape != y.shape or \
                not np.allclose(x.to_numpy(float), y.to_numpy(float), rtol=0, atol=1e-9, equal_nan=True):
            out.append(f"{k}: кадри різняться")
    return out


def fig_310(json_path, outdir):
    """Рисунок 3.10 тим самим кодом, що й у figures_ua.py, з іншого файлу результату."""
    from src.ml import figures_ua as F
    import matplotlib.pyplot as plt
    syn = json.load(open(json_path))
    cats_all = ["opening_acpl", "middlegame_acpl", "endgame_acpl", "conversion_acpl", "defense_acpl", "blunder_rate", "hanging_rate"]
    cats = syn["summary"]["injection"]["categories"]
    mat = np.array([[syn["confusion"][g][p] for p in cats_all] for g in cats])
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(mat, cmap="Greens"); ax.grid(False)
    ax.set_xticks(range(len(cats_all))); ax.set_yticks(range(len(cats)))
    ax.set_xticklabels([F.CATEGORY[c] for c in cats_all], rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels([F.CATEGORY[c] for c in cats], fontsize=9)
    F.heat_text(ax, mat, 8, white_above=0.6 * mat.max())
    ax.set_xlabel("Виявлена слабкість (базове правило)"); ax.set_ylabel("Внесена слабкість")
    fig.colorbar(im, fraction=0.046); fig.tight_layout()
    problems = []
    F.save(fig, outdir, "fig_3_10", problems)
    assert not problems, problems
    return os.path.join(outdir, "fig_3_10.png")


def same_pixels(a, b):
    from PIL import Image
    x, y = np.asarray(Image.open(a).convert("RGBA")), np.asarray(Image.open(b).convert("RGBA"))
    return x.shape == y.shape and bool((x == y).all())


def key_numbers(d):
    """Числа, які наводить текст (підрозділи 3.3.4, 3.6, 3.7, висновки), з теки d."""
    J = lambda f: json.load(open(os.path.join(d, f)))
    raw, std, sf, rf, hd = (J(f"synthetic_validation_{t}.json")
                            for t in ("bands", "bands_std", "bands_signfilter", "rawfeat", "highdose"))
    agg, cmp_ = J("synthetic_aggregate_wide.json"), J("rules_synthetic_comparison.json")

    def lifts(s):
        S = s["summary"]; n = S["n_games"]
        return {c: {"clean": S["baseline_detect_profile"][c] / n, "treated": v["sensitivity_top1"],
                    "lift": v["sensitivity_top1"] - S["baseline_detect_profile"][c] / n}
                for c, v in S["per_category"].items()}

    def coverage(s):
        return {c: {"treated": v["treated"], "lift": [t - c0 for t, c0 in zip(v["treated"], v["clean"])]}
                for c, v in s["summary"]["coverage_at_k"].items()}

    n = raw["summary"]["n_games"]
    return {
        "3.12 базове правило (чисті / після / приріст)": lifts(raw),
        "3.12 загальна чутливість базового правила": raw["summary"]["overall_top1"],
        "3.12 ДІ загальної чутливості (за категоріями)": raw["summary"]["overall_ci95"],
        "3.12 загальна чутливість за інтервалами рейтингу": {b: v["overall_top1"] for b, v in raw["summary"]["elo_bands"].items()},
        "3.10 грубі помилки → точність у дебюті (з 80)": raw["confusion"]["blunder_rate"]["opening_acpl"],
        "3.10 висячі фігури → точність у дебюті (з 80)": raw["confusion"]["hanging_rate"]["opening_acpl"],
        "3.10 матриця": raw["confusion"],
        "3.13 покриття @1/@2/@3 і приріст": coverage(raw),
        "дозовий контроль (чисті / після / приріст)": lifts(hd),
        "без критерію знака: загальна чутливість": std["summary"]["overall_top1"],
        "без критерію знака: частки відповідей на чистих партіях": {c: v / n for c, v in std["summary"]["baseline_detect_profile"].items()},
        "базове: частки відповідей на чистих партіях": {c: v / n for c, v in raw["summary"]["baseline_detect_profile"].items()},
        "3.14 без критерію знака (чисті / після / приріст)": lifts(std),
        "без критерію знака: покриття @1/@2/@3 і приріст": coverage(std),
        "правило (2.7): загальна чутливість": sf["summary"]["overall_top1"],
        "3.14 правило (2.7) (чисті / після / приріст)": lifts(sf),
        "контрольне правило: загальна чутливість": rf["summary"]["overall_top1"],
        "контрольне правило: точність у захисті, погіршені / чисті (з 80)":
            [rf["confusion"]["defense_acpl"]["defense_acpl"], rf["summary"]["baseline_detect_profile"]["defense_acpl"]],
        "3.14 парні прирости за категоріями": cmp_["per_category"],
        "3.14 парні прирости за групами": cmp_["overall"],
        "агрегування: приріст пооб'єктно / агреговано": {c: [v["per_game_lift"], v["aggregated_lift"]] for c, v in agg["per_category"].items()},
        "агрегування: загальна чутливість пооб'єктно / агреговано": [agg["overall_per_game"], agg["overall_aggregated"]],
    }


def flat(x, path=""):
    if isinstance(x, dict):
        return [r for k, v in x.items() for r in flat(v, f"{path} / {k}" if path else k)]
    if isinstance(x, list) and any(isinstance(v, (dict, list)) for v in x):
        return [r for i, v in enumerate(x) for r in flat(v, f"{path}[{i}]")]
    return [(path, x)]


def fmt(v):
    if isinstance(v, list):
        return "[" + "; ".join(fmt(x) for x in v) + "]"
    return f"{v:+.4f}" if isinstance(v, float) else str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results/noleak")
    args = ap.parse_args()
    out = args.outdir; ctl = os.path.join(out, "control"); logs = os.path.join(out, "logs")
    for d in (out, ctl, logs):
        os.makedirs(d, exist_ok=True)
    summary = {"protocol": "модель бліцу (train.train_model) і три довідники норм побудовано без гравців, "
                           "які є цільовими в синтетичних партіях; прогони – з кешу рушія тими самими скриптами"}

    # ---- 1. гравці синтетичних партій, що є в навчальному наборі ----
    df = pd.read_parquet(PARQUET)
    df = df[df["target_elo"].between(400, 3000)].reset_index(drop=True)
    P = pd.read_csv(PLAYERS)
    gid = P["site"].str.rsplit("/", n=1).str[-1]
    assert not gid.isin(set(df["game_id"].astype(str))).any(), "синтетична партія є в навчальному наборі"
    excl = sorted(set(P["target"]) & set(df["player"]))
    keep = df[~df["player"].isin(excl)].reset_index(drop=True)
    noleak_parquet = os.path.join(out, "dataset_blitz_2k_noleak.parquet")
    keep.to_parquet(noleak_parquet)
    summary["excluded_players"] = {p: {"games_in_training": int((df["player"] == p).sum()),
                                       "synthetic_sets": sorted(P.loc[P["target"] == p, "set"].unique().tolist())}
                                   for p in excl}
    summary["training_set"] = {"full": [int(df["player"].nunique()), int(len(df))],
                               "noleak": [int(keep["player"].nunique()), int(len(keep))]}
    print(f"цільові гравці синтетичних партій у навчальному наборі: {excl}")
    print(f"навчальний набір: {df['player'].nunique()} гравців / {len(df)} партій -> "
          f"{keep['player'].nunique()} / {len(keep)}")

    # ---- 2. контроль навчання: той самий код на повному наборі = розгорнута модель ----
    ctl_model = os.path.join(ctl, "xgboost_elo_blitz_control.json")
    logged(os.path.join(ctl, "train_control.log"), train_model, PARQUET, ctl_model)
    agg = df.groupby("player")[FEATURE_NAMES].mean()
    a, b = ELOModel(model_path=DEPLOYED["model"]), ELOModel(model_path=ctl_model)
    gap = float(max(np.abs(a.model.predict(agg) - b.model.predict(agg)).max(),
                    np.abs(a.model.predict(df[FEATURE_NAMES]) - b.model.predict(df[FEATURE_NAMES])).max()))
    summary["control_training_max_abs_pred_diff"] = gap
    print(f"контроль навчання: найбільша різниця прогнозів із розгорнутою моделлю {gap:.3g}")

    # ---- 3. контроль довідників: розгорнута модель, повний набір = збережені .pkl ----
    ctl_refs = build_refs(DEPLOYED["model"], PARQUET, ctl, "_control")
    summary["control_references"] = {k: diff_ref(joblib.load(DEPLOYED[k]), joblib.load(p)) for k, p in ctl_refs.items()}
    print(f"контроль довідників: {summary['control_references']}")

    # ---- 4. контроль прогонів: розгорнута модель і збережені довідники = збережені results/*.json ----
    run_all(DEPLOYED, DEPLOYED["model"], ctl, ctl)
    summary["control_runs"] = {f: diff_json(json.load(open(os.path.join("results", f))), json.load(open(os.path.join(ctl, f))))
                               for _, _, f in runs(DEPLOYED, ctl)}
    summary["control_fig_3_10_same_pixels"] = same_pixels(fig_310(os.path.join("results", "synthetic_validation_bands.json"), ctl), FIG_UA)
    print(f"контроль прогонів: {sum(len(v) for v in summary['control_runs'].values())} розбіжностей; "
          f"рисунок 3.10 відтворено попіксельно: {summary['control_fig_3_10_same_pixels']}")

    # ---- 5. без витоку: модель і довідники без цільових гравців ----
    model = os.path.join(out, "xgboost_elo_blitz_noleak.json")
    logged(os.path.join(logs, "train_noleak.log"), train_model, noleak_parquet, model)
    summary["noleak_model_metrics"] = json.load(open(model.replace(".json", "_metrics.json")))
    refs = build_refs(model, noleak_parquet, out, "_noleak")
    sf = joblib.load(refs["signfilter"])
    summary["noleak_sign_excluded"] = sf["sign_excluded"]
    summary["noleak_sign_excluded_equals_SIGN_EXCLUDED"] = sorted(sf["sign_excluded"]) == sorted(SIGN_EXCLUDED)
    print(f"без витоку: критерій знака вилучає {sf['sign_excluded']} "
          f"(= SIGN_EXCLUDED: {summary['noleak_sign_excluded_equals_SIGN_EXCLUDED']})")
    run_all(refs, model, out, logs)
    fig_310(os.path.join(out, "synthetic_validation_bands.json"), out)

    # таблиця 3.11 від моделі не залежить; σ без цільових гравців – для повноти
    # (synthetic_specificity бере σ з усього parquet, без фільтра рейтингу)
    pool = pd.read_parquet(PARQUET)
    sd_full = pool[SV.CATS].std(ddof=0); sd_keep = pool[~pool["player"].isin(excl)][SV.CATS].std(ddof=0)
    summary["table_3_11_sigma_ratio_full_over_noleak"] = (sd_full / sd_keep).to_dict()

    # ---- 6. порівняння чисел тексту: збережені (повний набір) і без витоку ----
    old, new = key_numbers("results"), key_numbers(out)
    rows = [(p, v, dict(flat(new))[p]) for p, v in flat(old)]
    summary["key_numbers"] = {"old": old, "noleak": new}
    with open(os.path.join(out, "comparison_old_vs_noleak.txt"), "w") as f:
        for p, v, w in rows:
            mark = "" if fmt(v) == fmt(w) else "\t*"
            f.write(f"{p}\t{fmt(v)}\t{fmt(w)}{mark}\n")
    json.dump(summary, open(os.path.join(out, "summary_noleak.json"), "w"), ensure_ascii=False, indent=1)
    print(f"-> {out}/summary_noleak.json, {out}/comparison_old_vs_noleak.txt")


if __name__ == "__main__":
    main()
