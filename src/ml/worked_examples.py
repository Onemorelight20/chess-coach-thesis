"""
Таблиця 3.24: приклади роботи методу для трьох гравців пулу бліц.

Для кожного гравця беруться його партії з навчального набору (до 8), ознаки
усереднюються, розгорнута модель дає прогноз і три напрями за правилом (2.7),
а рекомендатор добирає під кожен напрям задачу з локального індексу Lichess
(data/puzzles.parquet) із рейтингом у вікні навколо прогнозу — саме так, як
це робить app.py. Рейтинг задачі в таблиці — реальний рейтинг знайденої задачі.

Usage:
    python -m src.ml.worked_examples --players Aldo333,AB9920,AAPA23 --outdir results
"""
import argparse
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.features.extractor import FEATURE_NAMES, DISPLAY_NAMES
from src.features.aggregate import aggregate_features
from src.ml.model import ELOModel
from src.ml.recommender import RecommenderEngine

UA = {  # підписи напрямів у роботі (таблиця 2.2)
    "opening_acpl": "Дебютна точність", "middlegame_acpl": "Точність у міттельшпілі",
    "endgame_acpl": "Точність в ендшпілі", "conversion_acpl": "Реалізація переваги",
    "defense_acpl": "Стійкість у захисті", "acpl_critical": "Точність у критичних позиціях",
    "blunder_rate": "Частка грубих помилок", "mistake_rate": "Частка помилок",
    "inaccuracy_rate": "Частка неточностей", "hanging_rate": "Висячі фігури",
    "pin_exposure": "Схильність до зв’язок", "king_danger": "Небезпека для короля",
    "center_control_avg": "Контроль центру", "mobility_avg": "Рухливість фігур",
    "material_volatility": "Втрати матеріалу", "capture_rate": "Частка взяттів",
    "check_given_rate": "Частка шахів",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="blitz")
    ap.add_argument("--players", default="Aldo333,AB9920,AAPA23")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    df = pd.read_parquet(f"data/dataset_{args.pool}_2k.parquet")
    model = ELOModel(pool=args.pool)
    rec = RecommenderEngine(seed=args.seed)
    rows = []
    for player in args.players.split(","):
        g = df[df["player"] == player].head(args.games)
        if g.empty:
            print(f"! {player}: немає партій у наборі"); continue
        feats = aggregate_features(g[FEATURE_NAMES].to_dict("records"), "mean")
        out = model.predict_and_explain(feats, k=3)
        plan = rec.get_plan(out["predicted_elo"], out["weaknesses"])
        fact = int(round(g["target_elo"].median()))
        print(f"\n{player}: факт {fact}, прогноз {out['predicted_elo']} "
              f"({len(g)} партій, правило {out['weakness_rule']}, "
              f"відрив {out['weakest_margin']:.2f})")
        for w, r in zip(out["weaknesses"], plan):
            rows.append({"гравець": player, "факт": fact, "прогноз": out["predicted_elo"],
                         "№": r["priority"], "напрям": w["feature"],
                         "напрям_укр": UA.get(w["feature"], DISPLAY_NAMES.get(w["feature"])),
                         "z": round(w["score"], 2), "тема": r["puzzle_theme"],
                         "задача": r.get("puzzle_id"), "рейт.": r.get("puzzle_rating"),
                         "джерело": r.get("source")})
            print(f"  {r['priority']}. {UA.get(w['feature'], w['feature']):<32} z={w['score']:+.2f}"
                  f"  -> {r['puzzle_theme']}, задача {r.get('puzzle_id')} ({r.get('puzzle_rating')})")
    out_df = pd.DataFrame(rows)
    path = os.path.join(args.outdir, "worked_examples.csv")
    out_df.to_csv(path, index=False)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
