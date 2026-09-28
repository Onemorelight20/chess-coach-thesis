"""
Змішаний навчальний набір для моделі рейтингу (підрозділ 2.3.1).

Застосунок подає моделі усереднений за серією з N партій вектор ознак, а N
задає користувач (від однієї партії до восьми). Модель, навчена лише на
одиничних партіях, на інференсі бачить менш розсіяні вектори й зміщує прогноз
до середнього; модель, навчена на векторах фіксованого N, програє на інших N.
Тому навчальний набір змішаний: усі одиничні партії плюс по REPS випадкових
підвибірок з N партій на гравця для кожного N з LEVELS. Ознаки підвибірки
усереднюються так само, як у застосунку (середнє, пропуски ігноруються), ціль
рядка — середній рейтинг гравця за партіями підвибірки (для одиничної партії —
його рейтинг у цій партії). Групою для перехресної перевірки лишається гравець,
тож жодна підвибірка не перетинає межу між навчальними й тестовими гравцями.

    from src.ml.mixed_training import mixed_training_set
    X, y, groups, n_games = mixed_training_set(df)          # df: FEATURE_NAMES + target_elo + player
"""
import warnings

import numpy as np
import pandas as pd

from src.features.extractor import FEATURE_NAMES

LEVELS = (2, 3, 5, 8)   # довжини серій, доданих до одиничних партій
REPS = 2                # підвибірок на гравця для кожної довжини
SEED = 0


def mixed_training_set(df, feature_names=None, levels=LEVELS, reps=REPS, seed=SEED):
    """Повертає (X, y, groups, n_games) для df з ознаками, target_elo і player."""
    feats = list(feature_names or FEATURE_NAMES)
    rng = np.random.default_rng(seed)
    X_parts = [df[feats].to_numpy(dtype=float)]
    y_parts = [df["target_elo"].to_numpy(dtype=float)]
    g_parts = [df["player"].to_numpy()]
    n_parts = [np.ones(len(df), dtype=int)]
    agg_X, agg_y, agg_g, agg_n = [], [], [], []
    for player, g in df.groupby("player", sort=True):
        vals = g[feats].to_numpy(dtype=float)
        elo = g["target_elo"].to_numpy(dtype=float)
        for N in levels:
            if len(g) < N:
                continue
            for _ in range(reps):
                idx = rng.choice(len(g), N, replace=False)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)   # стовпець без значень -> NaN, XGBoost це обробляє
                    agg_X.append(np.nanmean(vals[idx], axis=0))
                agg_y.append(elo[idx].mean())
                agg_g.append(player)
                agg_n.append(N)
    if agg_X:
        X_parts.append(np.vstack(agg_X)); y_parts.append(np.array(agg_y))
        g_parts.append(np.array(agg_g, dtype=object)); n_parts.append(np.array(agg_n))
    X = pd.DataFrame(np.vstack(X_parts), columns=feats)
    y = pd.Series(np.concatenate(y_parts), name="target_elo")
    groups = pd.Series(np.concatenate(g_parts), name="player")
    n_games = pd.Series(np.concatenate(n_parts), name="n_games")
    return X, y, groups, n_games


def describe(n_games):
    """Короткий підпис складу набору для журналів: '17 234 одиничних + 16 872 серій (N=2,3,5,8)'."""
    n1 = int((n_games == 1).sum()); na = int((n_games > 1).sum())
    lv = ",".join(str(int(v)) for v in sorted(set(n_games[n_games > 1])))
    return f"{n1:,} одиничних партій + {na:,} серій (N={lv})".replace(",", " ")
