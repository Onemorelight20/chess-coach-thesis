# Інтелектуальна система шахового тренера на основі методів машинного навчання та пояснювального ШІ

Explainable ML-система, що оцінює рейтинг Elo шахіста за партіями (без доступу до
його реального рейтингу) та визначає його найслабшу тренувальну категорію —
виключно на основі аналізу рушієм Stockfish, без нейромереж (Maia тощо) чи
сторонніх моделей людської гри.

```
PGN → Stockfish (MultiPV) → вектор із 29 ознак → XGBoost (Elo) → SHAP (слабкість) → Streamlit (пазл)
```

## Зміст
- [Конвеєр](#конвеєр)
- [Структура репозиторію](#структура-репозиторію)
- [Дані](#дані)
- [Ознаки (29)](#ознаки-29)
- [Моделі та метрики](#моделі-та-метрики)
- [Відтворення результатів](#відтворення-результатів)
- [Ключові наукові знахідки](#ключові-наукові-знахідки)
- [Зовнішня валідація: Chess.com](#зовнішня-валідація-chesscom)
- [Відомі обмеження](#відомі-обмеження)
- [Встановлення та запуск](#встановлення-та-запуск)

## Конвеєр

1. **Збір даних** (`src/data_layer/collector.py`) — стратифікована по рейтингу
   вибірка партій з Lichess Open Database, окремо на кожен пул часового
   контролю (bullet/blitz/rapid/classical), оскільки Elo і ACPL не порівнювані
   між пулами.
2. **Аналіз рушієм** (`src/engine/analyzer.py`) — Stockfish з MultiPV=5 оцінює
   кожну позицію партії: втрату центіпішаків відносно найкращого ходу, ранг
   зіграного ходу серед кандидатів рушія, критичність позиції.
3. **Вилучення ознак** (`src/features/extractor.py`) — 29 ознак на партію:
   точність по фазах, ситуативна точність, win-probability-зважена точність,
   тактико-позиційні сигнали, керування часом, engine-agreement.
4. **Агрегація** (`src/features/aggregate.py`) — усереднення вектора ознак по N
   партіях гравця (single game — зашумлений зріз навички, див.
   [aggregation curve](#ключові-наукові-знахідки)).
5. **ML-прогноз** (`src/ml/model.py`) — окрема модель XGBoost на кожен пул
   передбачає Elo з агрегованого вектора.
6. **XAI** (SHAP) — визначає ознаку, чий внесок найбільше знижує прогнозований
   Elo — це і є "найслабша тренувальна категорія" гравця.
7. **Рекомендація** (`src/ml/recommender.py`) — підбирає пазл Lichess з
   потрібною темою і рейтингом, близьким до прогнозованого Elo.
8. **Інтерфейс** (`app.py`) — Streamlit: завантаження PGN або пошук партій за
   ніком (Lichess/Chess.com), вибір партій, аналіз, візуалізація SHAP,
   рекомендація.

## Структура репозиторію

```
app.py                        Streamlit-застосунок
requirements.txt
data/                          Датасети ознак (parquet) + індекс пазлів
  dataset_{pool}_2k.parquet    Основні стратифіковані датасети (400-2800 Elo)
  legacy/                      Старі, зміщені по активності датасети (для порівняння калібрування)
  crossplatform/               Chess.com-семпл для зовнішньої валідації (discovery.json + parquet)
  puzzles.parquet              Локальний індекс пазлів Lichess (для точного підбору за рейтингом)
src/
  data_layer/
    collector.py               Збір датасету (стратифікований, резюмований)
    collect_crossplatform.py    Snowball-збір Chess.com-семплу (без bulk-дампу)
    fetch_games.py              Фетчинг партій з Lichess/Chess.com за ніком
    fetch_puzzles.py            Завантаження бази пазлів Lichess
    ingestion.py                Валідація PGN
  engine/analyzer.py            Обгортка Stockfish (MultiPV аналіз)
  features/
    extractor.py                 29 ознак + FEATURE_ORIENTATION + DISPLAY_NAMES
    aggregate.py                  Агрегація N партій в один вектор
  ml/
    train.py                      Тренування XGBoost-моделі на пул
    model.py                      Завантаження моделі + SHAP-прогноз
    recommender.py                 SHAP-слабкість → пазл
    evaluate.py                    Основний евалюаційний харнес (Розділ "Експерименти")
    calibration.py                  Каліброваційне дослідження (стара vs стратиф. вибірка)
    aggregation_methods.py          Порівняння стратегій агрегації (mean/median/...)
    aggregation_curve.py            R²/MAE як функція N ігор на гравця
    learning_curve.py               R²/MAE як функція кількості гравців
    validate_xai.py                 Валідність і надійність SHAP-детектора слабкості
    detector_order.py               Тест-ретест: агрегування перед SHAP vs модальне голосування
    xai_deep.py                     Поглиблений аналіз SHAP (профілі по рейтингу, залежності)
    synthetic_validation.py         Контрольований експеримент: штучно внесена слабкість
    synthetic_aggregate.py          Та сама ін'єкція: пооб'єктна vs агрегована атрибуція
    external_validation.py          Зовнішня валідація: held-out Lichess vs Chess.com + рекалібровка
    production_bias.py              Поза-фолдове зміщення поточної продакшн-моделі за діапазонами Elo
    eval_clip_stats.py              Обґрунтування клипу оцінки ±1000 cp (частка позицій понад 2000 cp)
    plot_architecture.py            Схема конвеєра (fig_architecture.png), без залежності від даних
    models/                        Треновані моделі (xgboost_elo_<pool>.json + метрики)
      chesscom_recalibration_<pool>.pkl  Пост-хок карта Lichess-scale -> Chess.com-scale
    plotstyle.py                    Єдиний стиль усіх фігур
scripts/collect_2k.sh           Резюмований запуск збору всіх 4 пулів
tests/test_features.py          Швидкі unit-тести (без Stockfish, <1с)
results/                         Усі таблиці (.csv/.json) і фігури (.png), що цитуються нижче
```

## Дані

Партії — з публічного дампу Lichess (`lichess_db_standard_rated_2026-01.pgn`).
Наївний збір (найактивніші гравці) дає вибірку, зміщену до середнього рейтингу:
у старому датасеті (`data/legacy/`) лише **1.4%** партій належали гравцям з
рейтингом <900. Це напряму спричиняло помилкове передбачення: гравець з
реальним рейтингом ~700 отримував прогноз ~1600 (докладніше —
[Каліброваційна знахідка](#ключові-наукові-знахідки)).

Виправлення — стратифікований збір (`collector.py --stratified`): гравці
вибираються рівномірно по бінах рейтингу 200 Elo в діапазоні **400–2800**, а не
за активністю. Основні датасети (`scripts/collect_2k.sh`, ~2000
гравців/пул, до 8 партій/гравця):

| пул | рядків | гравців | діапазон Elo |
|---|---|---|---|
| bullet | 16 413 | 2056 | 400–3016 |
| blitz | 17 234 | 2160 | 400–2806 |
| rapid | 15 949 | 1998 | 400–2932 |
| classical | 10 112 | 1270 | 531–2795 |

Збір **резюмований**: `src/data_layer/collector.py` пише чекпойнт кожні 200
партій (`<output>.parquet.partial`); перерваний запуск `bash
scripts/collect_2k.sh` продовжується з чекпойнта, а не з нуля.

## Ознаки (29)

`src/features/extractor.py::FEATURE_NAMES`, три групи:

1. **Базові (21)** — точність (ACPL, inaccuracy/mistake/blunder rate), точність
   по фазах (opening/middlegame/endgame), ситуативна точність
   (conversion/defense ACPL), стиль (engine-match rate, capture/check rate),
   тактико-позиційні сигнали (hanging pieces, king danger, mobility, center
   control, material volatility, pins), керування часом (think-time
   consistency/avg/time-pressure).
2. **Win-probability (3, `wp_*`)** — втрата ймовірності виграшу за хід
   (логістична модель Лічесс/Maia **застосована до оцінки Stockfish** — формула,
   не нейромережа), частка "блондерів" за критерієм ≥10% втрати ймовірності,
   консистентність втрат (`cpl_std`).
3. **Engine agreement (5, MultiPV)** — частка ходів, що збігаються з
   найкращим/топ-3 ходом рушія (`top1_agree`, `top3_agree`), середній ранг
   зіграного ходу (`avg_move_rank`), точність і збіг з найкращим ходом саме в
   "критичних" позиціях, де є єдиний явно найкращий хід (`acpl_critical`,
   `critical_agree`).

Кожна ознака має записану **орієнтацію** (`FEATURE_ORIENTATION`: чи "більше" —
це "гірше" чи "краще"), що критично для коректного визначення слабкості на
основі сирих значень ознак (див. [нижче](#ключові-наукові-знахідки)).

## Моделі та метрики

Player-grouped 5-fold CV (гравець ніколи не потрапляє одночасно в train і
test), XGBoost (`n_estimators=300, max_depth=5, learning_rate=0.05`), окрема
модель на пул (`results/pool_summary.csv`, `src/ml/models/*_metrics.json`):

| пул | MAE (Elo) | RMSE | R² |
|---|---|---|---|
| bullet | 278.5 | 357.2 | 0.694 |
| **blitz** | **277.3** | **355.9** | **0.718** |
| rapid | 267.5 | 345.2 | 0.689 |
| classical | 210.6 | 269.1 | 0.571 |

Порівняно з baseline-моделлю, що завжди передбачає середній Elo вибірки: MAE
583.8, R² ≈ 0 (`results/model_comparison.csv`).

## Відтворення результатів

Кожен артефакт у `results/` відтворюється однією командою (запускати з
активованим `.venv`):

| Команда | Артефакти |
|---|---|
| `python -m src.ml.evaluate data/dataset_blitz_2k.parquet --outdir results` | `feature_signal.csv`, `leakage_comparison.csv`, `model_comparison.csv` + `fig_model_comparison.png`, `vif.csv` + `fig_corr.png`, `ablation.csv`, `ablation_additive.csv` + `fig_feature_groups.png`, `aggregation_curve.csv` + `fig_aggregation.png`, `feature_importance.csv` + `fig_shap_global.png`, `fig_dataset_distribution.png`, `summary.json` |
| `python -m src.ml.evaluate --pools data/dataset_bullet_2k.parquet,data/dataset_blitz_2k.parquet,data/dataset_rapid_2k.parquet,data/dataset_classical_2k.parquet --outdir results` | `pool_summary.csv` + `fig_pool_summary.png` |
| `python -m src.ml.calibration --outdir results` | **Архівний рядок, не виконується.** Вхідний `data/legacy/dataset_blitz_1600p.parquet` втрачено безповоротно (старий екстрактор мав лише 21 спільну ознаку, тому цей набір не можна ні відновити, ні перезібрати поточним конвеєром). Збережені результати попереднього запуску — `calibration_by_band.csv`, `calibration_summary.json`, `fig_calibration.png` — лишаються в `results/` як є й не видалені; для поточної, чинної продакшн-моделі те саме питання (зміщення прогнозу за діапазонами) закриває рядок `production_bias.py` нижче. |
| `python -m src.ml.aggregation_methods data/dataset_blitz_2k.parquet --outdir results` | `aggregation_methods.json` |
| `python -m src.ml.aggregation_curve data/dataset_blitz_2k.parquet --outdir results` | `aggregation_curve.json` + `fig_aggregation_curve.png` (те саме зображення показує `app.py`) |
| `python -m src.ml.aggregation_curve --pools data/dataset_bullet_2k.parquet,data/dataset_blitz_2k.parquet,data/dataset_rapid_2k.parquet,data/dataset_classical_2k.parquet --outdir results` | `aggregation_curve_by_pool.json` + `fig_aggregation_by_pool.png` (та сама mean-агрегація, окремо на кожен з 4 пулів) |
| `python -m src.ml.production_bias --outdir results` | `oof_bias_by_band.json`, `oof_bias_by_band.csv` (поза-фолдові прогнози повної 29-ознакової продакшн-моделі, GroupKFold(5), зміщення за діапазонами рейтингу) |
| `python -m src.ml.eval_clip_stats lichess_db_standard_rated_2026-01.pgn --games 120 --event blitz --outdir results` | `eval_clip_justification.json` (частка позицій понад ±2000 cp і максимальна оцінка на вибірці послідовних партій; обґрунтування обмеження оцінки ±1000 cp, ~5-10 хв) |
| `python -m src.ml.plot_architecture --outdir results` | `fig_architecture.png` (схема конвеєра; не залежить від жодних даних чи попередніх результатів) |
| `python -m src.ml.learning_curve data/dataset_blitz_2k.parquet --outdir results` | `learning_curve.json` + `fig_learning_curve.png` |
| `python -m src.ml.validate_xai data/dataset_blitz_2k.parquet --outdir results` | `xai_validation.json`, `xai_groundtruth_vs_pred.csv` + `fig_xai_confusion.png` |
| `python -m src.ml.detector_order --pool blitz --boot 1000` | `detector_order.json` (порівняння порядку операцій у детекторі слабкості; ~5–8 хв) |
| `python -m src.ml.xai_deep data/dataset_blitz_2k.parquet --outdir results` | `shap_band_profiles.csv` + `fig_shap_bands.png`, `fig_shap_dependence.png`, `xai_reliability_vs_n.csv` + `fig_xai_reliability.png`, `xai_deep_summary.json` |
| `python -m src.ml.synthetic_validation lichess_db_standard_rated_2026-01.pgn --games 80 --elo-bands "1000-1400,1400-1800,1800-2200,2200-3000" --depth 12 --event blitz --outdir results --tag _bands` | `synthetic_validation_bands.json` + `fig_synth_confusion_bands.png` (~1,5–2 год; для тиші термінала — `nohup … & caffeinate -is -w <pid>`) |
| `python -m src.ml.synthetic_aggregate lichess_db_standard_rated_2026-01.pgn --players 15 --per-player 4 --elo-bands "1000-1400,1400-1800,1800-2200,2200-3000" --depth 12 --outdir results --tag _wide` | `synthetic_aggregate_wide.json` (~1,5 год) |
| `python -m src.data_layer.collect_crossplatform discover --seeds <ніки> --pool blitz --per-bin 6 --out data/crossplatform/chesscom_blitz_discovery.json` (резюмований; можна дозапускати з новими `--seeds` для "мостових" гравців у порожні біни) → `python -m src.data_layer.collect_crossplatform analyze data/crossplatform/chesscom_blitz_discovery.json --out data/crossplatform/dataset_chesscom_blitz.parquet` | `data/crossplatform/dataset_chesscom_blitz.parquet` (~10–30 хв мережевих запитів + ~10–20 хв аналізу рушієм) |
| `python -m src.ml.external_validation --pool blitz --outdir results` | `external_validation_bias.csv`, `external_validation_summary.json` + `fig_external_validation_bias.png`, `fig_external_validation_recalibration.png`, `src/ml/models/chesscom_recalibration_blitz.pkl` |
| `python -m src.ml.train data/dataset_<pool>_2k.parquet --out src/ml/models/xgboost_elo_<pool>.json` | Модель + `*_features.pkl` + `*_metrics.json` |
| `bash scripts/collect_2k.sh` | `data/dataset_{bullet,blitz,rapid,classical}_2k.parquet` (резюмований) |
| `pytest -q` | — (модульні тести ознак, <1с) |

Усі скрипти використовують `src/ml/plotstyle.py` для єдиного вигляду фігур.

## Ключові наукові знахідки

**1. Каліброваційна проблема "700 → 1600" та її виправлення.**
Модель, натренована на зміщеній (найактивніші гравці) вибірці, регресує
передбачення до середнього: гравець з реальним Elo ≈678 отримував прогноз
**1337.7** (bias +659.8); гравець ≈2403 — заниження до 1956.6. Стратифікований
збір скорочує це зміщення (≈678→**913.6**), а пост-хок ізотонічна калібровка
(fit виключно на train-фолдах CV, leakage-free) — додатково (≈678→**876.0**).
Сумарно MAE на нижньому кінці рейтингу (<1100 Elo) впало **529.6 → 276.2
(-43%)**. Див. `results/fig_calibration.png`, `results/calibration_by_band.csv`.

**2. Ознаки engine-agreement (MultiPV) несуть основний сигнал.**
Адитивна абляція: базові 21 ознака → R²=0.691, MAE=291.1; +win-probability (3
ознаки) → R²=0.695, MAE=289.0 (майже нейтрально); **+engine agreement (5
ознак) → R²=0.718, MAE=277.3** (Δ=-13.8 MAE). Найсильніші за кореляцією з Elo:
`opening_acpl` (ρ=-0.61), `avg_move_rank` (ρ=-0.57), `acpl` (ρ=-0.56),
`top3_agree` (ρ=+0.54). `results/ablation_additive.csv`,
`results/fig_feature_groups.png`.

**3. Крива навчання вирівнялась — система signal-limited, не data-limited.**
R² зростає з 0.573 (100 гравців) до 0.718 (2160 гравців), але нахил хвоста
становить лише +0.010 R² на +1000 гравців (`results/learning_curve.json`,
вердикт `"FLATTENED … signal-limited, not data-limited"`). Це означає:
подальше нарощування вибірки дає спадну віддачу — пріоритет варто віддавати
новим ознакам, а не новим гравцям.

**4. SHAP-детектор слабкості: помірна надійність, слабка відповідність
агрегованій поведінковій "правді".** Test-retest (Experiment A,
`validate_xai.py`): split-half згода 0.413 проти випадкової 0.142 (2.9×,
Cohen's κ=0.314, 95% CI [0.297; 0.330] за кластерним бутстрепом по гравцях) —
детектор не випадковий, але й не ідеально стабільний партія-до-партії. Обидва
експерименти визначають слабкість у порядку `app.py`: спершу усереднюють вектори
ознак половини, потім пояснюють один агрегований вектор. Якщо натомість брати
модальну відповідь по окремих партіях, κ падає до 0.262 [0.243; 0.279] —
різниця 0.052 [0.036; 0.068], P(Δ>0)=1.000 (`src/ml/detector_order.py`,
`results/detector_order.json`). Experiment B (відповідність population z-score
"істинній" слабкості, усередненій по всіх партіях гравця): top-1 точність лише
0.052 (1.5× випадкового рівня 0.034, κ=0.009), top-2 0.105 — рівно на рівні
тривіального орієнтира "мода" — **набагато слабше**; тут агрегування нічого не
змінює, тобто розбіжність є властивістю механізму, а не порядку обчислень.
Додатково,
`xai_deep.py::reliability_vs_n` показує: агрегована (по z-score) оцінка
слабкості дійсно покращується з числом ігор (0.106→0.173 для N=1→4), тоді як
**"модальний голос" окремих single-game SHAP-визначень залишається на рівні
випадковості** (~0.035–0.04) незалежно від N. Це прямо обґрунтовує архітектурне
рішення `app.py`: агрегувати вектори ознак **перед** запуском SHAP, а не
голосувати по незалежних поясненнях окремих партій.

**5. Розглянута, але не прийнята альтернатива агрегації.**
`aggregation_methods.py` показує, що `mean+std` (58 вимірів: середнє + std на
ознаку) дає дещо кращий R² (0.896 проти 0.889 для звичайного `mean`), але
подвоює розмірність вектора для приросту ~0.007 R². Оскільки це вимагало б
іншої архітектури (тренування на агрегованих, а не пооб'єктних рядках),
застосунок залишає `mean` — прийнятний, добре обґрунтований компроміс.

**6. Синтетична перевірка: детектор надійний для одних категорій і майже
сліпий до інших — на всьому діапазоні рейтингів.** `synthetic_validation.py`
(80 партій, стратифікованих по 20 у смугах 1000–1400/1400–1800/1800–2200/
2200–3000 Elo, глибина рушія 12, критерій ін'єкції — падіння ймовірності
виграшу ≥10%) штучно вносить відому слабкість в одну категорію й перевіряє, чи
SHAP її розпізнає. Чутливість top-1 сильно залежить від категорії: **Opening
ACPL 0.91** (95% CI 0.85–0.96), Middlegame ACPL 0.62, Defensive 0.31, але
**Blunder Rate 0.01** — 77 з 80 внесених "блондер"-ін'єкцій детектор помилково
приписав **Opening Accuracy** (`results/fig_synth_confusion_bands.png`),
Hanging-Piece 0.01 (54 з 80 → Opening), Endgame/Conversion ACPL — 0.00/0.00.
Загальна чутливість top-1 — 0.27 при випадковому рівні 0.14 (95% CI 0.05–0.53).
**Між смугами картина не змінюється** (загальна 0.25–0.31, усі чотири "сліпі"
категорії — нуль у кожній смузі), тож обмеження властиве механізму, а не
вибірці сильних гравців. Причина видима і на `results/vif.csv`: `opening_acpl` —
найважливіша ознака моделі (`xgb_gain`=0.298, найбільша в наборі) і сильно
корелює з рештою ACPL-похідних ознак, тож SHAP схильний приписувати ЛЮБЕ
падіння точності саме їй, навіть коли реальна причина — інша категорія.
`results/synthetic_validation_bands.json`.

**7. Агрегування підвищує надійність, але не валідність.**
`synthetic_aggregate.py` (15 умовних гравців по 4 партії близького рейтингу, та
сама слабкість внесена в усі 4) порівнює пооб'єктний і агрегований порядок
атрибуції на однакових оброблених партіях. Агрегування **не усуває сліпої
зони**: усі 15 "блондер"-ін'єкцій знову приписані Opening ACPL, Conversion і
Hanging-Piece лишаються 0.00. Натомість воно **підсилює наявний перекіс** —
Opening 0.87→1.00, Middlegame 0.57→0.87, тоді як Defensive падає 0.32→0.07
(7 з 15 → Middlegame); загальна 0.25→0.28. Тобто той самий порядок операцій, що
дав +0.052 κ у тест-ретесті (знахідка 4), не покращує здатність знайти реально
внесену слабкість: надійність і змістовна правильність тут дисоціюють.
`results/synthetic_aggregate_wide.json`.

## Зовнішня валідація: Chess.com

Мотивація: користувач, що тестував систему на власному профілі Chess.com,
повідомив про стабільно завищений прогноз. Оскільки модель тренована виключно
на Lichess, потрібно було кількісно відокремити (A) відоме залишкове зміщення
моделі від (B) різниці рейтингових шкал платформ — і, якщо можливо, виправити
другу причину.

**Методологія збору Chess.com-семплу.** Chess.com не публікує bulk-дамп партій
(на відміну від Lichess), тому `collect_crossplatform.py` збирає дані
snowball-краулінгом: кожна отримана партія містить нік і рейтинг суперника,
тож BFS по суперниках від кількох seed-акаунтів природно відкриває нових
гравців без жодного завантаження повної бази. Гравці зберігаються лише коли
потрапляють у недозаповнений рейтинг-бін (стратифікація 400–2800, крок 200,
адаптована з `collector.py` під фронтир-краулінг замість лінійного сканування
файлу). **Знахідка методології**: чистий snowball із казуальних акаунтів
застряг на низьких/середніх рейтингах (асортативне змішування — казуальні
гравці здебільшого грають з такими ж) — біни 1400+ лишались порожніми після
230+ відвідувань. Виправлено додаванням "мостових" seed-ніків із публічних
списків титулованих гравців Chess.com (`/pub/titled/{CM,NM,WCM,FM}`, рейтинг
~1900–2700) — після цього 10.5 з 12 бінів заповнились (61 гравець, 513–2790
Elo); лише бін 1400–1600 лишився порожнім — по суті "мертва зона" між
казуальною й титулованою мережами суперників.

**Частина A — наскільки завищено, і чому.** Порівняння зміщення (прогноз −
факт) на двох групах, обидві через *точно той самий* живий пайплайн
(`EngineAnalyzer(time_limit=0.05)`, заморожена blitz-модель):

| група | n | середнє зміщення | MAE |
|---|---|---|---|
| Lichess, небачені гравці | 12 | **+126** | 211 |
| Chess.com-семпл | 61 | **+373** | 434 |

`results/fig_external_validation_bias.png` показує обидві групи разом:
однаковий спадний тренд (регресія до середнього — та сама причина, що й на
Lichess), але Chess.com-точки систематично зсунуті ВИЩЕ на кожному рівні
рейтингу — чіткий, візуально відокремлюваний доказ різниці шкал поверх уже
відомого залишкового зміщення моделі.

**Частина B — пост-хок рекалібровка.** K-fold (5), leakage-free порівняння
ізотонічної та лінійної карти "сирий прогноз (шкала Lichess) → рейтинг
Chess.com", натреновані виключно на Chess.com-семплі; сама модель залишається
замороженою і не змінюється:

| | сирий прогноз | ізотонічна | лінійна |
|---|---|---|---|
| MAE | 434.4 | **193.3** | 223.0 |
| середнє зміщення | +372.8 | **+17.9** | +0.5 |

Ізотонічна калібровка перемагає за MAE (**−55%**, з 434 до 193) і практично
усуває зміщення. `results/fig_external_validation_recalibration.png` показує
до/після: "сирі" точки лежать далеко вище діагоналі (пряме передбачення vs
факт), рекалібровані — щільно вздовж неї. Фінальна карта, натренована на всіх
61 гравцях, збережена в `src/ml/models/chesscom_recalibration_blitz.pkl` і
**автоматично застосовується в `app.py`**, коли партії отримано з Chess.com:
показується рекалібрований Elo як основний, із підписом, що вказує на сирий
Lichess-масштабний прогноз і використаний метод корекції. Для пулів без
власної карти (поки лише blitz має достатній семпл) `app.py` показує чіткий
дисклеймер, що прогноз — у шкалі Lichess.

## Відомі обмеження

- **Швидкість аналізу.** Дані для тренування та прогнози в `app.py` рахуються
  при `time_limit=0.05с/хід` (не фіксована глибина) — це узгоджено між
  збором даних і інференсом, але є неглибоким і трохи шумним аналізом.
  `synthetic_validation.py` натомість використовує фіксовану глибину 12 —
  глибший і відтворюваніший, але лише для контрольованого експерименту, не для
  продакшн-моделі.
- **Шкала рейтингу — лише Lichess.** Прогноз відповідає рейтинговій шкалі
  Lichess; рейтинги інших платформ систематично відрізняються. Кількісно
  підтверджено в [Зовнішній валідації](#зовнішня-валідація-chesscom): на 61
  реальному гравці Chess.com середнє зміщення +373 Elo проти +126 на небачених
  гравцях Lichess (та сама модель, той сам пайплайн) — тобто ефект шкали
  платформи додає приблизно втричі більше зміщення, ніж власне залишкове
  зміщення моделі. Для blitz це виправлено пост-хок рекалібровкою
  (MAE 434→193); для bullet/rapid/classical карти ще немає.
- **Тонкі хвости розподілу залишаються.** Навіть після стратифікації дуже
  низькі (<800) і дуже високі (>2300) рейтинги — рідкісні в реальних даних
  Lichess (особливо classical: лише 133 партії <800 Elo), тож калібрування на
  екстремумах — найменш надійне.
- **Невідповідність train/inference.** Модель тренується на векторах ознак
  ОДНІЄЇ партії, а `app.py` на інференсі усереднює вектори N партій. Це
  свідомий вибір, обґрунтований `aggregation_curve.py`/`learning_curve.py`
  (агрегація суттєво знижує шум), але є архітектурною відмінністю, яку варто
  проговорити явно.
- **Мультиколінеарність.** Кілька ACPL-похідних ознак сильно корелюють
  (VIF(`acpl`)=44.4, VIF(`avg_move_rank`)=30.8, `results/vif.csv`) — очікувано
  для дерев (XGBoost стійкий до цього для прогнозу), але ускладнює
  інтерпретацію ІНДИВІДУАЛЬНИХ SHAP-внесків корельованих ознак.
- **SHAP-детектор слабкості надійний нерівномірно між категоріями.**
  Синтетична перевірка (знахідка 6 вище) підтверджує це кількісно: top-1
  чутливість 0.91 для Opening ACPL, але 0.01 для Blunder Rate й Hanging-Piece
  Rate (їх систематично "перекриває" `opening_acpl` — найважливіша й
  найсильніше корельована ознака моделі). У поєднанні з Experiment B
  (`validate_xai.py`, знахідка 4) — слабка відповідність агрегованій
  поведінковій "правді" — це означає, що рекомендації пазлів найменш надійні
  саме для тактичних категорій (блондери, "висячі" фігури), де користувачу
  найбільше й потрібна конкретна вправа.

## Встановлення та запуск

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export STOCKFISH_PATH=/opt/homebrew/bin/stockfish   # або де встановлено Stockfish

streamlit run app.py
```

Треновані моделі (`src/ml/models/xgboost_elo_<pool>.json`) уже включені; для
перетренування на нових даних — див. таблицю відтворення результатів вище.
