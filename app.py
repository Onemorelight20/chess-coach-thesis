import io
import os
import sys
from collections import Counter

import altair as alt
import chess.pgn
import joblib
import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.data_layer.ingestion import is_valid_game
from src.data_layer.collector import pool_of
from src.data_layer.fetch_games import (
    fetch_user_games, FetchError, LICHESS_POOLS, CHESSCOM_POOLS,
)
from src.engine.analyzer import EngineAnalyzer
from src.features.extractor import FeatureExtractor, DISPLAY_NAMES
from src.features.aggregate import aggregate_features
from src.ml.model import ELOModel
from src.ml.recommender import RecommenderEngine

RECAL_PATH_TMPL = os.path.join(os.path.dirname(__file__), "src", "ml", "models",
                               "chesscom_recalibration_{pool}.pkl")


@st.cache_resource(show_spinner=False)
def _load_chesscom_recalibration(pool: str):
    """
    Post-hoc map (raw Lichess-scale prediction -> Chess.com-scale Elo); see
    external_validation.py. None if this pool has no map yet.
    """
    path = RECAL_PATH_TMPL.format(pool=pool)
    if not os.path.exists(path):
        return None
    return joblib.load(path)

st.set_page_config(page_title="AI Chess coach", page_icon="♟️", layout="wide")

POOLS = ["bullet", "blitz", "rapid", "classical"]

st.title("AI Chess coach")
st.markdown("""
This tool runs an explainable ML pipeline over **several of a player's games**:
1. **Engine analysis** - Stockfish evaluates every move of each game.
2. **Feature extraction**.
3. **Aggregation** — the per-game vectors are averaged, since a single game is a noisy snapshot of skill.
4. **ML prediction + SHAP** — a per-game-type XGBoost model estimates Elo, then the feature
   contributions are compared against players at the same predicted level to pick the weakest areas.
5. **Adaptive training** — recommends an Elo-matched Lichess puzzle for each area.
""")

st.info(
    "ℹ️ Accuracy improves sharply with more games per player. "
    "A separate model is trained **per time-control pool** (bullet, blitz, rapid, classical), since Elo "
    "is not comparable across them."
)


def parse_increment(time_control: str) -> float:
    if not time_control or "+" not in time_control:
        return 0.0
    try:
        return float(time_control.split("+")[1])
    except (ValueError, IndexError):
        return 0.0


def parse_all_games(pgn_text: str):
    games, stream = [], io.StringIO(pgn_text)
    while True:
        try:
            g = chess.pgn.read_game(stream)
        except Exception:  # noqa: BLE001
            break
        if g is None:
            break
        if is_valid_game(g):
            games.append(g)
    return games


@st.cache_data(ttl=600, show_spinner=False)
def cached_fetch(platform: str, username: str, pool: str, max_games: int, cursor):
    """Thin cached wrapper so Streamlit reruns don't re-hit the APIs."""
    return fetch_user_games(platform, username, pool, max_games, cursor)


def analyze_and_render(games_with_color, target, pool, key, platform=None):
    """
    Shared analysis + results UI. `games_with_color` is a list of
    (chess.pgn.Game, color); `platform` controls the rating-scale disclaimer and
    whether the Chess.com recalibration is applied.
    """
    n_games = len(games_with_color)
    if not st.button(f"Analyze {n_games} game(s)", type="primary", key=f"analyze_{key}"):
        return

    with st.status(f"Analysing {n_games} game(s)...", expanded=True) as status:
        try:
            analyzer = EngineAnalyzer(time_limit=0.05)  # matches the training-data budget
            feats_list = []
            progress = st.empty()  # one line, updated in place
            for i, (g, color) in enumerate(games_with_color, 1):
                progress.write(f"\U0001f7e2 Game {i}/{n_games}: Stockfish + feature extraction...")
                inc = parse_increment(g.headers.get("TimeControl", "-"))
                results = analyzer.analyze_game(g)
                feats_list.append(
                    FeatureExtractor(target_player_color=color).extract(results, increment=inc))
            analyzer.close()

            progress.write("\U0001f7e2 Aggregating (mean) + predicting Elo + SHAP...")
            features = aggregate_features(feats_list, method="mean")
            model = ELOModel(pool=pool)
            prediction_data = model.predict_and_explain(features)
            raw_elo = prediction_data["predicted_elo"]  # always Lichess-scale

            # Puzzle matching below keeps the RAW Lichess-scale number on purpose.
            display_elo, recal_info = raw_elo, None
            if platform == "Chess.com":
                recal = _load_chesscom_recalibration(pool)
                if recal is not None:
                    x = [[raw_elo]] if recal["method"] == "linear" else [raw_elo]
                    display_elo = int(round(float(recal["model"].predict(x)[0])))
                    recal_info = recal

            progress.write("\U0001f7e2 Generating adaptive recommendation...")
            plan = RecommenderEngine().get_plan(
                predicted_elo=raw_elo,
                weaknesses=prediction_data["weaknesses"],
            )
            status.update(label="Pipeline complete!", state="complete", expanded=False)
        except Exception as e:  # noqa: BLE001
            status.update(label="Error in pipeline", state="error")
            st.error(f"Analysis failed: {e}")
            return

    # ------------------------------------------------------------ results
    st.divider()
    st.markdown(f"### Results for **{target}**")
    cols = st.columns(3)
    cols[0].metric("\U0001f9e0 Predicted Elo", display_elo)
    cols[1].metric("Games analysed", n_games)
    cols[2].metric("Game type", pool)

    if prediction_data.get("is_mock", False):
        st.warning("⚠️ Mock model. Run the collector + `python -m src.ml.train` for real predictions.")
    elif model.model_path and f"_{pool}.json" not in model.model_path:
        st.caption(f"No {pool}-specific model yet — using the blitz model as a fallback.")

    # Rating-scale disclaimer: the model is trained exclusively on Lichess data.
    if recal_info is not None:
        st.caption("⚖️ Adjusted for Chess.com's rating scale.")
    elif platform == "Chess.com":
        st.caption("⚖️ No Chess.com adjustment yet for this game type — expect this number "
                   "to read higher than your Chess.com rating.")
    else:
        st.caption("⚖️ Predicted Elo is on the **Lichess** rating scale.")

    ws = prediction_data["weaknesses"]
    st.info("**Priority training areas:** " + " · ".join(w["display"] for w in ws))
    if prediction_data.get("weakness_rule") == "band-standardised":
        st.caption(
            "Ranked by how unusual each contribution is **for players of your predicted "
            "level**, not by its raw size. Several areas are listed on purpose: on split "
            "halves of the same player the single top pick repeats 11% of the time, the "
            "top-3 set 57%."
        )

    st.subheader("SHAP Interpretability")
    st.markdown("How each feature pushed the predicted Elo up (positive) or down (negative).")
    shap_df = pd.DataFrame(
        [(DISPLAY_NAMES.get(k, k), v) for k, v in prediction_data["shap_values"].items()],
        columns=["Feature", "SHAP Impact"],
    )
    shap_chart = (
        alt.Chart(shap_df)
        .mark_bar(color="#FF4B4B")
        .encode(
            x=alt.X("SHAP Impact:Q", title="Impact on predicted Elo"),
            y=alt.Y("Feature:N", sort="x", title=None),
            tooltip=["Feature", "SHAP Impact"],
        )
        .properties(height=520)
    )
    st.altair_chart(shap_chart, width="stretch")

    st.divider()
    st.markdown("### Next Steps \U0001f680")
    if recal_info is not None:
        st.caption(
            f"ℹ️ Your Elo above is adjusted for Chess.com's scale. The puzzle below is matched "
            f"to your Lichess-equivalent rating (~{raw_elo}) instead, since Lichess's puzzles "
            "use that same scale."
        )
    for rec in plan:
        area = DISPLAY_NAMES.get(rec["weakness_identified"], rec["weakness_identified"])
        st.markdown(f"**{rec['priority']}. {area}**")
        if rec.get("puzzle_id"):
            st.markdown(
                f"[\U0001f449 Solve this {rec['puzzle_theme']} puzzle "
                f"(rating {rec['puzzle_rating']}) on Lichess]({rec['link']})"
            )
            st.caption("Puzzle themes: " + ", ".join(rec["puzzle_themes"]))
        else:
            st.markdown(
                f"[\U0001f449 Train {rec['puzzle_theme']} puzzles on Lichess]({rec['link']})"
            )


def render_fetch_tab():
    st.caption("Pull a player's recent games straight from Lichess or Chess.com — no copy-paste needed.")
    c1, c2 = st.columns([1, 2])
    platform = c1.radio("Platform", ["Lichess", "Chess.com"], horizontal=True, key="fetch_platform")
    username = c2.text_input("Username", key="fetch_username",
                             placeholder="e.g. DrNykterstein").strip()

    pool_options = LICHESS_POOLS if platform == "Lichess" else CHESSCOM_POOLS
    fmt = st.selectbox("Time format", pool_options, key="fetch_pool")
    if platform == "Chess.com":
        st.caption("Chess.com has no *classical* pool, so only bullet/blitz/rapid are available here.")

    current_key = (platform, username.lower(), fmt)

    if st.button("Fetch games", type="primary", key="fetch_btn", disabled=not username):
        try:
            with st.spinner(f"Fetching {fmt} games for {username}..."):
                page = cached_fetch(platform, username, fmt, 10, None)
            st.session_state["fetch_key"] = current_key
            st.session_state["fetch_games"] = page.games
            st.session_state["fetch_cursor"] = page.cursor
            st.session_state["fetch_has_more"] = page.has_more
        except FetchError as e:
            st.session_state.pop("fetch_key", None)
            st.error(str(e))
            return

    # Only show a table for the inputs that were actually fetched.
    if st.session_state.get("fetch_key") != current_key:
        st.caption("Pick a platform, time format and username, then **Fetch games**.")
        return

    games = st.session_state.get("fetch_games") or []
    if not games:
        st.warning(f"No {fmt} games found for **{username}** on {platform}.")
        return

    df = pd.DataFrame([{
        "Time": g["time_control"],
        "Opponent": g["opponent"],
        "Opp. Elo": g["opponent_elo"],
        "Result": g["result"],
        "Color": g["color"].capitalize(),
        "Date": g["played_at"],
        "Note": f"⚠️ {g['note']}" if g.get("note") else "",
    } for g in games])
    df["Opp. Elo"] = df["Opp. Elo"].astype("Int64")

    event = st.dataframe(
        df, hide_index=True, width="stretch",
        on_select="rerun", selection_mode="multi-row", key="games_table",
        column_config={
            "Opp. Elo": st.column_config.NumberColumn(format="%d"),
            "Result": st.column_config.TextColumn(width="small"),
            "Note": st.column_config.TextColumn(
                "Note", help="Games flagged here can't be analysed (e.g. fewer than 10 moves) "
                             "and are skipped if selected."),
        },
    )
    selected_rows = event.selection.rows

    lc, rc = st.columns([1, 3])
    if st.session_state.get("fetch_has_more"):
        if lc.button("Load more", key="load_more"):
            try:
                with st.spinner("Fetching more games..."):
                    page = cached_fetch(platform, username, fmt, 10,
                                        st.session_state.get("fetch_cursor"))
                st.session_state["fetch_games"] = games + page.games
                st.session_state["fetch_cursor"] = page.cursor
                st.session_state["fetch_has_more"] = page.has_more
                st.rerun()
            except FetchError as e:
                st.error(str(e))
    rc.caption(f"Showing **{len(games)}** games · **{len(selected_rows)}** selected.")

    if not selected_rows:
        st.info("Select one or more games above (checkboxes), then analyze them as a batch.")
        return

    games_with_color = []
    for i in selected_rows:
        g = games[i]
        if g.get("note"):  # flagged as unanalysable in the Note column
            continue
        parsed = chess.pgn.read_game(io.StringIO(g["pgn"]))
        if parsed is not None:
            games_with_color.append((parsed, g["color"]))

    if not games_with_color:
        st.warning("None of the selected games can be analysed — pick rows without a ⚠️ note.")
        return

    analyze_and_render(games_with_color, username, fmt, key="fetch", platform=platform)


def render_manual_tab():
    uploaded_file = st.file_uploader("Upload PGN (one or more games of the same player)",
                                     type=["pgn"], key="manual_upload")
    pgn_text = st.text_area("Or paste PGN text here:", height=150, key="manual_text")
    pgn_data = uploaded_file.getvalue().decode("utf-8") if uploaded_file is not None else pgn_text

    games = parse_all_games(pgn_data) if pgn_data.strip() else []
    if not games:
        st.caption("Upload or paste at least one valid game (≥10 moves) to begin.")
        return

    names = Counter()
    for g in games:
        for side in ("White", "Black"):
            n = g.headers.get(side, "?")
            if n and n != "?":
                names[n] += 1
    options = [n for n, _ in names.most_common()] or ["?"]

    c1, c2, c3 = st.columns([2, 1, 1])
    target = c1.selectbox("Player to analyze", options, key="manual_target")
    player_games = [g for g in games
                    if target in (g.headers.get("White"), g.headers.get("Black"))]

    pool_counts = Counter(pool_of(g.headers.get("TimeControl", "-")) for g in player_games)
    detected = pool_counts.most_common(1)[0][0] if pool_counts else "blitz"
    if detected not in POOLS:
        detected = "blitz"
    pool = c2.selectbox("Game type", POOLS, index=POOLS.index(detected), key="manual_pool")

    max_n = max(1, len(player_games))
    n_games = c3.slider("Games (N)", 1, max_n, min(8, max_n)) if max_n > 1 else 1

    st.caption(f"Found **{len(player_games)}** game(s) for **{target}** · "
               f"analysing **{n_games}** as **{pool}** · more games → more reliable.")

    selected = player_games[:n_games]
    games_with_color = [
        (g, "white" if g.headers.get("White") == target else "black") for g in selected
    ]
    analyze_and_render(games_with_color, target, pool, key="manual")


# ---------------------------------------------------------------- input tabs
tab_fetch, tab_manual = st.tabs(["🔎 Fetch by username", "📋 Upload / paste PGN"])
with tab_fetch:
    render_fetch_tab()
with tab_manual:
    render_manual_tab()
