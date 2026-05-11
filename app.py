"""
app.py — WNBA Season Dashboard (Streamlit)

Run with: streamlit run app.py
"""

import base64
import io
import json
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from fetch_data import (
    TEAM_NAMES,
    get_schedule,
    get_team_rosters,
    get_player_lebron,
    get_team_logo,
    prefetch_all_logos,
)
def _norm(name: str) -> str:
    """Strip accents for accent-insensitive name matching.
    Also repairs UTF-8 mojibake (UTF-8 bytes misread as Latin-1) before stripping."""
    try:
        # If the string was decoded as Latin-1 when it should be UTF-8,
        # re-encoding as Latin-1 gives back the original bytes, then UTF-8 decode fixes it.
        name = name.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")


from model import (
    LEAGUE_AVG_ORTG,
    SimResults,
    get_next_gamedays,
    project_game_total,
    simulate_season,
    win_probability,
    win_prob_to_spread,
)

SEASON = 2026

st.set_page_config(
    page_title="WNBA Season Dashboard",
    page_icon="🏀",
    layout="wide",
)

# ── Styles ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.section-header {
    font-size: 1.4rem;
    font-weight: 700;
    color: #ff6b35;
    margin-bottom: 0.5rem;
    border-bottom: 2px solid #ff6b35;
    padding-bottom: 4px;
}
.game-card {
    background: #1e1e2e;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
}
.favored { color: #4ade80; font-weight: 600; }
.underdog { color: #94a3b8; }
</style>
""", unsafe_allow_html=True)

# ── Data Loading ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Loading schedule...")
def load_schedule():
    return get_schedule(SEASON)


def load_team_lebron():
    records = load_player_lebron_data()
    team_lbr: dict[str, float] = {}
    for r in records:
        team = r.get("team", "")
        if not team:
            continue
        team_lbr[team] = team_lbr.get(team, 0.0) + r["lebron"] * r["mpg"] / 40
    return team_lbr


@st.cache_data(ttl=3600, show_spinner="Loading rosters...")
def load_rosters() -> dict:
    rosters = get_team_rosters(SEASON)
    # Convert DataFrames to JSON-serialisable dicts for Streamlit cache
    return {abbr: df.to_dict(orient="records") for abbr, df in rosters.items()}


SEASON_GAMES = 44  # 2026 WNBA regular season length

# Google Sheets rotation minutes — base spreadsheet ID
_SHEETS_ID = "15lSFSwU_eyt8kE2Ti7SYWKq9eLqKOjJnfZ60Z3USxRI"


def _import_minutes_from_sheets(team_abbr: str, roster_players: list[dict]) -> tuple[int, list[str]]:
    """
    Fetch player minutes from the Google Sheets MPG doc (one tab per team)
    and write them into session state for the given team.
    Matches on last name (case-insensitive).
    Slash-separated entries like 'Clark/Prosper' split minutes evenly.
    Returns (count_matched, unmatched_names).
    """
    from io import StringIO
    import urllib.request
    import urllib.parse

    # Try team abbreviation first, then full team name as sheet tab name
    team_full = TEAM_NAMES.get(team_abbr, team_abbr)
    sheet_df = None
    last_err = None
    for sheet_name in [team_abbr, team_full]:
        encoded = urllib.parse.quote(sheet_name)
        url = (
            f"https://docs.google.com/spreadsheets/d/{_SHEETS_ID}"
            f"/gviz/tq?tqx=out:csv&sheet={encoded}"
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = resp.read().decode("utf-8")
            candidate = pd.read_csv(StringIO(raw))
            # A valid sheet has a "Player Name" column; error pages won't
            if "Player Name" in candidate.columns:
                sheet_df = candidate
                break
        except Exception as e:
            last_err = e

    if sheet_df is None:
        raise RuntimeError(
            f"Could not find a sheet tab for '{team_abbr}' or '{team_full}'. "
            f"Last error: {last_err}"
        )

    # Build last-name → full player name lookup for this team's roster
    last_to_full: dict[str, str] = {}
    for p in roster_players:
        if "player" not in p:
            continue
        full = p["player"]
        last = full.split()[-1].lower()
        last_to_full[last] = full

    matched = 0
    unmatched: list[str] = []

    for _, row in sheet_df.iterrows():
        cell = str(row.get("Player Name", "")).strip()
        if not cell or cell.lower() == "total":
            continue
        try:
            total_min = float(row.get("Total", 0) or 0)
        except (ValueError, TypeError):
            continue
        if total_min <= 0:
            continue

        parts = [n.strip() for n in cell.split("/")]
        per_player = round(total_min / len(parts), 1)

        for name in parts:
            full = last_to_full.get(name.lower())
            if full:
                st.session_state[f"rot_{team_abbr}_{full}"] = per_player
                matched += 1
            else:
                unmatched.append(name)

    return matched, unmatched


@st.cache_data(ttl=3600, show_spinner="Loading player ratings...")
def load_player_lebron_data() -> list[dict]:
    df = get_player_lebron()
    if df.empty or "player" not in df.columns:
        return []
    df = df.copy()
    df["minutes"] = pd.to_numeric(df.get("minutes", 0), errors="coerce").fillna(0)
    df["lebron_war"] = pd.to_numeric(df.get("lebron_war", 0), errors="coerce").fillna(0)
    if "mpg" not in df.columns:
        df["mpg"] = df["minutes"] / SEASON_GAMES
    df["mpg"] = pd.to_numeric(df["mpg"], errors="coerce").fillna(0).round(1)
    df["o_lebron"] = pd.to_numeric(df.get("o_lebron", 0), errors="coerce").fillna(0)
    df["d_lebron"] = pd.to_numeric(df.get("d_lebron", 0), errors="coerce").fillna(0)
    df["lebron"] = df["o_lebron"] + df["d_lebron"]
    return df[["player", "team", "minutes", "mpg", "lebron_war",
               "o_lebron", "d_lebron", "lebron"]].to_dict(orient="records")


@st.cache_data(ttl=3600, show_spinner="Running simulations...")
def load_simulation(schedule_json: str, team_lebron_json: str) -> SimResults:
    import json
    schedule_df = pd.DataFrame(json.loads(schedule_json))
    team_lebron = json.loads(team_lebron_json)
    return simulate_season(schedule_df, team_lebron)


def logo_image(team: str, size: int = 40):
    path = get_team_logo(team)
    if path and path.exists():
        img = Image.open(path).convert("RGBA")
        img.thumbnail((size, size))
        return img
    return None


# ── Rotation persistence helpers ─────────────────────────────────────────────

_ROTATION_SAVE_PATH = Path(__file__).parent / "data" / "custom_rotation.json"
_REDIS_KEY = "wnba_custom_rotation"


def _get_redis():
    """Return an Upstash Redis client if credentials are configured, else None."""
    try:
        from upstash_redis import Redis
        url = st.secrets.get("UPSTASH_REDIS_REST_URL")
        token = st.secrets.get("UPSTASH_REDIS_REST_TOKEN")
        if url and token:
            return Redis(url=url, token=token)
    except Exception:
        pass
    return None


def _save_rotation(redis=None) -> None:
    data = {
        k: float(v)
        for k, v in st.session_state.items()
        if isinstance(k, str) and (k.startswith("rot_") or k.startswith("o_lbr_") or k.startswith("d_lbr_") or k.startswith("ts_"))
        and isinstance(v, (int, float))
    }
    payload = json.dumps(data)
    if redis is not None:
        redis.set(_REDIS_KEY, payload)
    else:
        _ROTATION_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _ROTATION_SAVE_PATH.write_text(payload)


def _load_rotation(redis=None) -> bool:
    if redis is not None:
        payload = redis.get(_REDIS_KEY)
        if not payload:
            return False
        data = json.loads(payload) if isinstance(payload, str) else payload
    else:
        if not _ROTATION_SAVE_PATH.exists():
            return False
        data = json.loads(_ROTATION_SAVE_PATH.read_text())
    for k, v in data.items():
        st.session_state[k] = float(v)
    return True


def _compute_effective_team_war(
    player_records: list[dict],
    rosters_raw: dict,
    baseline: dict,
) -> dict:
    """
    Build team WAR using saved rotation MPG values where available.
    Player list comes from API rosters; WAR/MPG rates come from LEBRON data.
    Falls back to baseline WAR for teams with no LEBRON coverage.
    """
    if not player_records:
        return baseline

    o_lbr_lookup = {_norm(r["player"]): r.get("o_lebron", 0.0) for r in player_records}
    d_lbr_lookup = {_norm(r["player"]): r.get("d_lebron", 0.0) for r in player_records}
    mpg_lookup   = {_norm(r["player"]): r["mpg"] for r in player_records}

    all_teams = set(baseline.keys()) | set(rosters_raw.keys())
    effective = {}
    for team in all_teams:
        players = [p["player"] for p in rosters_raw.get(team, []) if "player" in p]
        if not players:
            effective[team] = baseline.get(team, 0.0)
            continue

        has_lebron = any(
            (o_lbr_lookup.get(_norm(p), 0.0) + d_lbr_lookup.get(_norm(p), 0.0)) != 0
            for p in players
        )
        if not has_lebron:
            effective[team] = baseline.get(team, 0.0)
            continue

        lbr_total = sum(
            (st.session_state.get(f"o_lbr_{team}_{p}", o_lbr_lookup.get(_norm(p), 0.0))
             + st.session_state.get(f"d_lbr_{team}_{p}", d_lbr_lookup.get(_norm(p), 0.0)))
            * st.session_state.get(f"rot_{team}_{p}", mpg_lookup.get(_norm(p), 0.0))
            / 40
            for p in players
        )
        effective[team] = lbr_total

    return effective


# Scale: 1 unit of team O/D-LEBRON above league avg ≈ this many ORtg/DRtg points
_ORTG_SCALE = 1.5
_DRTG_SCALE = 1.5


def _compute_effective_team_od(
    player_records: list[dict],
    rosters_raw: dict,
) -> tuple[dict, dict]:
    """
    Separate effective O-LEBRON and D-LEBRON per team (weighted by custom MPG).
    Returns (team_o_dict, team_d_dict).
    """
    o_lbr_lookup = {_norm(r["player"]): r.get("o_lebron", 0.0) for r in player_records}
    d_lbr_lookup = {_norm(r["player"]): r.get("d_lebron", 0.0) for r in player_records}
    mpg_lookup   = {_norm(r["player"]): r["mpg"] for r in player_records}

    team_o, team_d = {}, {}
    for team, players_list in rosters_raw.items():
        players = [p["player"] for p in players_list if "player" in p]
        team_o[team] = sum(
            st.session_state.get(f"o_lbr_{team}_{p}", o_lbr_lookup.get(_norm(p), 0.0))
            * st.session_state.get(f"rot_{team}_{p}", mpg_lookup.get(_norm(p), 0.0))
            / 40
            for p in players
        )
        team_d[team] = sum(
            st.session_state.get(f"d_lbr_{team}_{p}", d_lbr_lookup.get(_norm(p), 0.0))
            * st.session_state.get(f"rot_{team}_{p}", mpg_lookup.get(_norm(p), 0.0))
            / 40
            for p in players
        )
    return team_o, team_d


def _build_team_ratings(team_o: dict, team_d: dict) -> dict[str, tuple[float, float]]:
    """
    Convert team O/D-LEBRON dicts to {team: (ortg, drtg)}.
    Teams are shifted relative to the league average so the mean team projects to LEAGUE_AVG_ORTG.
    Higher D-LEBRON → lower (better) DRtg.
    """
    if not team_o:
        return {}
    teams = set(team_o) | set(team_d)
    avg_o = sum(team_o.get(t, 0.0) for t in teams) / len(teams)
    avg_d = sum(team_d.get(t, 0.0) for t in teams) / len(teams)
    return {
        t: (
            LEAGUE_AVG_ORTG + (team_o.get(t, avg_o) - avg_o) * _ORTG_SCALE,
            LEAGUE_AVG_ORTG - (team_d.get(t, avg_d) - avg_d) * _DRTG_SCALE,
        )
        for t in teams
    }


# ── Load Data ────────────────────────────────────────────────────────────────

schedule_df = load_schedule()
team_lebron = load_team_lebron()
player_records = load_player_lebron_data()
rosters_raw = load_rosters()

# Strip preseason — regular season starts May 7 2026
REGULAR_SEASON_START = "2026-05-07"
if not schedule_df.empty:
    schedule_df = schedule_df[schedule_df["date"] >= REGULAR_SEASON_START].reset_index(drop=True)

if schedule_df.empty:
    st.warning(
        "⚠️ Schedule data is not yet available for the 2026 WNBA season. "
        "The season typically starts in May. Check back soon!"
    )
    st.stop()

# Auto-load saved rotation once per browser session (must run before simulation)
if "rotation_file_loaded" not in st.session_state:
    _redis = _get_redis()
    _load_rotation(_redis)
    st.session_state["rotation_file_loaded"] = True

effective_team_war = _compute_effective_team_war(player_records, rosters_raw, team_lebron)

_team_o, _team_d = _compute_effective_team_od(player_records, rosters_raw)
team_ratings = _build_team_ratings(_team_o, _team_d)

sim_results = load_simulation(
    schedule_df.to_json(orient="records"),
    json.dumps(effective_team_war, sort_keys=True),
)

# ── Header ───────────────────────────────────────────────────────────────────

st.title("🏀 WNBA 2026 Season Dashboard")
col_meta1, col_meta2, col_meta3 = st.columns(3)
completed = (schedule_df["status"] == "Final").sum()
total_games = len(schedule_df)
remaining = total_games - completed
col_meta1.metric("Games Played", completed)
col_meta2.metric("Games Remaining", remaining)
col_meta3.metric("Simulations", "1,000")

st.divider()

tab_season, tab_rosters, tab_rotation, tab_players = st.tabs(["Season Projections", "Rosters", "Rotation Builder", "Player Rankings"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SEASON PROJECTIONS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_season:

    # ── Section 1: Next 10 Games ──────────────────────────────────────────────
    st.markdown('<div class="section-header">📅 Upcoming Games</div>', unsafe_allow_html=True)

    from datetime import date as _date
    upcoming_games = schedule_df[
        (schedule_df["status"] != "Final") &
        (pd.to_datetime(schedule_df["date"]).dt.date >= _date.today())
    ].head(10)

    if upcoming_games.empty:
        st.info("No upcoming games found.")
    else:
        game_list = list(upcoming_games.iterrows())
        for row_start in range(0, len(game_list), 3):
            row_games = game_list[row_start:row_start + 3]
            cols = st.columns(len(row_games))
            for col, (_, game) in zip(cols, row_games):
                home = game["home_team"]
                away = game["away_team"]
                home_lbr = effective_team_war.get(home, 0.0)
                away_lbr = effective_team_war.get(away, 0.0)
                home_wp = win_probability(home_lbr, away_lbr)
                away_wp = 1.0 - home_wp
                spread = win_prob_to_spread(home_wp)
                home_favored = home_wp >= 0.5

                home_ortg, home_drtg = team_ratings.get(home, (LEAGUE_AVG_ORTG, LEAGUE_AVG_ORTG))
                away_ortg, away_drtg = team_ratings.get(away, (LEAGUE_AVG_ORTG, LEAGUE_AVG_ORTG))
                proj_total = project_game_total(home_ortg, home_drtg, away_ortg, away_drtg)

                with col:
                    with st.container(border=True):
                        lc, mc, rc = st.columns([2, 1, 2])
                        with lc:
                            img = logo_image(away, 50)
                            if img:
                                st.image(img, width=50)
                            away_name = TEAM_NAMES.get(away, away)
                            st.markdown(f"**{away_name}**" if not home_favored else away_name)
                            st.markdown(f"{'🟢' if not home_favored else ''} **{away_wp:.0%}**")
                        with mc:
                            st.markdown("<br><div style='text-align:center;font-size:1.2rem'>@</div>",
                                        unsafe_allow_html=True)
                        with rc:
                            img = logo_image(home, 50)
                            if img:
                                st.image(img, width=50)
                            home_name = TEAM_NAMES.get(home, home)
                            st.markdown(f"**{home_name}**" if home_favored else home_name)
                            st.markdown(f"{'🟢' if home_favored else ''} **{home_wp:.0%}**")

                        if abs(spread) < 0.5:
                            st.caption(f"Pick 'em  |  O/U: {proj_total:.1f}")
                        elif home_favored:
                            st.caption(f"Spread: {home_name} **{spread:.1f}** / {away_name} +{abs(spread):.1f}  |  O/U: {proj_total:.1f}")
                        else:
                            st.caption(f"Spread: {away_name} **{-abs(spread):.1f}** / {home_name} +{abs(spread):.1f}  |  O/U: {proj_total:.1f}")

    st.divider()

    # ── Section 2: Projected Season Wins ────────────────────────────────────
    st.markdown('<div class="section-header">📊 Projected Season Wins</div>', unsafe_allow_html=True)

    proj = sim_results.projected_wins

    def logo_b64(team: str, size: int = 28) -> str | None:
        path = get_team_logo(team)
        if not path or not path.exists():
            return None
        img = Image.open(path).convert("RGBA").resize((size, size), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    row_height = 44
    n_teams = len(proj)

    fig_wins = go.Figure()
    fig_wins.add_trace(go.Bar(
        y=proj["team"],
        x=proj["median_wins"],
        orientation="h",
        error_x=dict(
            type="data",
            array=proj["p90_wins"] - proj["median_wins"],
            arrayminus=proj["median_wins"] - proj["p10_wins"],
            visible=True,
            color="#aaa",
        ),
        marker_color="#ff6b35",
        text=proj["median_wins"].apply(lambda x: f"{x:.1f}"),
        textposition="outside",
    ))

    for i, (_, row) in enumerate(proj.iterrows()):
        src = logo_b64(row["team"], size=24)
        if src:
            y_paper = 1.0 - (i + 0.5) / n_teams
            fig_wins.add_layout_image(dict(
                source=src,
                xref="paper", yref="paper",
                x=0, y=y_paper,
                sizex=0.04, sizey=0.04 * (700 / max(300, n_teams * row_height)),
                xanchor="right", yanchor="middle",
                layer="above",
            ))

    fig_wins.update_layout(
        xaxis_title="Projected Wins",
        yaxis=dict(autorange="reversed", showticklabels=False),
        height=max(300, n_teams * row_height),
        margin=dict(l=35, r=80, t=20, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_wins, width="stretch")

    st.divider()

    # ── Section 3: Playoff Chances ───────────────────────────────────────────
    st.markdown('<div class="section-header">🏆 Playoff Chances (Top 8)</div>', unsafe_allow_html=True)

    playoff = sim_results.playoff_odds

    def playoff_color(pct):
        if pct >= 80:
            return "#4ade80"
        elif pct >= 50:
            return "#facc15"
        elif pct >= 20:
            return "#fb923c"
        else:
            return "#f87171"

    colors = [playoff_color(p) for p in playoff["playoff_pct"]]

    fig_playoff = go.Figure(go.Bar(
        x=playoff["team"],
        y=playoff["playoff_pct"],
        marker_color=colors,
        text=playoff["playoff_pct"].apply(lambda x: f"{x:.0f}%"),
        textposition="outside",
    ))
    fig_playoff.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)
    fig_playoff.update_layout(
        yaxis=dict(title="Playoff Probability (%)", range=[0, 110]),
        height=350,
        margin=dict(l=40, r=40, t=20, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_playoff, width="stretch")

    st.divider()

    # ── Section 4: Seed Probabilities ────────────────────────────────────────
    st.markdown('<div class="section-header">🔢 Seed Probability Table</div>', unsafe_allow_html=True)

    seed_df = sim_results.seed_probabilities
    seed_cols = [c for c in seed_df.columns if c != "team"]
    display_df = seed_df.set_index("team")[seed_cols]

    fig_heat = go.Figure(data=go.Heatmap(
        z=display_df.values,
        x=seed_cols,
        y=display_df.index.tolist(),
        colorscale=[
            [0.0, "#1e1e2e"],
            [0.3, "#1d4ed8"],
            [0.6, "#f59e0b"],
            [1.0, "#22c55e"],
        ],
        text=display_df.map(lambda x: f"{x:.0f}%").values,
        texttemplate="%{text}",
        showscale=False,
        hoverongaps=False,
    ))
    fig_heat.update_layout(
        height=max(300, len(display_df) * 40),
        margin=dict(l=80, r=40, t=20, b=40),
        xaxis=dict(side="top"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_heat, width="stretch")

    st.caption(
        "Win probabilities derived from team LEBRON totals (Basketball Index) "
        "via logistic model. Season simulated 1,000 times. "
        "LEBRON = Luck-adjusted player Estimate using Box prior Regularized ON-OFF."
    )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ROSTERS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_rosters:
    st.markdown('<div class="section-header">🏀 Team Rosters</div>', unsafe_allow_html=True)

    rosters_raw = load_rosters()

    if not rosters_raw:
        st.info("Roster data is not yet available.")
    else:
        sorted_teams = sorted(rosters_raw.keys(), key=lambda t: TEAM_NAMES.get(t, t))
        team_options = {TEAM_NAMES.get(t, t): t for t in sorted_teams}

        selected_name = st.selectbox(
            "Select a team",
            options=list(team_options.keys()),
            label_visibility="collapsed",
        )
        selected_abbr = team_options[selected_name]

        logo_col, name_col = st.columns([1, 8])
        with logo_col:
            logo = logo_image(selected_abbr, size=60)
            if logo:
                st.image(logo, width=60)
        with name_col:
            st.subheader(selected_name)

        roster_df = pd.DataFrame(rosters_raw[selected_abbr])

        if roster_df.empty:
            st.info("No roster data available for this team yet.")
        else:
            col_labels = {
                "player":      "Player",
                "num":         "#",
                "position":    "Pos",
                "height":      "Height",
                "weight":      "Weight",
                "age":         "Age",
                "experience":  "Exp",
                "school":      "School/Country",
                "how_acquired": "How Acquired",
            }
            display_cols = [c for c in col_labels if c in roster_df.columns]
            display = roster_df[display_cols].rename(columns=col_labels)

            if "Age" in display.columns:
                display["Age"] = pd.to_numeric(display["Age"], errors="coerce").astype("Int64")
            if "Exp" in display.columns:
                display["Exp"] = display["Exp"].replace("R", "Rookie")

            st.dataframe(
                display,
                width="stretch",
                hide_index=True,
                height=min(600, 36 + len(display) * 35),
            )
            st.caption(f"{len(roster_df)} players on roster")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ROTATION BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

with tab_rotation:
    st.markdown('<div class="section-header">⚙️ Rotation Builder</div>', unsafe_allow_html=True)
    st.caption(
        "Set minutes per game (MPG) for each player (max 40). "
        "Team WAR = Σ (WAR/40 × Custom MPG ÷ 40) across all players."
    )

    btn_col1, btn_col2, btn_col3, btn_col4, _ = st.columns([1, 1, 1, 1, 3])
    with btn_col1:
        if st.button("💾 Save", width="stretch", help="Save rotation"):
            _save_rotation(_get_redis())
            st.success("Rotation saved.")
    with btn_col2:
        if st.button("📂 Load", width="stretch", help="Reload saved rotation"):
            if _load_rotation(_get_redis()):
                # Clear frozen editor data + widget state so they re-seed from loaded values
                for _k in [f"editor_frozen_{selected_rot_abbr}", f"rotation_editor_{selected_rot_abbr}"]:
                    st.session_state.pop(_k, None)
                st.rerun()
            else:
                st.info("No saved rotation found.")
    with btn_col3:
        if st.button("🔄 Refresh", width="stretch", help="Re-fetch rosters from API"):
            load_rosters.clear()
            st.rerun()
    with btn_col4:
        import_clicked = st.button("📋 Sheets", width="stretch", help="Import MPG from Google Sheets")

    if not rosters_raw:
        st.info("Roster data not available.")
    else:
        # ── LEBRON lookup by player name ──────────────────────────────────────
        o_lbr_lookup = {_norm(r["player"]): r.get("o_lebron", 0.0) for r in player_records}
        d_lbr_lookup = {_norm(r["player"]): r.get("d_lebron", 0.0) for r in player_records}
        mpg_lookup   = {_norm(r["player"]): r["mpg"] for r in player_records}

        # ── Team selector ────────────────────────────────────────────────────
        # Union of roster data and all known teams so expansion teams always appear
        all_known = set(rosters_raw.keys()) | set(TEAM_NAMES.keys())
        rot_teams = sorted(all_known, key=lambda t: TEAM_NAMES.get(t, t))
        rot_team_options = {TEAM_NAMES.get(t, t): t for t in rot_teams}

        selected_rot_name = st.selectbox(
            "Select team",
            options=list(rot_team_options.keys()),
            label_visibility="collapsed",
            key="rotation_team_select",
        )
        selected_rot_abbr = rot_team_options[selected_rot_name]

        # ── Team header ──────────────────────────────────────────────────────
        logo_col, name_col = st.columns([1, 8])
        with logo_col:
            logo = logo_image(selected_rot_abbr, size=60)
            if logo:
                st.image(logo, width=60)
        with name_col:
            st.subheader(selected_rot_name)

        # Handle Sheets import (must know selected team + roster first)
        roster_players = rosters_raw.get(selected_rot_abbr, [])
        if import_clicked:
            try:
                n, unmatched = _import_minutes_from_sheets(selected_rot_abbr, roster_players)
                if n:
                    msg = f"Imported minutes for {n} player(s)."
                    if unmatched:
                        msg += f" Unmatched: {', '.join(unmatched)}."
                    for _k in [f"editor_frozen_{selected_rot_abbr}", f"rotation_editor_{selected_rot_abbr}"]:
                        st.session_state.pop(_k, None)
                    st.rerun()
                else:
                    st.warning(f"No players matched. Unmatched: {', '.join(unmatched) or 'none'}.")
            except Exception as e:
                st.error(f"Import failed: {e}")

        # Build editor rows from API roster, joined to LEBRON data by name
        editor_rows = sorted(
            [
                {
                    "Player":      p["player"],
                    "Pos":         p.get("position", ""),
                    "Actual MPG":  round(mpg_lookup.get(_norm(p["player"]), 0.0), 1),
                    "Custom MPG":  float(st.session_state.get(
                        f"rot_{selected_rot_abbr}_{p['player']}",
                        round(mpg_lookup.get(_norm(p["player"]), 0.0), 1),
                    )),
                    "Act O-LBR":  round(o_lbr_lookup.get(_norm(p["player"]), 0.0), 2),
                    "Cust O-LBR": float(st.session_state.get(
                        f"o_lbr_{selected_rot_abbr}_{p['player']}",
                        round(o_lbr_lookup.get(_norm(p["player"]), 0.0), 2),
                    )),
                    "Act D-LBR":  round(d_lbr_lookup.get(_norm(p["player"]), 0.0), 2),
                    "Cust D-LBR": float(st.session_state.get(
                        f"d_lbr_{selected_rot_abbr}_{p['player']}",
                        round(d_lbr_lookup.get(_norm(p["player"]), 0.0), 2),
                    )),
                    "Last Edited": (
                        datetime.fromtimestamp(ts).strftime("%b %d %H:%M")
                        if (ts := st.session_state.get(f"ts_{selected_rot_abbr}_{p['player']}"))
                        else ""
                    ),
                }
                for p in roster_players if "player" in p
            ],
            key=lambda x: -x["Custom MPG"],
        )

        if not editor_rows:
            st.info("No roster data available for this team.")
        else:
            # Freeze editor_df so the data argument never changes mid-session.
            # Changing data causes st.data_editor to reset its edit state, causing reversions.
            # Re-freeze only on explicit actions (Load, Sheets import, team switch).
            _frozen_key = f"editor_frozen_{selected_rot_abbr}"
            if _frozen_key not in st.session_state:
                st.session_state[_frozen_key] = pd.DataFrame(editor_rows)
            editor_df = st.session_state[_frozen_key]

            edited = st.data_editor(
                editor_df,
                key=f"rotation_editor_{selected_rot_abbr}",
                column_config={
                    "Player":      st.column_config.TextColumn("Player", disabled=True),
                    "Pos":         st.column_config.TextColumn("Pos", disabled=True),
                    "Actual MPG":  st.column_config.NumberColumn("Actual MPG", disabled=True, format="%.1f"),
                    "Custom MPG":  st.column_config.NumberColumn(
                        "Custom MPG", min_value=0.0, max_value=40.0, step=0.5, format="%.1f"
                    ),
                    "Act O-LBR":  st.column_config.NumberColumn("Act O-LBR", disabled=True, format="%.2f"),
                    "Cust O-LBR": st.column_config.NumberColumn(
                        "Cust O-LBR", min_value=-5.0, max_value=15.0, step=0.1, format="%.2f",
                        help="Offensive LEBRON (seasonal total). Override if needed."
                    ),
                    "Act D-LBR":   st.column_config.NumberColumn("Act D-LBR", disabled=True, format="%.2f"),
                    "Cust D-LBR":  st.column_config.NumberColumn(
                        "Cust D-LBR", min_value=-5.0, max_value=15.0, step=0.1, format="%.2f",
                        help="Defensive LEBRON (seasonal total). Override if needed."
                    ),
                    "Last Edited": st.column_config.TextColumn("Last Edited", disabled=True),
                },
                hide_index=True,
                width="stretch",
                num_rows="fixed",
            )

            for _, row in edited.iterrows():
                p = row["Player"]
                t = selected_rot_abbr
                new_mpg = round(float(row["Custom MPG"]), 1)
                new_o   = round(float(row["Cust O-LBR"]), 2)
                new_d   = round(float(row["Cust D-LBR"]), 2)
                if (new_mpg != st.session_state.get(f"rot_{t}_{p}")
                        or new_o != st.session_state.get(f"o_lbr_{t}_{p}")
                        or new_d != st.session_state.get(f"d_lbr_{t}_{p}")):
                    st.session_state[f"ts_{t}_{p}"] = datetime.now().timestamp()
                st.session_state[f"rot_{t}_{p}"]   = new_mpg
                st.session_state[f"o_lbr_{t}_{p}"] = new_o
                st.session_state[f"d_lbr_{t}_{p}"] = new_d

            # ── Team LEBRON metrics ──────────────────────────────────────────
            edited["Cust LEBRON"] = edited["Cust O-LBR"] + edited["Cust D-LBR"]
            edited["Proj LEBRON"] = edited["Cust LEBRON"] * edited["Custom MPG"] / 40
            edited["Proj O-LBR"]  = edited["Cust O-LBR"] * edited["Custom MPG"] / 40
            edited["Proj D-LBR"]  = edited["Cust D-LBR"] * edited["Custom MPG"] / 40
            actual_lebron   = ((editor_df["Act O-LBR"] + editor_df["Act D-LBR"]) * editor_df["Actual MPG"] / 40).sum()
            actual_o_lebron = (editor_df["Act O-LBR"] * editor_df["Actual MPG"] / 40).sum()
            actual_d_lebron = (editor_df["Act D-LBR"] * editor_df["Actual MPG"] / 40).sum()
            custom_lebron   = edited["Proj LEBRON"].sum()
            custom_o_lebron = edited["Proj O-LBR"].sum()
            custom_d_lebron = edited["Proj D-LBR"].sum()

            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric(
                "Proj O-LEBRON",
                f"{custom_o_lebron:.2f}",
                delta=f"{custom_o_lebron - actual_o_lebron:+.2f} vs baseline",
            )
            mc2.metric(
                "Proj D-LEBRON",
                f"{custom_d_lebron:.2f}",
                delta=f"{custom_d_lebron - actual_d_lebron:+.2f} vs baseline",
            )
            mc3.metric(
                "Proj Team LEBRON",
                f"{custom_lebron:.2f}",
                delta=f"{custom_lebron - actual_lebron:+.2f} vs baseline",
            )
            mc4.metric(
                "Total Custom MPG",
                f"{edited['Custom MPG'].round(1).sum():.1f}",
                delta=f"{edited['Custom MPG'].round(1).sum() - editor_df['Actual MPG'].sum():+.1f} vs actual",
            )
            mc5.metric("Players in Rotation", len(edited[edited["Custom MPG"] > 0]))
            st.caption("Proj LEBRON = Σ ((Cust O-LBR + Cust D-LBR) × Custom MPG ÷ 40). Changes flow into simulations on Save.")

        st.divider()

        # ── League-wide WAR comparison ───────────────────────────────────────
        st.markdown('<div class="section-header">📊 League WAR Comparison</div>', unsafe_allow_html=True)

        # Recompute effective WAR live (reflects any edits made this render)
        live_effective = _compute_effective_team_war(player_records, rosters_raw, team_lebron)

        teams_with_custom: set[str] = set()
        for team in live_effective:
            baseline_val = team_lebron.get(team, 0.0)
            if abs(live_effective[team] - baseline_val) > 0.01:
                teams_with_custom.add(team)

        league_df = pd.DataFrame(
            [{"team": t, "war": w} for t, w in sorted(live_effective.items(), key=lambda x: -x[1])]
        )

        bar_colors = [
            "#ff6b35" if row["team"] in teams_with_custom else "#64748b"
            for _, row in league_df.iterrows()
        ]

        fig_league = go.Figure(go.Bar(
            y=league_df["team"],
            x=league_df["war"],
            orientation="h",
            marker_color=bar_colors,
            text=league_df["war"].apply(lambda x: f"{x:.1f}"),
            textposition="outside",
        ))
        fig_league.update_layout(
            xaxis_title="Team WAR",
            yaxis=dict(autorange="reversed"),
            height=max(300, len(league_df) * 44),
            margin=dict(l=60, r=80, t=20, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_league, width="stretch")

        st.caption(
            "Orange bars = teams with custom rotations set. "
            "Gray bars = baseline (actual MPG from LEBRON data). "
            "Team LEBRON = Σ ((Cust O-LBR + Cust D-LBR) × Custom MPG ÷ 40) per player."
        )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PLAYER RANKINGS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_players:
    st.markdown('<div class="section-header">🏅 Player Rankings</div>', unsafe_allow_html=True)
    st.caption("Sorted by WAR/40. Ratings from LEBRON (Basketball Index).")

    if not player_records:
        st.info("Player rating data is not yet available.")
    else:
        # Build position, team, and display-name lookups from rosters (authoritative source)
        pos_lookup: dict[str, str] = {}
        roster_team_lookup: dict[str, str] = {}
        roster_name_lookup: dict[str, str] = {}  # normalized key -> clean roster name for display
        for _team, _players in rosters_raw.items():
            for _p in _players:
                if "player" in _p:
                    _key = _norm(_p["player"])
                    pos_lookup[_key] = _p.get("position", "")
                    roster_team_lookup[_key] = _team
                    roster_name_lookup[_key] = _p["player"]

        ranking_rows = []
        for r in player_records:
            name = r["player"]
            key_name = _norm(name)  # normalize to match Rotation Builder session state keys
            display_name = roster_name_lookup.get(key_name, name)  # use clean roster name
            team = roster_team_lookup.get(key_name, r.get("team", ""))
            act_o = round(r.get("o_lebron", 0.0), 2)
            act_d = round(r.get("d_lebron", 0.0), 2)
            cust_o = round(st.session_state.get(f"o_lbr_{team}_{key_name}", act_o), 2)
            cust_d = round(st.session_state.get(f"d_lbr_{team}_{key_name}", act_d), 2)
            act_mpg = round(r["mpg"], 1)
            cust_mpg = round(st.session_state.get(f"rot_{team}_{key_name}", act_mpg), 1)
            ranking_rows.append({
                "Player":      display_name,
                "Team":        team,
                "Pos":         pos_lookup.get(key_name, ""),
                "Minutes":     int(r.get("minutes", 0)),
                "MPG":         act_mpg,
                "Cust MPG":    cust_mpg,
                "O-LEBRON":    act_o,
                "D-LEBRON":    act_d,
                "LEBRON":      round(r.get("lebron", 0.0), 2),
                "Cust O-LBR":  cust_o,
                "Cust D-LBR":  cust_d,
                "Cust LEBRON": round(cust_o + cust_d, 2),
                "Last Edited": (
                    datetime.fromtimestamp(ts).strftime("%b %d %H:%M")
                    if (ts := st.session_state.get(f"ts_{team}_{key_name}"))
                    else ""
                ),
            })

        rankings_df = (
            pd.DataFrame(ranking_rows)
            .sort_values("Cust LEBRON", ascending=False)
            .reset_index(drop=True)
        )
        rankings_df.index += 1  # 1-based rank

        # Team filter
        team_filter_options = ["All Teams"] + sorted(
            rankings_df["Team"].unique().tolist(),
            key=lambda t: TEAM_NAMES.get(t, t),
        )
        selected_filter = st.selectbox(
            "Filter by team",
            options=team_filter_options,
            label_visibility="collapsed",
            key="player_rankings_team_filter",
        )
        if selected_filter != "All Teams":
            display_rankings = rankings_df[rankings_df["Team"] == selected_filter]
        else:
            display_rankings = rankings_df

        st.dataframe(
            display_rankings,
            width="stretch",
            height=min(700, 36 + len(display_rankings) * 35),
            column_config={
                "MPG":         st.column_config.NumberColumn("MPG", format="%.1f"),
                "Cust MPG":    st.column_config.NumberColumn("Cust MPG", format="%.1f"),
                "O-LEBRON":    st.column_config.NumberColumn("O-LEBRON", format="%.2f"),
                "D-LEBRON":    st.column_config.NumberColumn("D-LEBRON", format="%.2f"),
                "LEBRON":      st.column_config.NumberColumn("LEBRON", format="%.2f"),
                "Cust O-LBR":  st.column_config.NumberColumn("Cust O-LBR", format="%.2f"),
                "Cust D-LBR":  st.column_config.NumberColumn("Cust D-LBR", format="%.2f"),
                "Cust LEBRON": st.column_config.NumberColumn("Cust LEBRON", format="%.2f"),
                "Last Edited": st.column_config.TextColumn("Last Edited"),
            },
        )
