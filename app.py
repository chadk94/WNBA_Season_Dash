"""
app.py — WNBA Season Dashboard (Streamlit)

Run with: streamlit run app.py
"""

import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from fetch_data import (
    TEAM_NAMES,
    get_schedule,
    get_team_lebron,
    get_team_rosters,
    get_player_lebron,
    get_team_logo,
    prefetch_all_logos,
)
from model import (
    SimResults,
    get_next_gamedays,
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


@st.cache_data(ttl=3600, show_spinner="Loading LEBRON ratings...")
def load_team_lebron():
    player_df = get_player_lebron()
    rosters = get_team_rosters(SEASON)
    return get_team_lebron(player_df, rosters)


@st.cache_data(ttl=3600, show_spinner="Loading rosters...")
def load_rosters() -> dict:
    rosters = get_team_rosters(SEASON)
    # Convert DataFrames to JSON-serialisable dicts for Streamlit cache
    return {abbr: df.to_dict(orient="records") for abbr, df in rosters.items()}


@st.cache_data(ttl=3600, show_spinner="Loading player ratings...")
def load_player_lebron_data() -> list[dict]:
    df = get_player_lebron()
    if df.empty or "player" not in df.columns:
        return []
    df = df.copy()
    df["minutes"] = pd.to_numeric(df.get("minutes", 0), errors="coerce").fillna(0)
    df["lebron_war"] = pd.to_numeric(df.get("lebron_war", 0), errors="coerce").fillna(0)
    df["war_per_min"] = df.apply(
        lambda r: r["lebron_war"] / r["minutes"] if r["minutes"] > 0 else 0.0, axis=1
    )
    # Use MPG from data if present; otherwise derive from total minutes (fallback only)
    if "mpg" not in df.columns:
        df["mpg"] = df["minutes"] / 40  # rough fallback if CSV lacks MPG column
    df["mpg"] = pd.to_numeric(df["mpg"], errors="coerce").fillna(0).round(1)
    # WAR per MPG: scaling MPG up/down scales WAR proportionally
    df["war_per_mpg"] = df.apply(
        lambda r: r["lebron_war"] / r["mpg"] if r["mpg"] > 0 else 0.0, axis=1
    ).round(4)
    return df[["player", "team", "minutes", "mpg", "lebron_war", "war_per_min", "war_per_mpg"]].to_dict(orient="records")


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


def _save_rotation_to_disk() -> None:
    data = {
        k: float(v)
        for k, v in st.session_state.items()
        if isinstance(k, str) and (k.startswith("rot_") or k.startswith("war_"))
        and isinstance(v, (int, float))
    }
    _ROTATION_SAVE_PATH.write_text(json.dumps(data, indent=2))


def _load_rotation_from_disk() -> bool:
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

    war_lookup = {r["player"]: r["war_per_mpg"] for r in player_records}
    mpg_lookup = {r["player"]: r["mpg"] for r in player_records}

    all_teams = set(baseline.keys()) | set(rosters_raw.keys())
    effective = {}
    for team in all_teams:
        players = [p["player"] for p in rosters_raw.get(team, []) if "player" in p]
        if not players:
            effective[team] = baseline.get(team, 0.0)
            continue

        has_lebron = any(war_lookup.get(p, 0.0) > 0 for p in players)
        if not has_lebron:
            effective[team] = baseline.get(team, 0.0)
            continue

        war_total = sum(
            st.session_state.get(f"war_{team}_{p}", war_lookup.get(p, 0.0))
            * st.session_state.get(f"rot_{team}_{p}", mpg_lookup.get(p, 0.0))
            for p in players
        )
        effective[team] = war_total

    return effective


# ── Load Data ────────────────────────────────────────────────────────────────

schedule_df = load_schedule()
team_lebron = load_team_lebron()
player_records = load_player_lebron_data()
rosters_raw = load_rosters()

if schedule_df.empty:
    st.warning(
        "⚠️ Schedule data is not yet available for the 2026 WNBA season. "
        "The season typically starts in May. Check back soon!"
    )
    st.stop()

# Auto-load saved rotation once per browser session (must run before simulation)
if "rotation_file_loaded" not in st.session_state:
    _load_rotation_from_disk()
    st.session_state["rotation_file_loaded"] = True

effective_team_war = _compute_effective_team_war(player_records, rosters_raw, team_lebron)

sim_results = load_simulation(
    schedule_df.to_json(orient="records"),
    json.dumps(effective_team_war),
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

tab_season, tab_rosters, tab_rotation = st.tabs(["Season Projections", "Rosters", "Rotation Builder"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SEASON PROJECTIONS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_season:

    # ── Section 1: Next Two Gamedays ─────────────────────────────────────────
    st.markdown('<div class="section-header">📅 Next Two Gamedays</div>', unsafe_allow_html=True)

    next_dates = get_next_gamedays(schedule_df, n=2)

    if not next_dates:
        st.info("No upcoming games found.")
    else:
        for game_date in next_dates:
            st.subheader(pd.to_datetime(game_date).strftime("%A, %B %#d, %Y"))
            day_games = schedule_df[schedule_df["date"] == game_date]

            cols = st.columns(min(len(day_games), 3))
            for i, (_, game) in enumerate(day_games.iterrows()):
                col = cols[i % len(cols)]
                home = game["home_team"]
                away = game["away_team"]
                home_lbr = team_lebron.get(home, 0.0)
                away_lbr = team_lebron.get(away, 0.0)
                home_wp = win_probability(home_lbr, away_lbr)
                away_wp = 1.0 - home_wp
                spread = win_prob_to_spread(home_wp)
                home_favored = home_wp >= 0.5

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
                            st.caption("Pick 'em")
                        elif home_favored:
                            st.caption(f"Spread: {home_name} **{spread:.1f}** / {away_name} +{abs(spread):.1f}")
                        else:
                            st.caption(f"Spread: {away_name} **{-abs(spread):.1f}** / {home_name} +{abs(spread):.1f}")

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
        error_x=dict(type="data", array=proj["std_wins"], visible=True, color="#aaa"),
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
    st.plotly_chart(fig_wins, use_container_width=True)

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
    st.plotly_chart(fig_playoff, use_container_width=True)

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
    st.plotly_chart(fig_heat, use_container_width=True)

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
                use_container_width=True,
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
        "Team WAR = Σ (WAR/MPG × Custom MPG) across all players."
    )

    btn_col1, btn_col2, _ = st.columns([1, 1, 6])
    with btn_col1:
        if st.button("💾 Save", use_container_width=True, help="Save rotation to disk"):
            _save_rotation_to_disk()
            st.success("Rotation saved.")
    with btn_col2:
        if st.button("📂 Load", use_container_width=True, help="Reload rotation from disk"):
            if _load_rotation_from_disk():
                st.success("Rotation loaded.")
                st.rerun()
            else:
                st.info("No saved rotation found.")

    if not rosters_raw:
        st.info("Roster data not available.")
    else:
        # ── LEBRON lookup by player name ──────────────────────────────────────
        war_lookup = {r["player"]: r["war_per_mpg"] for r in player_records}
        mpg_lookup = {r["player"]: r["mpg"] for r in player_records}

        # ── Team selector ────────────────────────────────────────────────────
        rot_teams = sorted(rosters_raw.keys(), key=lambda t: TEAM_NAMES.get(t, t))
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

        # Build editor rows from API roster, joined to LEBRON data by name
        roster_players = rosters_raw.get(selected_rot_abbr, [])
        editor_rows = sorted(
            [
                {
                    "Player": p["player"],
                    "Pos": p.get("position", ""),
                    "Actual MPG": round(mpg_lookup.get(p["player"], 0.0), 1),
                    "Custom MPG": float(st.session_state.get(
                        f"rot_{selected_rot_abbr}_{p['player']}",
                        round(mpg_lookup.get(p["player"], 0.0), 1),
                    )),
                    "Actual WAR/MPG": round(war_lookup.get(p["player"], 0.0), 4),
                    "Custom WAR/MPG": float(st.session_state.get(
                        f"war_{selected_rot_abbr}_{p['player']}",
                        round(war_lookup.get(p["player"], 0.0), 4),
                    )),
                }
                for p in roster_players if "player" in p
            ],
            key=lambda x: -x["Actual WAR/MPG"],
        )

        if not editor_rows:
            st.info("No roster data available for this team.")
        else:
            editor_df = pd.DataFrame(editor_rows)

            edited = st.data_editor(
                editor_df,
                column_config={
                    "Player":           st.column_config.TextColumn("Player", disabled=True),
                    "Pos":              st.column_config.TextColumn("Pos", disabled=True),
                    "Actual MPG":       st.column_config.NumberColumn("Actual MPG", disabled=True, format="%.1f"),
                    "Custom MPG":       st.column_config.NumberColumn(
                        "Custom MPG", min_value=0.0, max_value=40.0, step=1.0, format="%.1f"
                    ),
                    "Actual WAR/MPG":   st.column_config.NumberColumn("Actual WAR/MPG", disabled=True, format="%.4f"),
                    "Custom WAR/MPG":   st.column_config.NumberColumn(
                        "Custom WAR/MPG", min_value=-1.0, max_value=1.0, step=0.001, format="%.4f",
                        help="Override WAR rate (WAR per minute-per-game). Defaults to LEBRON value."
                    ),
                },
                hide_index=True,
                use_container_width=True,
                key=f"editor_{selected_rot_abbr}",
                num_rows="fixed",
            )

            # Persist custom MPG and WAR/MPG in session state
            for _, row in edited.iterrows():
                st.session_state[f"rot_{selected_rot_abbr}_{row['Player']}"] = float(row["Custom MPG"])
                st.session_state[f"war_{selected_rot_abbr}_{row['Player']}"] = float(row["Custom WAR/MPG"])

            # ── WAR summary metrics ──────────────────────────────────────────
            edited["Proj WAR"] = edited["Custom MPG"] * edited["Custom WAR/MPG"]
            actual_war = (editor_df["Actual MPG"] * editor_df["Actual WAR/MPG"]).sum()
            custom_war = edited["Proj WAR"].sum()

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric(
                "Custom Team WAR",
                f"{custom_war:.2f}",
                delta=f"{custom_war - actual_war:+.2f} vs baseline",
            )
            mc2.metric(
                "Total Custom MPG",
                f"{edited['Custom MPG'].sum():.1f}",
                delta=f"{edited['Custom MPG'].sum() - editor_df['Actual MPG'].sum():+.1f} vs actual",
            )
            mc3.metric("Players in Rotation", len(edited[edited["Custom MPG"] > 0]))

            st.caption("Proj WAR = Custom MPG × Custom WAR/MPG. Changes flow into simulations on Save.")

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
        st.plotly_chart(fig_league, use_container_width=True)

        st.caption(
            "Orange bars = teams with custom rotations set. "
            "Gray bars = baseline (actual MPG from LEBRON data). "
            "Team WAR = Σ (WAR/MPG × Custom MPG) per player."
        )
