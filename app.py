"""
app.py — WNBA Season Dashboard (Streamlit)

Run with: streamlit run app.py
"""

import base64
import io
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
    return get_team_lebron(player_df)


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


# ── Load Data ────────────────────────────────────────────────────────────────

schedule_df = load_schedule()
team_lebron = load_team_lebron()

if schedule_df.empty:
    st.warning(
        "⚠️ Schedule data is not yet available for the 2026 WNBA season. "
        "The season typically starts in May. Check back soon!"
    )
    st.stop()

import json
sim_results = load_simulation(
    schedule_df.to_json(orient="records"),
    json.dumps(team_lebron),
)

# ── Header ───────────────────────────────────────────────────────────────────

st.title("🏀 WNBA 2026 Season Dashboard")
col_meta1, col_meta2, col_meta3 = st.columns(3)
completed = (schedule_df["status"] == "Final").sum() // 2
total_games = len(schedule_df) // 2
remaining = total_games - completed
col_meta1.metric("Games Played", completed)
col_meta2.metric("Games Remaining", remaining)
col_meta3.metric("Simulations", "1,000")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — NEXT TWO GAMEDAYS
# ═══════════════════════════════════════════════════════════════════════════════

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
            spread = win_prob_to_spread(home_wp)  # positive = home favored
            home_favored = home_wp >= 0.5

            with col:
                with st.container(border=True):
                    # Logos + teams
                    lc, mc, rc = st.columns([2, 1, 2])
                    with lc:
                        img = logo_image(away, 50)
                        if img:
                            st.image(img, width=50)
                        away_name = TEAM_NAMES.get(away, away)
                        if not home_favored:
                            st.markdown(f"**{away_name}**")
                        else:
                            st.markdown(f"{away_name}")
                        st.markdown(f"{'🟢' if not home_favored else ''} **{away_wp:.0%}**")
                    with mc:
                        st.markdown("<br><div style='text-align:center;font-size:1.2rem'>@</div>",
                                    unsafe_allow_html=True)
                    with rc:
                        img = logo_image(home, 50)
                        if img:
                            st.image(img, width=50)
                        home_name = TEAM_NAMES.get(home, home)
                        if home_favored:
                            st.markdown(f"**{home_name}**")
                        else:
                            st.markdown(f"{home_name}")
                        st.markdown(f"{'🟢' if home_favored else ''} **{home_wp:.0%}**")

                    # Spread: negative = favorite, positive = underdog (standard convention)
                    # spread < 0 means home favored; spread > 0 means away favored
                    if abs(spread) < 0.5:
                        st.caption("Pick 'em")
                    elif home_favored:
                        st.caption(f"Spread: {home_name} **{spread:.1f}** / {away_name} +{abs(spread):.1f}")
                    else:
                        st.caption(f"Spread: {away_name} **{-abs(spread):.1f}** / {home_name} +{abs(spread):.1f}")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PROJECTED SEASON WINS
# ═══════════════════════════════════════════════════════════════════════════════

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

# Logos replace y-axis tick labels: hide text, embed image per row
# y position in paper coords: rows are evenly spaced from top (reversed axis)
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

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PLAYOFF CHANCES
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">🏆 Playoff Chances (Top 8)</div>', unsafe_allow_html=True)

playoff = sim_results.playoff_odds

# Color bars by probability bracket
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

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — SEED PROBABILITIES
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">🔢 Seed Probability Table</div>', unsafe_allow_html=True)

seed_df = sim_results.seed_probabilities
seed_cols = [c for c in seed_df.columns if c != "team"]
display_df = seed_df.set_index("team")[seed_cols]

# Format as percentages
display_formatted = display_df.map(lambda x: f"{x:.0f}%")

# Heatmap using plotly
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

# ── Footer ───────────────────────────────────────────────────────────────────
st.caption(
    "Win probabilities derived from team LEBRON totals (Basketball Index) "
    "via logistic model. Season simulated 1,000 times. "
    "LEBRON = Luck-adjusted player Estimate using Box prior Regularized ON-OFF."
)
