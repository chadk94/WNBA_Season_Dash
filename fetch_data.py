"""
fetch_data.py — WNBA data acquisition
  - WNBA schedule via nba_api
  - Player LEBRON stats scraped from bball-index.com
  - Team logos from ESPN CDN
Results are cached in data/ for 24 hours.
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
LOGOS_DIR = DATA_DIR / "logos"
LOGOS_DIR.mkdir(exist_ok=True)

CACHE_TTL_HOURS = 24

# WNBA team abbreviations → ESPN logo slug mapping
TEAM_LOGO_SLUGS = {
    "ATL": "atl",
    "CHI": "chi",
    "CON": "con",
    "DAL": "dal",
    "GSV": "gsv",
    "IND": "ind",
    "LAS": "la",
    "LVA": "lv",
    "MIN": "min",
    "NYL": "ny",
    "PDX": "por",
    "PHX": "phx",
    "SEA": "sea",
    "TOR": "tor",
    "WAS": "was",
}

# Full team names for display
TEAM_NAMES = {
    "ATL": "Atlanta Dream",
    "CHI": "Chicago Sky",
    "CON": "Connecticut Sun",
    "DAL": "Dallas Wings",
    "GSV": "Golden State Valkyries",
    "IND": "Indiana Fever",
    "LAS": "Los Angeles Sparks",
    "LVA": "Las Vegas Aces",
    "MIN": "Minnesota Lynx",
    "NYL": "New York Liberty",
    "PDX": "Portland Fire",
    "PHX": "Phoenix Mercury",
    "SEA": "Seattle Storm",
    "TOR": "Toronto Tempo",
    "WAS": "Washington Mystics",
}


def _cache_path(name: str) -> Path:
    return DATA_DIR / f"{name}.json"


def _is_cache_valid(path: Path) -> bool:
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return datetime.now() - mtime < timedelta(hours=CACHE_TTL_HOURS)


def _load_cache(name: str):
    path = _cache_path(name)
    if _is_cache_valid(path):
        with open(path) as f:
            return json.load(f)
    return None


def _save_cache(name: str, data):
    with open(_cache_path(name), "w") as f:
        json.dump(data, f)


def get_schedule(season: int = 2026) -> pd.DataFrame:
    """
    Fetch the full WNBA schedule (including future games) via ScheduleLeagueV2.
    Returns DataFrame with columns:
      game_id, date, home_team, away_team, home_score, away_score, status
    """
    cache_key = f"schedule_{season}"
    cached = _load_cache(cache_key)
    if cached:
        return pd.DataFrame(cached)

    # WNBA season year format: 2025 season → "2025-26", 2026 → "2026-27"
    season_str = f"{season}-{str(season + 1)[-2:]}"
    known_teams = set(TEAM_LOGO_SLUGS.keys())

    try:
        from nba_api.stats.endpoints import scheduleleaguev2
        sched = scheduleleaguev2.ScheduleLeagueV2(league_id="10", season=season_str)
        games_df = sched.get_data_frames()[0]
    except Exception as e:
        print(f"[fetch_data] nba_api2 schedule fetch failed: {e}")
        return pd.DataFrame(columns=["game_id", "date", "home_team", "away_team",
                                     "home_score", "away_score", "status"])

    if games_df.empty:
        print("[fetch_data] No schedule data returned (season may not have started).")
        return pd.DataFrame(columns=["game_id", "date", "home_team", "away_team",
                                     "home_score", "away_score", "status"])

    rows = []
    for _, row in games_df.iterrows():
        home = str(row["homeTeam_teamTricode"]).upper()
        away = str(row["awayTeam_teamTricode"]).upper()
        # Filter to regular WNBA teams only (excludes preseason international opponents)
        if home not in known_teams or away not in known_teams:
            continue

        game_status = int(row["gameStatus"])  # 1=Scheduled, 2=InProgress, 3=Final
        home_score = row.get("homeTeam_score")
        away_score = row.get("awayTeam_score")

        if game_status == 3:
            status = "Final"
        elif game_status == 2:
            status = "InProgress"
        else:
            status = "Scheduled"

        rows.append({
            "game_id": str(row["gameId"]),
            "date": pd.to_datetime(row["gameDate"]).strftime("%Y-%m-%d"),
            "home_team": home,
            "away_team": away,
            "home_score": int(home_score) if game_status >= 2 and pd.notna(home_score) and home_score != "" and int(home_score) > 0 else None,
            "away_score": int(away_score) if game_status >= 2 and pd.notna(away_score) and away_score != "" and int(away_score) > 0 else None,
            "status": status,
        })

    df = pd.DataFrame(rows)
    _save_cache(cache_key, df.to_dict(orient="records"))
    print(f"[fetch_data] Schedule: {len(df)} games ({(df['status']=='Final').sum()} completed, "
          f"{(df['status']=='Scheduled').sum()} upcoming)")
    return df


def get_player_lebron() -> pd.DataFrame:
    """
    Scrape WNBA player LEBRON ratings from bball-index.com.
    Returns DataFrame with columns: player, team, lebron
    """
    cached = _load_cache("player_lebron")
    if cached:
        return pd.DataFrame(cached)

    url = "https://www.bball-index.com/wnba-player-ratings/"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"[fetch_data] LEBRON scrape failed: {e}")
        print("[fetch_data] Falling back to empty LEBRON data. Place a lebron_manual.csv in data/ to provide values.")
        return _load_manual_lebron()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Try to find a table or React/JSON data embedded in the page
    rows = []

    # Attempt 1: look for an HTML table
    table = soup.find("table")
    if table:
        headers_row = [th.get_text(strip=True) for th in table.find_all("th")]
        # Find relevant column indices
        try:
            name_idx = next(i for i, h in enumerate(headers_row) if "player" in h.lower() or "name" in h.lower())
            team_idx = next(i for i, h in enumerate(headers_row) if "team" in h.lower())
            lebron_idx = next(i for i, h in enumerate(headers_row) if "lebron" in h.lower())
        except StopIteration:
            print("[fetch_data] Could not find expected columns in LEBRON table.")
            return _load_manual_lebron()

        for tr in table.find_all("tr")[1:]:
            cells = tr.find_all("td")
            if len(cells) <= max(name_idx, team_idx, lebron_idx):
                continue
            try:
                rows.append({
                    "player": cells[name_idx].get_text(strip=True),
                    "team": cells[team_idx].get_text(strip=True).upper(),
                    "lebron": float(cells[lebron_idx].get_text(strip=True)),
                })
            except (ValueError, IndexError):
                continue

    # Attempt 2: look for embedded JSON (common in React-rendered pages)
    if not rows:
        import re
        json_match = re.search(r'window\.__INITIAL_STATE__\s*=\s*(\{.*?\});', resp.text, re.DOTALL)
        if not json_match:
            json_match = re.search(r'"players"\s*:\s*(\[.*?\])', resp.text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if isinstance(data, list):
                    for item in data:
                        if "lebron" in str(item).lower():
                            rows.append({
                                "player": item.get("player", item.get("name", "")),
                                "team": str(item.get("team", "")).upper(),
                                "lebron": float(item.get("lebron", item.get("LEBRON", 0))),
                            })
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

    if not rows:
        print("[fetch_data] Could not parse LEBRON data from page. Check bball-index.com structure.")
        return _load_manual_lebron()

    df = pd.DataFrame(rows)
    _save_cache("player_lebron", df.to_dict(orient="records"))
    return df


def _load_manual_lebron() -> pd.DataFrame:
    """Load from data/lebron_manual.csv if it exists (columns: team, lebron)."""
    manual_path = DATA_DIR / "lebron_manual.csv"
    if manual_path.exists():
        df = pd.read_csv(manual_path, encoding="utf-8-sig")
        # Normalize column names to lowercase
        df.columns = [c.strip().lower() for c in df.columns]
        # Map 'lebron war' or other variants — keep only what we need
        df.columns = [c.strip().lower() for c in df.columns]
        col_map = {}
        for c in df.columns:
            if c == "lebron":
                col_map[c] = "lebron"
            elif "lebron war" in c or c == "lebron war":
                col_map[c] = "lebron_war"
            elif c == "o-lebron":
                col_map[c] = "o_lebron"
            elif c == "d-lebron":
                col_map[c] = "d_lebron"
            elif "player" in c or "name" in c:
                col_map[c] = "player"
            elif c == "team":
                col_map[c] = "team"
            elif c == "minutes":
                col_map[c] = "minutes"
            elif c == "mpg":
                col_map[c] = "mpg"
        df = df.rename(columns=col_map)
        # Normalize legacy/alternate tricodes to current ones
        TRICODE_MAP = {
            "PHO": "PHX",
            "NY":  "NYL",
        }
        if "team" in df.columns:
            df["team"] = df["team"].str.upper().str.strip().replace(TRICODE_MAP)
            df = df[df["team"].notna() & ~df["team"].isin(["FA", "FREE AGENT", ""])]
        if "minutes" in df.columns:
            df["minutes"] = df["minutes"].astype(str).str.replace(",", "").astype(float)
        print(f"[fetch_data] Loaded manual LEBRON data: {len(df)} rows")
        keep = [c for c in ["player", "team", "lebron", "o_lebron", "d_lebron", "lebron_war", "minutes", "mpg"] if c in df.columns]
        return df[keep]
    # Return empty frame — model will handle missing teams gracefully
    return pd.DataFrame(columns=["player", "team", "lebron"])


def get_team_lebron(player_df: pd.DataFrame = None, rosters: dict = None) -> dict:
    """
    Aggregate player LEBRON WAR to team level (sum).
    LEBRON WAR is already minutes-adjusted, so summing gives total roster quality.
    Falls back to minutes-weighted average LEBRON if WAR is unavailable.

    If rosters (from get_team_rosters) are provided, team membership is determined
    by the API roster — the 'team' column in player_df is ignored for grouping.
    Players in player_df not found on any API roster are excluded.

    Returns {team_abbr: team_value}.
    """
    if player_df is None:
        player_df = get_player_lebron()

    if player_df.empty:
        return {}

    # Team-level CSV (no player column) — use values directly, no roster join possible
    if "player" not in player_df.columns:
        return dict(zip(player_df["team"].str.upper(), player_df["lebron"].astype(float)))

    df = player_df.copy()

    if rosters:
        # Build player name → team map from the API roster (authoritative)
        name_to_team = {}
        for abbr, roster_df in rosters.items():
            if "player" in roster_df.columns:
                for name in roster_df["player"]:
                    name_to_team[str(name).strip()] = abbr

        df["team"] = df["player"].map(lambda n: name_to_team.get(str(n).strip()))
        unmatched = df["team"].isna().sum()
        if unmatched:
            print(f"[fetch_data] {unmatched} LEBRON players not matched to any API roster (excluded)")
        df = df[df["team"].notna()]
    else:
        df["team"] = df["team"].str.upper()

    if df.empty:
        return {}

    if "lebron_war" in df.columns:
        df["lebron_war"] = pd.to_numeric(df["lebron_war"], errors="coerce").fillna(0)
        return df.groupby("team")["lebron_war"].sum().to_dict()

    df["lebron"] = pd.to_numeric(df["lebron"], errors="coerce").fillna(0)
    if "minutes" in df.columns:
        df["lebron_x_min"] = df["lebron"] * df["minutes"]
        agg = df.groupby("team").agg(lebron_x_min=("lebron_x_min", "sum"), total_min=("minutes", "sum"))
        return (agg["lebron_x_min"] / agg["total_min"]).to_dict()

    return df.groupby("team")["lebron"].mean().to_dict()


def get_team_rosters(season: int = 2026) -> dict[str, pd.DataFrame]:
    """
    Fetch the roster for every WNBA team via nba_api.
    Returns {team_abbr: DataFrame} where each DataFrame has columns:
      player_id, player, num, position, height, weight, birth_date,
      age, experience, school
    Results are cached in data/rosters_<season>.json for 24 hours.
    """
    cache_key = f"rosters_{season}"
    cached = _load_cache(cache_key)
    if cached:
        return {abbr: pd.DataFrame(rows) for abbr, rows in cached.items()}

    # Load stale cache as a fallback for teams whose fresh API fetch fails
    stale_cache = {}
    stale_path = _cache_path(cache_key)
    if stale_path.exists():
        try:
            import json as _json
            stale_cache = _json.loads(stale_path.read_text())
        except Exception:
            pass

    season_str = f"{season}-{str(season + 1)[-2:]}"

    try:
        from nba_api.stats.static import teams as static_teams
        from nba_api.stats.endpoints import commonteamroster
    except ImportError as e:
        print(f"[fetch_data] nba_api not available: {e}")
        return {}

    # Static abbreviations that differ from our tricodes
    STATIC_TO_TRICODE = {
        "PHO": "PHX",
        "NY":  "NYL",
    }

    # Build tricode → team_id map from the static WNBA teams list
    known_teams = set(TEAM_LOGO_SLUGS.keys())
    team_id_map = {}
    for t in static_teams.get_wnba_teams():
        static_abbr = t["abbreviation"].upper()
        abbr = STATIC_TO_TRICODE.get(static_abbr, static_abbr)
        if abbr in known_teams:
            team_id_map[abbr] = t["id"]

    if not team_id_map:
        print("[fetch_data] No WNBA teams found in static data.")
        return {}

    # Fetch roster for each team; fall back to stale cache entry on failure
    rosters = {}
    for abbr, team_id in sorted(team_id_map.items()):
        try:
            roster_ep = commonteamroster.CommonTeamRoster(
                team_id=team_id,
                season=season_str,
                league_id_nullable="10",
            )
            df = roster_ep.get_data_frames()[0]
            if df.empty:
                raise ValueError("API returned empty roster")

            col_map = {
                "PLAYER":     "player",
                "PLAYER_ID":  "player_id",
                "NUM":        "num",
                "POSITION":   "position",
                "HEIGHT":     "height",
                "WEIGHT":     "weight",
                "BIRTH_DATE": "birth_date",
                "AGE":        "age",
                "EXP":        "experience",
                "SCHOOL":     "school",
                "HOW_ACQUIRED": "how_acquired",
            }
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
            keep = [c for c in ["player_id", "player", "num", "position", "height",
                                "weight", "birth_date", "age", "experience", "school",
                                "how_acquired"] if c in df.columns]
            rosters[abbr] = df[keep]
            time.sleep(0.3)  # be polite to the API
        except Exception as e:
            print(f"[fetch_data] Roster fetch failed for {abbr}: {e}")
            if abbr in stale_cache and len(stale_cache[abbr]) > 0:
                print(f"[fetch_data] Using stale cache for {abbr}")
                rosters[abbr] = pd.DataFrame(stale_cache[abbr])

    # Preserve any stale entries that still couldn't be fetched
    for abbr, rows in stale_cache.items():
        if abbr not in rosters and len(rows) > 0:
            print(f"[fetch_data] Preserving stale roster for {abbr}")
            rosters[abbr] = pd.DataFrame(rows)

    _save_cache(cache_key, {abbr: df.to_dict(orient="records") for abbr, df in rosters.items()})
    print(f"[fetch_data] Rosters: fetched {len(rosters)}/{len(team_id_map)} teams")
    return rosters


def get_team_stats(season: int) -> pd.DataFrame:
    """
    Fetch WNBA team ORtg/DRtg from nba_api LeagueDashTeamStats (Advanced).
    Returns DataFrame with columns: team, ortg, drtg
    Cached for CACHE_TTL_HOURS like other endpoints.
    """
    cache_key = f"team_stats_{season}"
    cached = _load_cache(cache_key)
    if cached:
        return pd.DataFrame(cached)

    season_str = f"{season}-{str(season + 1)[-2:]}"
    STATIC_TO_TRICODE = {"PHO": "PHX", "NY": "NYL"}

    try:
        from nba_api.stats.endpoints import leaguedashteamstats
        stats = leaguedashteamstats.LeagueDashTeamStats(
            league_id_nullable="10",
            season=season_str,
            season_type_all_star="Regular Season",
            measure_type_detailed_defense="Advanced",
        )
        df = stats.get_data_frames()[0]
    except Exception as e:
        print(f"[fetch_data] Team stats fetch failed for {season}: {e}")
        return pd.DataFrame(columns=["team", "ortg", "drtg"])

    if df.empty:
        return pd.DataFrame(columns=["team", "ortg", "drtg"])

    name_to_abbr = {v: k for k, v in TEAM_NAMES.items()}

    rows = []
    for _, row in df.iterrows():
        name = str(row.get("TEAM_NAME", ""))
        abbr = name_to_abbr.get(name, STATIC_TO_TRICODE.get(name.upper(), name.upper()))
        ortg = float(row.get("E_OFF_RATING", row.get("OFF_RATING", 0)) or 0)
        drtg = float(row.get("E_DEF_RATING", row.get("DEF_RATING", 0)) or 0)
        if ortg and drtg:
            rows.append({"team": abbr, "ortg": ortg, "drtg": drtg})

    result = pd.DataFrame(rows)
    if not result.empty:
        _save_cache(cache_key, result.to_dict(orient="records"))
    return result


def get_team_logo(team_abbr: str) -> Path | None:
    """
    Fetch and cache team logo PNG from ESPN CDN.
    Returns local path or None if unavailable.
    """
    slug = TEAM_LOGO_SLUGS.get(team_abbr.upper())
    if not slug:
        return None

    logo_path = LOGOS_DIR / f"{team_abbr.lower()}.png"
    if logo_path.exists():
        return logo_path

    url = f"https://a.espncdn.com/i/teamlogos/wnba/500/{slug}.png"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        with open(logo_path, "wb") as f:
            f.write(resp.content)
        return logo_path
    except Exception as e:
        print(f"[fetch_data] Logo fetch failed for {team_abbr}: {e}")
        return None


def prefetch_all_logos():
    """Pre-download all team logos."""
    for abbr in TEAM_LOGO_SLUGS:
        get_team_logo(abbr)
        time.sleep(0.1)


if __name__ == "__main__":
    print("=== Fetching WNBA Schedule ===")
    schedule = get_schedule(2026)
    print(f"Games found: {len(schedule)}")
    if not schedule.empty:
        print(schedule.head(5).to_string())

    print("\n=== Fetching Player LEBRON ===")
    player_df = get_player_lebron()
    print(f"Players found: {len(player_df)}")
    if not player_df.empty:
        print(player_df.head(10).to_string())

    print("\n=== Team LEBRON Totals ===")
    team_lebron = get_team_lebron(player_df)
    for team, val in sorted(team_lebron.items(), key=lambda x: -x[1]):
        print(f"  {team}: {val:.2f}")

    print("\n=== Fetching Team Rosters ===")
    rosters = get_team_rosters(2026)
    for abbr, df in sorted(rosters.items()):
        print(f"  {abbr}: {len(df)} players")
        if not df.empty:
            print(df[["player", "position", "age"]].head(3).to_string(index=False))

    print("\n=== Fetching Logos ===")
    prefetch_all_logos()
    logos = list(LOGOS_DIR.glob("*.png"))
    print(f"Logos cached: {len(logos)}")
