"""
model.py — Win probability model and season simulation
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

# --- Tunable constants ---
# k: how much a unit of LEBRON difference shifts the logit
# Calibrated so that a 21-point WAR gap (best vs. worst team) ≈ 97% win probability → -15.5 spread
K = 0.15

# Home court advantage in logit space (≈ +3.5% win probability for equal teams)
HOME_ADV_LOGIT = 0.20

# Number of Monte Carlo simulations
N_SIMS = 1000

# WNBA playoff cutoff (top N teams make playoffs)
PLAYOFF_CUTOFF = 8


@dataclass
class SimResults:
    teams: list          # list of team abbreviations
    wins_matrix: np.ndarray   # shape (n_teams, N_SIMS) — wins per team per sim
    actual_wins: dict    # {team: wins already locked in from completed games}
    total_games: dict    # {team: total games in season}

    @property
    def projected_wins(self) -> pd.DataFrame:
        """Returns DataFrame with median, p10, p90, and actual wins per team."""
        total = self.wins_matrix  # already includes actual wins
        return pd.DataFrame({
            "team": self.teams,
            "median_wins": np.median(total, axis=1),
            "mean_wins": np.mean(total, axis=1),
            "std_wins": np.std(total, axis=1),
            "p10_wins": np.percentile(total, 10, axis=1),
            "p90_wins": np.percentile(total, 90, axis=1),
            "actual_wins": [self.actual_wins.get(t, 0) for t in self.teams],
        }).sort_values("median_wins", ascending=False).reset_index(drop=True)

    @property
    def playoff_odds(self) -> pd.DataFrame:
        """% chance each team finishes in top PLAYOFF_CUTOFF by wins."""
        n_sims = self.wins_matrix.shape[1]
        playoff_pct = []
        for i in range(len(self.teams)):
            # Count sims where this team is in top 8
            team_wins = self.wins_matrix[i]
            # For each sim, rank teams; count how many times this team is top-8
            count = 0
            for s in range(n_sims):
                sim_wins = self.wins_matrix[:, s]
                rank = np.sum(sim_wins > team_wins[s])  # teams with strictly more wins
                if rank < PLAYOFF_CUTOFF:
                    count += 1
            playoff_pct.append(count / n_sims * 100)

        return pd.DataFrame({
            "team": self.teams,
            "playoff_pct": playoff_pct,
        }).sort_values("playoff_pct", ascending=False).reset_index(drop=True)

    @property
    def seed_probabilities(self) -> pd.DataFrame:
        """
        Returns DataFrame: rows=teams, columns=Seed 1..PLAYOFF_CUTOFF + Missed.
        Values are percentages.
        """
        n_sims = self.wins_matrix.shape[1]
        n_teams = len(self.teams)
        seed_counts = np.zeros((n_teams, PLAYOFF_CUTOFF + 1))  # +1 for "missed"

        for s in range(n_sims):
            sim_wins = self.wins_matrix[:, s]
            # Sort descending by wins; handle ties by random tiebreak
            noise = np.random.uniform(0, 0.001, n_teams)
            order = np.argsort(-(sim_wins + noise))
            for rank, team_idx in enumerate(order):
                if rank < PLAYOFF_CUTOFF:
                    seed_counts[team_idx, rank] += 1
                else:
                    seed_counts[team_idx, PLAYOFF_CUTOFF] += 1

        seed_pct = seed_counts / n_sims * 100
        cols = [f"Seed {i+1}" for i in range(PLAYOFF_CUTOFF)] + ["Missed"]
        df = pd.DataFrame(seed_pct, columns=cols)
        df.insert(0, "team", self.teams)

        # Sort by most likely seed (weighted average seed position)
        seed_weights = np.arange(1, PLAYOFF_CUTOFF + 2, dtype=float)
        df["_avg_seed"] = seed_pct @ seed_weights / 100
        df = df.sort_values("_avg_seed").drop(columns="_avg_seed").reset_index(drop=True)
        return df


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def win_probability(
    home_lebron: float,
    away_lebron: float,
    is_neutral: bool = False,
) -> float:
    """
    Probability that the HOME team wins.
    Uses logistic model: sigmoid(K * (home_lebron - away_lebron) + home_adv)
    """
    diff = home_lebron - away_lebron
    home_adv = 0.0 if is_neutral else HOME_ADV_LOGIT
    return float(sigmoid(K * diff + home_adv))


def win_prob_to_spread(home_win_prob: float) -> float:
    """
    Convert home win probability to a point spread from the home team's perspective.
    Negative = home team is favored. Positive = away team is favored.
    Uses ~2.5% per point (wider than NBA to reflect larger WNBA talent disparities).
    e.g. 50% → 0 (pick 'em), 96.6% → -18.6 (max realistic with ~21pt WAR gap)
    """
    return -(home_win_prob - 0.5) * 40.0


LEAGUE_AVG_ORTG = 82.0  # WNBA avg team points per game


def project_game_total(
    home_ortg: float,
    home_drtg: float,
    away_ortg: float,
    away_drtg: float,
) -> float:
    """
    Project combined game total from team ORtg/DRtg.
    Formula: each team's expected score = their ORtg adjusted for opponent DRtg vs league avg.
    """
    home_exp = home_ortg + (away_drtg - LEAGUE_AVG_ORTG)
    away_exp = away_ortg + (home_drtg - LEAGUE_AVG_ORTG)
    return round(home_exp + away_exp, 1)


def lebron_to_spread(home_lebron: float, away_lebron: float, is_home: bool = True) -> float:
    """
    Convenience: compute spread from LEBRON values directly (includes home court).
    """
    wp = win_probability(home_lebron, away_lebron, is_neutral=not is_home)
    return win_prob_to_spread(wp)


def simulate_season(
    schedule_df: pd.DataFrame,
    team_lebron: dict,
    n_sims: int = N_SIMS,
) -> SimResults:
    """
    Monte Carlo simulation of the remaining WNBA season.

    schedule_df: full season schedule (from fetch_data.get_schedule)
    team_lebron: {team_abbr: total_lebron_value}
    n_sims: number of simulations

    Returns SimResults with wins_matrix of shape (n_teams, n_sims).
    """
    all_teams = sorted(
        set(schedule_df["home_team"].tolist() + schedule_df["away_team"].tolist())
    )
    team_idx = {t: i for i, t in enumerate(all_teams)}
    n_teams = len(all_teams)

    # Default LEBRON for teams not in the ratings data (league average = 0)
    def get_lebron(team: str) -> float:
        return team_lebron.get(team, 0.0)

    # Lock in actual results from completed games
    actual_wins = {t: 0 for t in all_teams}
    completed = schedule_df[schedule_df["status"] == "Final"]
    for _, row in completed.iterrows():
        if row["home_score"] is not None and row["away_score"] is not None:
            if row["home_score"] > row["away_score"]:
                actual_wins[row["home_team"]] = actual_wins.get(row["home_team"], 0) + 1
            else:
                actual_wins[row["away_team"]] = actual_wins.get(row["away_team"], 0) + 1

    # Remaining (unplayed) games
    remaining = schedule_df[schedule_df["status"] != "Final"].reset_index(drop=True)

    # Pre-compute win probabilities for remaining games
    win_probs = np.array([
        win_probability(get_lebron(row["home_team"]), get_lebron(row["away_team"]))
        for _, row in remaining.iterrows()
    ])

    home_indices = np.array([team_idx[row["home_team"]] for _, row in remaining.iterrows()])
    away_indices = np.array([team_idx[row["away_team"]] for _, row in remaining.iterrows()])

    # Monte Carlo simulation
    rng = np.random.default_rng(seed=42)
    draws = rng.random((n_sims, len(remaining)))  # shape (n_sims, n_games)
    home_wins = draws < win_probs[np.newaxis, :]   # shape (n_sims, n_games)

    wins_matrix = np.zeros((n_teams, n_sims), dtype=np.float32)

    # Add actual wins baseline
    for team, w in actual_wins.items():
        wins_matrix[team_idx[team], :] = w

    # Add simulated wins
    for g in range(len(remaining)):
        wins_matrix[home_indices[g], home_wins[:, g]] += 1
        wins_matrix[away_indices[g], ~home_wins[:, g]] += 1

    # Total games per team
    total_games = {t: 0 for t in all_teams}
    for _, row in schedule_df.iterrows():
        total_games[row["home_team"]] = total_games.get(row["home_team"], 0) + 1
        total_games[row["away_team"]] = total_games.get(row["away_team"], 0) + 1

    return SimResults(
        teams=all_teams,
        wins_matrix=wins_matrix,
        actual_wins=actual_wins,
        total_games=total_games,
    )


def get_next_gamedays(schedule_df: pd.DataFrame, n: int = 2) -> list[str]:
    """Return the next N calendar dates with scheduled (unplayed) games."""
    from datetime import date
    today = date.today()
    upcoming = schedule_df[
        (schedule_df["status"] != "Final") &
        (pd.to_datetime(schedule_df["date"]).dt.date >= today)
    ]
    future_dates = sorted(pd.to_datetime(upcoming["date"]).dt.date.unique())
    return [str(d) for d in future_dates[:n]]


if __name__ == "__main__":
    print("=== Win Probability Tests ===")
    tests = [
        (10.0, 10.0, "equal teams"),
        (15.0, 10.0, "home +5 LEBRON"),
        (10.0, 15.0, "home -5 LEBRON"),
        (20.0, 10.0, "home +10 LEBRON"),
    ]
    for h, a, desc in tests:
        p = win_probability(h, a)
        spread = win_prob_to_spread(p)
        print(f"  {desc}: home win% = {p:.1%}, spread = {spread:+.1f}")

    print("\n=== Simulation Test (synthetic schedule) ===")
    import pandas as pd

    synthetic_schedule = pd.DataFrame([
        {"game_id": 1, "date": "2026-05-15", "home_team": "LVA", "away_team": "NYL", "home_score": None, "away_score": None, "status": "Scheduled"},
        {"game_id": 2, "date": "2026-05-15", "home_team": "SEA", "away_team": "MIN", "home_score": None, "away_score": None, "status": "Scheduled"},
        {"game_id": 3, "date": "2026-05-16", "home_team": "CON", "away_team": "IND", "home_score": None, "away_score": None, "status": "Scheduled"},
        {"game_id": 4, "date": "2026-05-16", "home_team": "CHI", "away_team": "ATL", "home_score": 85, "away_score": 80, "status": "Final"},
    ])
    synthetic_lebron = {
        "LVA": 25.0, "NYL": 22.0, "SEA": 20.0, "MIN": 18.0,
        "CON": 19.0, "IND": 21.0, "CHI": 16.0, "ATL": 14.0,
    }

    results = simulate_season(synthetic_schedule, synthetic_lebron, n_sims=1000)
    print("\nProjected Wins:")
    print(results.projected_wins.to_string(index=False))
    print("\nPlayoff Odds:")
    print(results.playoff_odds.to_string(index=False))
    print("\nSeed Probabilities (top cols):")
    sp = results.seed_probabilities
    print(sp[["team", "Seed 1", "Seed 2", "Seed 3", "Seed 4", "Missed"]].to_string(index=False))
