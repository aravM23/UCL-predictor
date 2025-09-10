"""
Sample data generation for Champions League predictor demonstration.
Creates synthetic match data for testing and demo purposes.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
import random


def generate_sample_teams() -> List[str]:
    """Generate list of sample Champions League teams."""
    teams = [
        "Real Madrid", "Barcelona", "Atlético Madrid", "Sevilla",
        "Manchester City", "Liverpool", "Chelsea", "Arsenal", "Manchester United", "Tottenham",
        "Bayern München", "Borussia Dortmund", "RB Leipzig", "Eintracht Frankfurt",
        "PSG", "Marseille", "Monaco", "Lyon",
        "Juventus", "Milan", "Inter", "Napoli", "Roma", "Atalanta",
        "Ajax", "PSV Eindhoven", "Feyenoord",
        "Porto", "Benfica", "Sporting CP",
        "Club Brugge", "Genk",
        "Salzburg", "Rapid Wien",
        "Celtic", "Rangers",
        "Shakhtar Donetsk", "Dynamo Kyiv",
        "Olympiacos", "PAOK",
        "Galatasaray", "Fenerbahçe"
    ]
    return teams[:32]  # Select 32 teams for Champions League format


def generate_sample_matches(teams: List[str], seasons: List[str] = None) -> pd.DataFrame:
    """
    Generate sample Champions League match data.
    
    Args:
        teams: List of team names
        seasons: List of season strings (e.g., ["2021-22", "2022-23"])
        
    Returns:
        DataFrame with match data
    """
    if seasons is None:
        seasons = ["2021-22", "2022-23", "2023-24"]
    
    matches = []
    random.seed(42)
    np.random.seed(42)
    
    for season in seasons:
        # Group stage: 8 groups of 4 teams
        groups = [teams[i:i+4] for i in range(0, min(32, len(teams)), 4)]
        
        # Generate group stage matches
        for group_idx, group_teams in enumerate(groups):
            for home_team in group_teams:
                for away_team in group_teams:
                    if home_team != away_team:
                        # Each team plays each other twice (home and away)
                        for leg in [1, 2]:  # Use explicit values 1 and 2
                            match_date = generate_match_date(season, 'group', leg)
                            
                            # Simulate match result
                            home_goals, away_goals = simulate_match_score(home_team, away_team)
                            
                            # Generate realistic odds
                            odds_home, odds_draw, odds_away = generate_realistic_odds(
                                home_team, away_team, home_goals, away_goals
                            )
                            
                            matches.append({
                                'date': match_date,
                                'season': season,
                                'stage': 'GROUP_STAGE',
                                'group': f'Group {chr(65 + group_idx)}',
                                'home_team': home_team,
                                'away_team': away_team,
                                'home_goals': home_goals,
                                'away_goals': away_goals,
                                'result': 'H' if home_goals > away_goals else ('A' if away_goals > home_goals else 'D'),
                                'B365H': odds_home,
                                'B365D': odds_draw,
                                'B365A': odds_away,
                                'BWH': odds_home * random.uniform(0.95, 1.05),
                                'BWD': odds_draw * random.uniform(0.95, 1.05),
                                'BWA': odds_away * random.uniform(0.95, 1.05)
                            })
        
        # Knockout stage matches (simplified)
        knockout_teams = random.sample(teams, 16)  # Top 16 teams qualify
        knockout_stages = ['ROUND_OF_16', 'QUARTER_FINALS', 'SEMI_FINALS', 'FINAL']
        
        current_teams = knockout_teams
        for stage in knockout_stages:
            next_round_teams = []
            
            # Pair teams randomly
            random.shuffle(current_teams)
            pairs = [(current_teams[i], current_teams[i+1]) for i in range(0, len(current_teams), 2)]
            
            for home_team, away_team in pairs:
                # First leg
                match_date1 = generate_match_date(season, stage, 1)
                home_goals1, away_goals1 = simulate_match_score(home_team, away_team)
                odds_h1, odds_d1, odds_a1 = generate_realistic_odds(home_team, away_team, home_goals1, away_goals1)
                
                matches.append({
                    'date': match_date1,
                    'season': season,
                    'stage': stage,
                    'leg': 1,
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_goals': home_goals1,
                    'away_goals': away_goals1,
                    'result': 'H' if home_goals1 > away_goals1 else ('A' if away_goals1 > home_goals1 else 'D'),
                    'B365H': odds_h1,
                    'B365D': odds_d1,
                    'B365A': odds_a1,
                    'BWH': odds_h1 * random.uniform(0.95, 1.05),
                    'BWD': odds_d1 * random.uniform(0.95, 1.05),
                    'BWA': odds_a1 * random.uniform(0.95, 1.05)
                })
                
                # Second leg (except for final)
                if stage != 'FINAL':
                    match_date2 = generate_match_date(season, stage, 2)
                    home_goals2, away_goals2 = simulate_match_score(away_team, home_team)  # Reverse venue
                    odds_h2, odds_d2, odds_a2 = generate_realistic_odds(away_team, home_team, home_goals2, away_goals2)
                    
                    matches.append({
                        'date': match_date2,
                        'season': season,
                        'stage': stage,
                        'leg': 2,
                        'home_team': away_team,
                        'away_team': home_team,
                        'home_goals': home_goals2,
                        'away_goals': away_goals2,
                        'result': 'H' if home_goals2 > away_goals2 else ('A' if away_goals2 > home_goals2 else 'D'),
                        'B365H': odds_h2,
                        'B365D': odds_d2,
                        'B365A': odds_a2,
                        'BWH': odds_h2 * random.uniform(0.95, 1.05),
                        'BWD': odds_d2 * random.uniform(0.95, 1.05),
                        'BWA': odds_a2 * random.uniform(0.95, 1.05)
                    })
                
                # Determine aggregate winner (simplified)
                if stage != 'FINAL':
                    total_home = home_goals1 + away_goals2
                    total_away = away_goals1 + home_goals2
                    winner = home_team if total_home >= total_away else away_team
                else:
                    winner = home_team if home_goals1 > away_goals1 else away_team
                
                next_round_teams.append(winner)
            
            current_teams = next_round_teams
            if stage == 'FINAL':
                break
    
    return pd.DataFrame(matches)


def generate_match_date(season: str, stage: str, leg: int) -> str:
    """Generate realistic match date based on season and stage."""
    year = int(season.split('-')[0])
    
    # Define typical date ranges for each stage
    date_ranges = {
        'group': {
            1: (datetime(year, 9, 15), datetime(year, 12, 15)),
            2: (datetime(year, 9, 20), datetime(year, 12, 20))
        },
        'ROUND_OF_16': {
            1: (datetime(year + 1, 2, 15), datetime(year + 1, 3, 15)),
            2: (datetime(year + 1, 3, 1), datetime(year + 1, 3, 31))
        },
        'QUARTER_FINALS': {
            1: (datetime(year + 1, 4, 1), datetime(year + 1, 4, 15)),
            2: (datetime(year + 1, 4, 10), datetime(year + 1, 4, 25))
        },
        'SEMI_FINALS': {
            1: (datetime(year + 1, 4, 25), datetime(year + 1, 5, 5)),
            2: (datetime(year + 1, 5, 1), datetime(year + 1, 5, 10))
        },
        'FINAL': {
            1: (datetime(year + 1, 5, 25), datetime(year + 1, 6, 5)),
            2: (datetime(year + 1, 5, 25), datetime(year + 1, 6, 5))
        }
    }
    
    stage_key = stage.lower() if stage.lower() in date_ranges else 'group'
    start_date, end_date = date_ranges[stage_key][leg]
    
    # Random date within range
    delta = end_date - start_date
    random_days = random.randint(0, delta.days)
    match_date = start_date + timedelta(days=random_days)
    
    return match_date.strftime('%Y-%m-%d')


def simulate_match_score(home_team: str, away_team: str) -> tuple:
    """
    Simulate realistic match score based on team names.
    
    Args:
        home_team: Home team name
        away_team: Away team name
        
    Returns:
        Tuple of (home_goals, away_goals)
    """
    # Define team strength tiers (higher = stronger)
    elite_teams = ["Real Madrid", "Barcelona", "Manchester City", "Bayern München", "PSG", "Liverpool"]
    strong_teams = ["Chelsea", "Arsenal", "Juventus", "Milan", "Inter", "Atlético Madrid", "Borussia Dortmund"]
    
    # Assign base strength
    home_strength = 1.5  # Home advantage
    away_strength = 1.0
    
    if home_team in elite_teams:
        home_strength += 0.8
    elif home_team in strong_teams:
        home_strength += 0.4
    
    if away_team in elite_teams:
        away_strength += 0.8
    elif away_team in strong_teams:
        away_strength += 0.4
    
    # Simulate goals using Poisson distribution
    home_goals = np.random.poisson(home_strength)
    away_goals = np.random.poisson(away_strength)
    
    # Cap at reasonable values
    home_goals = min(home_goals, 6)
    away_goals = min(away_goals, 6)
    
    return int(home_goals), int(away_goals)


def generate_realistic_odds(home_team: str, away_team: str, home_goals: int, away_goals: int) -> tuple:
    """
    Generate realistic betting odds based on teams and actual result.
    
    Args:
        home_team: Home team name
        away_team: Away team name
        home_goals: Actual home goals scored
        away_goals: Actual away goals scored
        
    Returns:
        Tuple of (odds_home, odds_draw, odds_away)
    """
    elite_teams = ["Real Madrid", "Barcelona", "Manchester City", "Bayern München", "PSG", "Liverpool"]
    strong_teams = ["Chelsea", "Arsenal", "Juventus", "Milan", "Inter", "Atlético Madrid", "Borussia Dortmund"]
    
    # Base probabilities
    home_prob = 0.45  # Home advantage
    draw_prob = 0.27
    away_prob = 0.28
    
    # Adjust based on team strength
    if home_team in elite_teams and away_team not in elite_teams:
        home_prob += 0.15
        away_prob -= 0.10
        draw_prob -= 0.05
    elif away_team in elite_teams and home_team not in elite_teams:
        away_prob += 0.15
        home_prob -= 0.10
        draw_prob -= 0.05
    elif home_team in strong_teams and away_team not in strong_teams + elite_teams:
        home_prob += 0.08
        away_prob -= 0.05
        draw_prob -= 0.03
    elif away_team in strong_teams and home_team not in strong_teams + elite_teams:
        away_prob += 0.08
        home_prob -= 0.05
        draw_prob -= 0.03
    
    # Ensure probabilities are valid
    total_prob = home_prob + draw_prob + away_prob
    home_prob /= total_prob
    draw_prob /= total_prob
    away_prob /= total_prob
    
    # Convert to odds with bookmaker margin
    margin = 1.05  # 5% margin
    odds_home = margin / home_prob
    odds_draw = margin / draw_prob
    odds_away = margin / away_prob
    
    # Add some randomness
    odds_home *= random.uniform(0.95, 1.05)
    odds_draw *= random.uniform(0.95, 1.05)
    odds_away *= random.uniform(0.95, 1.05)
    
    return round(odds_home, 2), round(odds_draw, 2), round(odds_away, 2)


def generate_upcoming_fixtures(teams: List[str], num_fixtures: int = 10) -> pd.DataFrame:
    """
    Generate upcoming fixture data for predictions.
    
    Args:
        teams: List of team names
        num_fixtures: Number of fixtures to generate
        
    Returns:
        DataFrame with fixture data
    """
    fixtures = []
    start_date = datetime.now() + timedelta(days=1)
    
    for i in range(num_fixtures):
        # Random team pairing
        home_team, away_team = random.sample(teams, 2)
        
        # Future date
        fixture_date = start_date + timedelta(days=random.randint(1, 30))
        
        # Generate odds (no actual result yet)
        odds_home, odds_draw, odds_away = generate_realistic_odds(home_team, away_team, 0, 0)
        
        fixtures.append({
            'date': fixture_date.strftime('%Y-%m-%d'),
            'home_team': home_team,
            'away_team': away_team,
            'stage': 'UPCOMING',
            'odds_home': odds_home,
            'odds_draw': odds_draw,
            'odds_away': odds_away
        })
    
    return pd.DataFrame(fixtures)


def create_sample_datasets():
    """Create and save sample datasets for the predictor."""
    print("Generating sample Champions League data...")
    
    # Generate teams and matches
    teams = generate_sample_teams()
    matches_df = generate_sample_matches(teams)
    fixtures_df = generate_upcoming_fixtures(teams)
    
    # Save datasets
    from pathlib import Path
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    matches_df.to_csv(data_dir / "sample_matches.csv", index=False)
    fixtures_df.to_csv(data_dir / "sample_fixtures.csv", index=False)
    
    print(f"Generated {len(matches_df)} historical matches")
    print(f"Generated {len(fixtures_df)} upcoming fixtures")
    print(f"Data saved to {data_dir}/")
    
    # Print sample statistics
    print("\nSample Statistics:")
    print(f"Total seasons: {matches_df['season'].nunique()}")
    print(f"Total teams: {matches_df['home_team'].nunique()}")
    print(f"Matches per season: {len(matches_df) // matches_df['season'].nunique()}")
    print(f"Date range: {matches_df['date'].min()} to {matches_df['date'].max()}")
    
    # Result distribution
    result_dist = matches_df['result'].value_counts(normalize=True)
    print(f"\nResult distribution:")
    print(f"Home wins: {result_dist.get('H', 0):.1%}")
    print(f"Draws: {result_dist.get('D', 0):.1%}")
    print(f"Away wins: {result_dist.get('A', 0):.1%}")
    
    return matches_df, fixtures_df


if __name__ == "__main__":
    # Generate sample data
    matches_df, fixtures_df = create_sample_datasets()
    
    # Display sample records
    print("\nSample match records:")
    print(matches_df.head().to_string(index=False))
    
    print("\nSample fixture records:")
    print(fixtures_df.head().to_string(index=False))
