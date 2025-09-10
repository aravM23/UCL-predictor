"""
Feature engineering module for Champions League predictor.
Builds comprehensive features for match prediction including team strength,
form, schedule factors, and contextual variables.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ELORatingSystem:
    """ELO rating system for team strength estimation."""
    
    def __init__(self, k_factor: int = 25, initial_rating: int = 1500, home_advantage: int = 100):
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.home_advantage = home_advantage
        self.ratings = {}
        self.rating_history = []
        
    def get_rating(self, team: str, date: str = None) -> float:
        """Get current rating for a team."""
        if team not in self.ratings:
            self.ratings[team] = self.initial_rating
        return self.ratings[team]
    
    def expected_score(self, rating_a: float, rating_b: float, home_advantage: float = 0) -> float:
        """Calculate expected score for team A vs team B."""
        return 1 / (1 + 10**((rating_b - rating_a - home_advantage) / 400))
    
    def update_ratings(self, home_team: str, away_team: str, result: str, date: str, goals_home: int, goals_away: int):
        """
        Update ELO ratings after a match.
        
        Args:
            home_team: Home team name
            away_team: Away team name  
            result: Match result ('H', 'D', 'A')
            date: Match date
            goals_home: Home team goals
            goals_away: Away team goals
        """
        # Get current ratings
        rating_home = self.get_rating(home_team)
        rating_away = self.get_rating(away_team)
        
        # Calculate expected scores
        expected_home = self.expected_score(rating_home, rating_away, self.home_advantage)
        expected_away = 1 - expected_home
        
        # Actual scores
        if result == 'H':
            actual_home, actual_away = 1.0, 0.0
        elif result == 'A':
            actual_home, actual_away = 0.0, 1.0
        else:  # Draw
            actual_home, actual_away = 0.5, 0.5
            
        # Goal difference multiplier (bigger wins = bigger rating changes)
        goal_diff = abs(goals_home - goals_away)
        multiplier = np.log(max(goal_diff, 1)) + 1
        
        # Update ratings
        change_home = self.k_factor * multiplier * (actual_home - expected_home)
        change_away = self.k_factor * multiplier * (actual_away - expected_away)
        
        # Store old ratings for history
        old_home = self.ratings[home_team]
        old_away = self.ratings[away_team]
        
        # Apply changes
        self.ratings[home_team] += change_home
        self.ratings[away_team] += change_away
        
        # Record history
        self.rating_history.append({
            'date': date,
            'home_team': home_team,
            'away_team': away_team,
            'home_rating_old': old_home,
            'away_rating_old': old_away,
            'home_rating_new': self.ratings[home_team],
            'away_rating_new': self.ratings[away_team],
            'home_change': change_home,
            'away_change': change_away,
            'result': result
        })
        
    def fit_historical_data(self, matches_df: pd.DataFrame):
        """Fit ELO ratings on historical match data."""
        logger.info("Fitting ELO ratings on historical data...")
        
        matches_sorted = matches_df.sort_values('date').copy()
        
        for _, match in matches_sorted.iterrows():
            if pd.notna(match['result']):  # Only update on completed matches
                self.update_ratings(
                    match['home_team'],
                    match['away_team'], 
                    match['result'],
                    match['date'].strftime('%Y-%m-%d'),
                    match['home_goals'],
                    match['away_goals']
                )
        
        logger.info(f"ELO ratings fitted for {len(self.ratings)} teams")
        
    def get_rating_features(self, home_team: str, away_team: str, date: str, lookback_matches: int = 5) -> Dict:
        """Generate ELO-based features for a match."""
        
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)
        
        # Get recent rating changes
        recent_home = self._get_recent_changes(home_team, date, lookback_matches)
        recent_away = self._get_recent_changes(away_team, date, lookback_matches)
        
        return {
            'home_elo': home_rating,
            'away_elo': away_rating,
            'elo_diff': home_rating - away_rating,
            'elo_ratio': home_rating / max(away_rating, 1),
            'home_elo_change_5': recent_home,
            'away_elo_change_5': recent_away,
            'elo_change_diff': recent_home - recent_away
        }
    
    def _get_recent_changes(self, team: str, date: str, matches: int) -> float:
        """Get ELO rating change over recent matches."""
        team_history = [h for h in self.rating_history 
                       if (h['home_team'] == team or h['away_team'] == team) 
                       and h['date'] < date]
        
        if len(team_history) < 2:
            return 0.0
            
        recent = team_history[-matches:]
        if not recent:
            return 0.0
            
        # Sum of rating changes
        total_change = 0
        for h in recent:
            if h['home_team'] == team:
                total_change += h['home_change']
            else:
                total_change += h['away_change']
                
        return total_change


class FeatureBuilder:
    """Main feature engineering class."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.elo_system = ELORatingSystem(
            k_factor=config.get('features', {}).get('elo_k', 25),
            initial_rating=config.get('features', {}).get('elo_initial', 1500)
        )
        
    def build_features(self, df_matches: pd.DataFrame, df_odds: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build comprehensive feature set for match prediction.
        
        Args:
            df_matches: DataFrame with match data
            df_odds: Optional odds data
            
        Returns:
            Tuple of (features_df, targets_df)
        """
        logger.info("Building features...")
        
        df = df_matches.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Fit ELO system on completed matches
        completed_matches = df[df['result'].notna()].copy()
        self.elo_system.fit_historical_data(completed_matches)
        
        # Build rolling team statistics
        df = self._add_rolling_features(df)
        
        # Add ELO features
        df = self._add_elo_features(df)
        
        # Add contextual features
        df = self._add_context_features(df)
        
        # Add schedule features
        df = self._add_schedule_features(df)
        
        # Add odds features if available
        if df_odds is not None:
            df = self._add_odds_features(df, df_odds)
            
        # Create interaction features
        df = self._add_interaction_features(df)
        
        # Split features and targets
        feature_cols = [col for col in df.columns 
                       if not col in ['home_goals', 'away_goals', 'result', 'date', 'home_team', 'away_team']]
        
        features_df = df[['date', 'home_team', 'away_team'] + feature_cols].copy()
        
        # Create targets for completed matches
        targets_df = df[df['result'].notna()][['date', 'home_team', 'away_team', 'home_goals', 'away_goals', 'result']].copy()
        
        logger.info(f"Built {len(feature_cols)} features for {len(features_df)} matches")
        
        return features_df, targets_df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling team statistics."""
        
        # Convert to long format for easier rolling calculations
        home_df = df[['date', 'home_team', 'home_goals', 'away_goals', 'result']].copy()
        home_df.columns = ['date', 'team', 'goals_for', 'goals_against', 'result_raw']
        home_df['home'] = 1
        home_df['points'] = home_df['result_raw'].map({'H': 3, 'D': 1, 'A': 0})
        
        away_df = df[['date', 'away_team', 'away_goals', 'home_goals', 'result']].copy()  
        away_df.columns = ['date', 'team', 'goals_for', 'goals_against', 'result_raw']
        away_df['home'] = 0
        away_df['points'] = away_df['result_raw'].map({'A': 3, 'D': 1, 'H': 0})
        
        # Combine
        long_df = pd.concat([home_df, away_df]).sort_values(['date', 'team']).reset_index(drop=True)
        
        # Calculate rolling features
        lookbacks = self.config.get('features', {}).get('lookbacks', [5, 10])
        
        for window in lookbacks:
            rolling = long_df.groupby('team').rolling(window, min_periods=1)
            
            long_df[f'goals_for_l{window}'] = rolling['goals_for'].mean().reset_index(level=0, drop=True)
            long_df[f'goals_against_l{window}'] = rolling['goals_against'].mean().reset_index(level=0, drop=True)
            long_df[f'goal_diff_l{window}'] = long_df[f'goals_for_l{window}'] - long_df[f'goals_against_l{window}']
            long_df[f'points_l{window}'] = rolling['points'].mean().reset_index(level=0, drop=True)
            
        # Split back to home/away and merge
        home_features = long_df[long_df['home'] == 1].copy()
        away_features = long_df[long_df['home'] == 0].copy()
        
        # Rename columns
        feature_cols = [col for col in home_features.columns if col.endswith(tuple(f'_l{w}' for w in lookbacks))]
        
        home_features = home_features[['date', 'team'] + feature_cols].add_prefix('home_')
        home_features = home_features.rename(columns={'home_date': 'date', 'home_team': 'home_team'})
        
        away_features = away_features[['date', 'team'] + feature_cols].add_prefix('away_')  
        away_features = away_features.rename(columns={'away_date': 'date', 'away_team': 'away_team'})
        
        # Merge back to main dataframe
        df = df.merge(home_features, on=['date', 'home_team'], how='left')
        df = df.merge(away_features, on=['date', 'away_team'], how='left')
        
        return df
    
    def _add_elo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ELO rating features."""
        
        elo_features = []
        for _, row in df.iterrows():
            features = self.elo_system.get_rating_features(
                row['home_team'], 
                row['away_team'],
                row['date'].strftime('%Y-%m-%d')
            )
            elo_features.append(features)
            
        elo_df = pd.DataFrame(elo_features)
        df = pd.concat([df.reset_index(drop=True), elo_df], axis=1)
        
        return df
    
    def _add_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add contextual match features."""
        
        # Home field advantage
        df['home_flag'] = 1
        
        # Stage/round encoding (if available)
        if 'stage' in df.columns:
            stage_encoding = {
                'GROUP_STAGE': 1,
                'ROUND_OF_16': 2, 
                'QUARTER_FINALS': 3,
                'SEMI_FINALS': 4,
                'FINAL': 5
            }
            df['stage_encoded'] = df['stage'].map(stage_encoding).fillna(1)
            df['knockout_flag'] = (df['stage_encoded'] > 1).astype(int)
        else:
            df['stage_encoded'] = 1
            df['knockout_flag'] = 0
            
        # Time-based features
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'].isin([5, 6])).astype(int)
        
        return df
    
    def _add_schedule_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add schedule and fatigue features."""
        
        # Calculate rest days between matches for each team
        team_matches = {}
        rest_days_home = []
        rest_days_away = []
        
        for _, row in df.iterrows():
            home_team = row['home_team']
            away_team = row['away_team']
            match_date = row['date']
            
            # Home team rest days
            if home_team in team_matches:
                last_date = team_matches[home_team]
                rest_days = (match_date - last_date).days
            else:
                rest_days = 7  # Default
            rest_days_home.append(min(rest_days, 14))  # Cap at 14 days
            
            # Away team rest days  
            if away_team in team_matches:
                last_date = team_matches[away_team]
                rest_days = (match_date - last_date).days
            else:
                rest_days = 7  # Default
            rest_days_away.append(min(rest_days, 14))  # Cap at 14 days
            
            # Update last match dates
            team_matches[home_team] = match_date
            team_matches[away_team] = match_date
            
        df['home_rest_days'] = rest_days_home
        df['away_rest_days'] = rest_days_away
        df['rest_days_diff'] = df['home_rest_days'] - df['away_rest_days']
        
        return df
    
    def _add_odds_features(self, df: pd.DataFrame, df_odds: pd.DataFrame) -> pd.DataFrame:
        """Add bookmaker odds features."""
        
        # Merge odds data
        df = df.merge(df_odds, on=['date', 'home_team', 'away_team'], how='left')
        
        # Calculate consensus odds (average across bookmakers)
        odds_cols_h = [col for col in df.columns if col.endswith('H')]
        odds_cols_d = [col for col in df.columns if col.endswith('D')]  
        odds_cols_a = [col for col in df.columns if col.endswith('A')]
        
        if odds_cols_h:
            df['odds_home'] = df[odds_cols_h].mean(axis=1)
            df['odds_draw'] = df[odds_cols_d].mean(axis=1)
            df['odds_away'] = df[odds_cols_a].mean(axis=1)
            
            # Convert to implied probabilities
            df['imp_prob_home'] = 1 / df['odds_home']
            df['imp_prob_draw'] = 1 / df['odds_draw']
            df['imp_prob_away'] = 1 / df['odds_away']
            
            # Remove bookmaker margin (normalize)
            total_prob = df['imp_prob_home'] + df['imp_prob_draw'] + df['imp_prob_away']
            df['imp_prob_home_norm'] = df['imp_prob_home'] / total_prob
            df['imp_prob_draw_norm'] = df['imp_prob_draw'] / total_prob
            df['imp_prob_away_norm'] = df['imp_prob_away'] / total_prob
            
            # Market sentiment features
            df['home_favorite'] = (df['odds_home'] < df['odds_away']).astype(int)
            df['odds_ratio'] = df['odds_home'] / df['odds_away']
            
        return df
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction and delta features."""
        
        # ELO vs form interactions
        if 'home_elo' in df.columns and 'home_goals_for_l5' in df.columns:
            df['elo_form_home'] = df['home_elo'] * df['home_goals_for_l5']
            df['elo_form_away'] = df['away_elo'] * df['away_goals_for_l5']
            
        # Home advantage vs ELO
        if 'home_elo' in df.columns:
            df['elo_home_advantage'] = df['home_elo'] * df['home_flag']
            
        # Form deltas
        form_features = [col for col in df.columns if 'goals_for_l' in col or 'goals_against_l' in col]
        for feature in form_features:
            if feature.startswith('home_'):
                away_feature = feature.replace('home_', 'away_')
                if away_feature in df.columns:
                    delta_name = feature.replace('home_', '') + '_delta'
                    df[delta_name] = df[feature] - df[away_feature]
                    
        return df


def build_features(df_matches: pd.DataFrame, df_odds: pd.DataFrame = None, 
                  elo_table: pd.DataFrame = None, config: Dict = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main function to build features for the Champions League predictor.
    
    Args:
        df_matches: Historical match data
        df_odds: Odds data (optional)
        elo_table: Pre-computed ELO ratings (optional)
        config: Configuration dictionary
        
    Returns:
        Tuple of (features_df, targets_df)
    """
    if config is None:
        config = {}
        
    builder = FeatureBuilder(config)
    return builder.build_features(df_matches, df_odds)


if __name__ == "__main__":
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input match data CSV')
    parser.add_argument('--output', required=True, help='Output features pickle file')
    parser.add_argument('--config', default='configs/features.yaml')
    parser.add_argument('--odds', help='Optional odds data CSV')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    df_matches = pd.read_csv(args.input)
    df_matches['date'] = pd.to_datetime(df_matches['date'])
    
    df_odds = None
    if args.odds:
        df_odds = pd.read_csv(args.odds)
        df_odds['date'] = pd.to_datetime(df_odds['date'])
    
    # Build features
    features_df, targets_df = build_features(df_matches, df_odds, config=config)
    
    # Save
    import pickle
    with open(args.output, 'wb') as f:
        pickle.dump({'features': features_df, 'targets': targets_df}, f)
        
    print(f"Features saved to {args.output}")
    print(f"Feature shape: {features_df.shape}")
    print(f"Target shape: {targets_df.shape}")
