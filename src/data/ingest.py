"""
Data ingestion module for Champions League predictor.
Loads and preprocesses raw match data from various sources.
"""

import pandas as pd
import numpy as np
import requests
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DataIngester:
    """Main class for ingesting football data from various sources."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_dir = Path("data/raw")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def load_kaggle_data(self, database_path: str) -> pd.DataFrame:
        """
        Load Champions League data from Kaggle European Soccer Database.
        
        Args:
            database_path: Path to the SQLite database file
            
        Returns:
            DataFrame with match data
        """
        conn = sqlite3.connect(database_path)
        
        # Query for Champions League matches only
        query = """
        SELECT 
            m.date,
            m.season,
            m.stage,
            ht.team_long_name as home_team,
            at.team_long_name as away_team,
            m.home_team_goal,
            m.away_team_goal,
            m.B365H, m.B365D, m.B365A,  -- Bet365 odds
            m.BWH, m.BWD, m.BWA,        -- Betway odds  
            m.IWH, m.IWD, m.IWA,        -- Interwetten odds
            m.PSH, m.PSD, m.PSA,        -- Pinnacle odds
            m.WHH, m.WHD, m.WHA,        -- William Hill odds
            m.VCH, m.VCD, m.VCA,        -- VC Bet odds
            c.name as country,
            l.name as league
        FROM Match m
        JOIN Team ht ON m.home_team_api_id = ht.team_api_id
        JOIN Team at ON m.away_team_api_id = at.team_api_id
        JOIN Country c ON m.country_id = c.id
        JOIN League l ON m.league_id = l.id
        WHERE l.name LIKE '%Champions League%'
           OR l.name LIKE '%European Cup%'
        ORDER BY m.date
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Clean and standardize
        df = self._clean_match_data(df)
        logger.info(f"Loaded {len(df)} Champions League matches from Kaggle")
        
        return df
    
    def load_api_fixtures(self, api_key: str, days_ahead: int = 30) -> pd.DataFrame:
        """
        Load upcoming fixtures from football-data.org API.
        
        Args:
            api_key: API key for football-data.org
            days_ahead: How many days ahead to fetch fixtures
            
        Returns:
            DataFrame with upcoming fixtures
        """
        headers = {'X-Auth-Token': api_key}
        base_url = "https://api.football-data.org/v4"
        
        # Champions League competition ID
        competition_id = "CL"
        
        # Date range
        date_from = datetime.now().strftime('%Y-%m-%d')
        date_to = (datetime.now() + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
        
        url = f"{base_url}/competitions/{competition_id}/matches"
        params = {
            'dateFrom': date_from,
            'dateTo': date_to,
            'status': 'SCHEDULED'
        }
        
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        fixtures = []
        
        for match in data.get('matches', []):
            fixtures.append({
                'date': match['utcDate'][:10],  # Extract date part
                'home_team': match['homeTeam']['name'],
                'away_team': match['awayTeam']['name'],
                'stage': match.get('stage', 'UNKNOWN'),
                'matchday': match.get('matchday'),
                'venue': match.get('venue', {}).get('name'),
                'status': match['status']
            })
        
        df = pd.DataFrame(fixtures)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = self._standardize_team_names(df)
        
        logger.info(f"Loaded {len(df)} upcoming fixtures from API")
        return df
    
    def _clean_match_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize match data."""
        
        # Convert date
        df['date'] = pd.to_datetime(df['date'])
        
        # Rename columns for consistency
        df = df.rename(columns={
            'home_team_goal': 'home_goals',
            'away_team_goal': 'away_goals'
        })
        
        # Standardize team names
        df = self._standardize_team_names(df)
        
        # Create match result labels
        df['result'] = np.where(
            df['home_goals'] > df['away_goals'], 'H',
            np.where(df['home_goals'] < df['away_goals'], 'A', 'D')
        )
        
        # Clean odds data
        odds_cols = [col for col in df.columns if any(bookie in col for bookie in ['B365', 'BW', 'IW', 'PS', 'WH', 'VC'])]
        for col in odds_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Remove matches with missing core data
        df = df.dropna(subset=['home_goals', 'away_goals'])
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        return df
    
    def _standardize_team_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize team names across different data sources."""
        
        # Common team name mappings
        name_mappings = {
            'Manchester City FC': 'Manchester City',
            'Manchester United FC': 'Manchester United', 
            'FC Barcelona': 'Barcelona',
            'Real Madrid CF': 'Real Madrid',
            'Bayern Munich': 'Bayern München',
            'Borussia Dortmund': 'Borussia Dortmund',
            'Paris Saint-Germain': 'PSG',
            'Atletico Madrid': 'Atlético Madrid',
            'AC Milan': 'Milan',
            'Inter Milan': 'Inter',
            'Juventus FC': 'Juventus',
            'Liverpool FC': 'Liverpool',
            'Chelsea FC': 'Chelsea',
            'Arsenal FC': 'Arsenal',
            'Tottenham Hotspur': 'Tottenham'
        }
        
        for col in ['home_team', 'away_team']:
            if col in df.columns:
                df[col] = df[col].replace(name_mappings)
                
        return df
    
    def join_all_data(self, matches_df: pd.DataFrame, fixtures_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Join historical matches with upcoming fixtures and additional data.
        
        Args:
            matches_df: Historical match data
            fixtures_df: Upcoming fixtures (optional)
            
        Returns:
            Combined dataset
        """
        all_data = matches_df.copy()
        
        if fixtures_df is not None and not fixtures_df.empty:
            # Add placeholder columns for fixtures
            fixture_cols = set(matches_df.columns) - set(fixtures_df.columns)
            for col in fixture_cols:
                if col not in ['home_goals', 'away_goals', 'result']:
                    fixtures_df[col] = np.nan
                    
            # Combine
            all_data = pd.concat([matches_df, fixtures_df], ignore_index=True)
            all_data = all_data.sort_values('date').reset_index(drop=True)
            
        logger.info(f"Combined dataset has {len(all_data)} total records")
        return all_data


def load_match_data(paths: List[str], config: Dict) -> pd.DataFrame:
    """
    Main function to load match data from specified paths.
    
    Args:
        paths: List of file paths or database paths
        config: Configuration dictionary
        
    Returns:
        Combined match DataFrame
    """
    ingester = DataIngester(config)
    all_matches = []
    
    for path in paths:
        if path.endswith('.sqlite'):
            df = ingester.load_kaggle_data(path)
        elif path.endswith('.csv'):
            df = pd.read_csv(path)
            df = ingester._clean_match_data(df)
        else:
            logger.warning(f"Unknown file format: {path}")
            continue
            
        all_matches.append(df)
    
    if all_matches:
        combined = pd.concat(all_matches, ignore_index=True)
        combined = combined.drop_duplicates(
            subset=['date', 'home_team', 'away_team'], 
            keep='last'
        ).reset_index(drop=True)
        
        return combined
    
    return pd.DataFrame()


def load_odds_data(paths: List[str]) -> pd.DataFrame:
    """
    Load odds data from CSV files.
    
    Args:
        paths: List of CSV file paths with odds data
        
    Returns:
        DataFrame with odds data
    """
    all_odds = []
    
    for path in paths:
        try:
            df = pd.read_csv(path)
            df['date'] = pd.to_datetime(df['date'])
            all_odds.append(df)
        except Exception as e:
            logger.error(f"Error loading odds from {path}: {e}")
            
    if all_odds:
        return pd.concat(all_odds, ignore_index=True)
    
    return pd.DataFrame()


if __name__ == "__main__":
    # Example usage
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train.yaml')
    parser.add_argument('--source', choices=['kaggle', 'api'], default='kaggle')
    parser.add_argument('--output', default='data/raw/matches.csv')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.source == 'kaggle':
        # Assuming Kaggle database is downloaded
        db_path = "data/raw/european_soccer_database.sqlite"
        ingester = DataIngester(config)
        df = ingester.load_kaggle_data(db_path)
        df.to_csv(args.output, index=False)
        print(f"Saved {len(df)} matches to {args.output}")
    
    elif args.source == 'api':
        # Load fixtures from API (requires API key)
        api_key = input("Enter football-data.org API key: ")
        ingester = DataIngester(config)
        df = ingester.load_api_fixtures(api_key)
        df.to_csv(args.output.replace('.csv', '_fixtures.csv'), index=False)
        print(f"Saved {len(df)} fixtures to {args.output}")
