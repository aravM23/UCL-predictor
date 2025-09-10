"""
Market analysis and edge detection module.
Handles bookmaker odds conversion and value betting identification.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def implied_probs_from_odds(odds_home: float, odds_draw: float, odds_away: float) -> Tuple[float, float, float]:
    """
    Convert decimal odds to implied probabilities and remove bookmaker margin.
    
    Args:
        odds_home: Decimal odds for home win
        odds_draw: Decimal odds for draw
        odds_away: Decimal odds for away win
        
    Returns:
        Tuple of normalized implied probabilities (home, draw, away)
    """
    if any(odd <= 1.0 for odd in [odds_home, odds_draw, odds_away]):
        raise ValueError("All odds must be greater than 1.0")
    
    # Convert to implied probabilities
    implied_home = 1 / odds_home
    implied_draw = 1 / odds_draw
    implied_away = 1 / odds_away
    
    # Total probability (includes bookmaker margin)
    total_prob = implied_home + implied_draw + implied_away
    
    # Remove margin by normalizing
    normalized_home = implied_home / total_prob
    normalized_draw = implied_draw / total_prob
    normalized_away = implied_away / total_prob
    
    return normalized_home, normalized_draw, normalized_away


def calculate_bookmaker_margin(odds_home: float, odds_draw: float, odds_away: float) -> float:
    """Calculate bookmaker margin (overround) from odds."""
    total_implied = (1/odds_home) + (1/odds_draw) + (1/odds_away)
    margin = (total_implied - 1) * 100  # As percentage
    return margin


def calculate_fair_odds(prob_home: float, prob_draw: float, prob_away: float) -> Tuple[float, float, float]:
    """
    Convert model probabilities to fair decimal odds.
    
    Args:
        prob_home: Model probability of home win
        prob_draw: Model probability of draw
        prob_away: Model probability of away win
        
    Returns:
        Tuple of fair decimal odds (home, draw, away)
    """
    # Ensure probabilities sum to 1
    total_prob = prob_home + prob_draw + prob_away
    prob_home /= total_prob
    prob_draw /= total_prob
    prob_away /= total_prob
    
    # Convert to odds (with small epsilon to avoid division by zero)
    odds_home = 1 / max(prob_home, 1e-6)
    odds_draw = 1 / max(prob_draw, 1e-6)
    odds_away = 1 / max(prob_away, 1e-6)
    
    return odds_home, odds_draw, odds_away


class MarketAnalyzer:
    """Analyzes betting markets and identifies value opportunities."""
    
    def __init__(self, min_edge: float = 0.02, min_odds: float = 1.1, max_odds: float = 10.0):
        self.min_edge = min_edge
        self.min_odds = min_odds
        self.max_odds = max_odds
        
    def analyze_match(self, model_probs: Tuple[float, float, float], 
                     market_odds: Tuple[float, float, float],
                     home_team: str, away_team: str, date: str = None) -> Dict:
        """
        Analyze a single match for betting value.
        
        Args:
            model_probs: Model probabilities (home, draw, away)
            market_odds: Market odds (home, draw, away)
            home_team: Home team name
            away_team: Away team name
            date: Match date
            
        Returns:
            Dictionary with analysis results
        """
        prob_home, prob_draw, prob_away = model_probs
        odds_home, odds_draw, odds_away = market_odds
        
        # Market implied probabilities (normalized)
        market_probs = implied_probs_from_odds(odds_home, odds_draw, odds_away)
        market_prob_home, market_prob_draw, market_prob_away = market_probs
        
        # Calculate edges (model_prob - market_prob)
        edge_home = prob_home - market_prob_home
        edge_draw = prob_draw - market_prob_draw
        edge_away = prob_away - market_prob_away
        
        # Fair odds from model
        fair_odds = calculate_fair_odds(prob_home, prob_draw, prob_away)
        fair_odds_home, fair_odds_draw, fair_odds_away = fair_odds
        
        # Value calculations (expected return per unit bet)
        value_home = (odds_home * prob_home) - 1
        value_draw = (odds_draw * prob_draw) - 1
        value_away = (odds_away * prob_away) - 1
        
        # Bookmaker margin
        margin = calculate_bookmaker_margin(odds_home, odds_draw, odds_away)
        
        # Identify value bets
        value_bets = []
        
        outcomes = [
            ('home', edge_home, value_home, odds_home, prob_home),
            ('draw', edge_draw, value_draw, odds_draw, prob_draw),
            ('away', edge_away, value_away, odds_away, prob_away)
        ]
        
        for outcome, edge, value, odds, prob in outcomes:
            if (edge >= self.min_edge and 
                value > 0 and 
                self.min_odds <= odds <= self.max_odds):
                
                value_bets.append({
                    'outcome': outcome,
                    'edge': edge,
                    'value': value,
                    'odds': odds,
                    'model_prob': prob,
                    'market_prob': market_probs[['home', 'draw', 'away'].index(outcome)]
                })
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'date': date,
            'model_probs': {
                'home': prob_home,
                'draw': prob_draw,
                'away': prob_away
            },
            'market_probs': {
                'home': market_prob_home,
                'draw': market_prob_draw,
                'away': market_prob_away
            },
            'market_odds': {
                'home': odds_home,
                'draw': odds_draw,
                'away': odds_away
            },
            'fair_odds': {
                'home': fair_odds_home,
                'draw': fair_odds_draw,
                'away': fair_odds_away
            },
            'edges': {
                'home': edge_home,
                'draw': edge_draw,
                'away': edge_away
            },
            'values': {
                'home': value_home,
                'draw': value_draw,
                'away': value_away
            },
            'bookmaker_margin': margin,
            'value_bets': value_bets,
            'max_edge': max(edge_home, edge_draw, edge_away),
            'max_value': max(value_home, value_draw, value_away)
        }
    
    def analyze_multiple_matches(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze multiple matches for value betting opportunities.
        
        Args:
            predictions_df: DataFrame with columns:
                - home_team, away_team, date
                - prob_home, prob_draw, prob_away (model probabilities)
                - odds_home, odds_draw, odds_away (market odds)
                
        Returns:
            DataFrame with analysis results
        """
        results = []
        
        for _, row in predictions_df.iterrows():
            model_probs = (row['prob_home'], row['prob_draw'], row['prob_away'])
            market_odds = (row['odds_home'], row['odds_draw'], row['odds_away'])
            
            analysis = self.analyze_match(
                model_probs, market_odds, 
                row['home_team'], row['away_team'], 
                row.get('date')
            )
            
            results.append(analysis)
        
        # Convert to DataFrame
        df_results = pd.DataFrame(results)
        
        # Add summary columns
        df_results['has_value_bets'] = df_results['value_bets'].apply(len) > 0
        df_results['num_value_bets'] = df_results['value_bets'].apply(len)
        
        return df_results
    
    def get_top_value_bets(self, analysis_df: pd.DataFrame, top_n: int = 10, 
                          sort_by: str = 'max_edge') -> pd.DataFrame:
        """
        Get top value betting opportunities.
        
        Args:
            analysis_df: Output from analyze_multiple_matches
            top_n: Number of top bets to return
            sort_by: Sort criteria ('max_edge', 'max_value')
            
        Returns:
            DataFrame with top value bets
        """
        # Filter matches with value bets
        value_matches = analysis_df[analysis_df['has_value_bets']].copy()
        
        if value_matches.empty:
            logger.warning("No value bets found")
            return pd.DataFrame()
        
        # Sort by criteria
        value_matches = value_matches.sort_values(sort_by, ascending=False)
        
        # Expand value bets
        expanded_bets = []
        for _, match in value_matches.iterrows():
            for bet in match['value_bets']:
                expanded_bets.append({
                    'home_team': match['home_team'],
                    'away_team': match['away_team'],
                    'date': match['date'],
                    'outcome': bet['outcome'],
                    'edge': bet['edge'],
                    'value': bet['value'],
                    'odds': bet['odds'],
                    'model_prob': bet['model_prob'],
                    'market_prob': bet['market_prob'],
                    'bookmaker_margin': match['bookmaker_margin']
                })
        
        df_bets = pd.DataFrame(expanded_bets)
        
        # Sort and return top bets
        sort_col = 'edge' if sort_by == 'max_edge' else 'value'
        df_bets = df_bets.sort_values(sort_col, ascending=False).head(top_n)
        
        return df_bets.reset_index(drop=True)


def detect_arbitrage_opportunities(odds_data: pd.DataFrame, min_profit: float = 0.01) -> pd.DataFrame:
    """
    Detect arbitrage opportunities across different bookmakers.
    
    Args:
        odds_data: DataFrame with columns for different bookmaker odds
        min_profit: Minimum profit percentage for arbitrage
        
    Returns:
        DataFrame with arbitrage opportunities
    """
    arbitrages = []
    
    for _, match in odds_data.iterrows():
        # Find best odds for each outcome across bookmakers
        home_odds_cols = [col for col in match.index if 'H' in col and col != 'home_team']
        draw_odds_cols = [col for col in match.index if 'D' in col]
        away_odds_cols = [col for col in match.index if 'A' in col and col != 'away_team']
        
        if not (home_odds_cols and draw_odds_cols and away_odds_cols):
            continue
            
        best_home_odds = match[home_odds_cols].max()
        best_draw_odds = match[draw_odds_cols].max()
        best_away_odds = match[away_odds_cols].max()
        
        # Check for arbitrage
        implied_total = (1/best_home_odds) + (1/best_draw_odds) + (1/best_away_odds)
        
        if implied_total < (1 - min_profit):  # Arbitrage opportunity
            profit = (1 - implied_total) * 100  # Profit percentage
            
            arbitrages.append({
                'home_team': match.get('home_team', ''),
                'away_team': match.get('away_team', ''),
                'date': match.get('date', ''),
                'best_home_odds': best_home_odds,
                'best_draw_odds': best_draw_odds,
                'best_away_odds': best_away_odds,
                'implied_total': implied_total,
                'profit_pct': profit
            })
    
    return pd.DataFrame(arbitrages)


if __name__ == "__main__":
    # Example usage
    analyzer = MarketAnalyzer(min_edge=0.02)
    
    # Example match analysis
    model_probs = (0.45, 0.30, 0.25)  # Home, Draw, Away
    market_odds = (2.20, 3.40, 3.10)  # Home, Draw, Away
    
    analysis = analyzer.analyze_match(
        model_probs, market_odds,
        "Real Madrid", "Manchester City"
    )
    
    print("Market Analysis Example:")
    print(f"Model probabilities: {analysis['model_probs']}")
    print(f"Market probabilities: {analysis['market_probs']}")
    print(f"Edges: {analysis['edges']}")
    print(f"Values: {analysis['values']}")
    print(f"Bookmaker margin: {analysis['bookmaker_margin']:.1f}%")
    print(f"Value bets found: {len(analysis['value_bets'])}")
    
    for bet in analysis['value_bets']:
        print(f"  {bet['outcome']}: Edge={bet['edge']:.3f}, Value={bet['value']:.3f}, Odds={bet['odds']:.2f}")
