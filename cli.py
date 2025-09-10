#!/usr/bin/env python3
"""
Command-line interface for the Champions League predictor.
Provides batch prediction and analysis capabilities.
"""

import argparse
import pandas as pd
import numpy as np
import yaml
import pickle
import json
from datetime import datetime
from pathlib import Path
import sys
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from models.train_wdl import WDLTrainer
    from models.poisson import DixonColesModel
    from bet.market import MarketAnalyzer, implied_probs_from_odds
    from bet.kelly import KellyCalculator
    from data.features import FeatureBuilder
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure to install requirements and run from project root")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CLIPredictor:
    """Command-line interface for match predictions."""
    
    def __init__(self, config_path: str = "configs/train.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        self.wdl_model = None
        self.dc_model = None
        self.feature_builder = None
        self.market_analyzer = MarketAnalyzer()
        self.kelly_calculator = KellyCalculator()
        
    def load_config(self) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found, using defaults")
            return {}
    
    def load_models(self):
        """Load trained models."""
        try:
            # Load W/D/L model
            wdl_path = Path("models/wdl_model.pkl")
            if wdl_path.exists():
                self.wdl_model = WDLTrainer(self.config)
                self.wdl_model.load_model(str(wdl_path))
                logger.info("Loaded W/D/L model")
            else:
                logger.warning("W/D/L model not found")
            
            # Load Dixon-Coles model
            dc_path = Path("models/dixon_coles_model.pkl")
            if dc_path.exists():
                self.dc_model = DixonColesModel()
                self.dc_model.load_model(str(dc_path))
                logger.info("Loaded Dixon-Coles model")
            else:
                logger.warning("Dixon-Coles model not found")
            
            # Initialize feature builder
            if self.config:
                self.feature_builder = FeatureBuilder(self.config)
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def predict_single_match(self, home_team: str, away_team: str, date: str = None,
                           odds_home: float = None, odds_draw: float = None, 
                           odds_away: float = None) -> dict:
        """
        Predict a single match.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            date: Match date (YYYY-MM-DD)
            odds_home: Home win odds (optional)
            odds_draw: Draw odds (optional)
            odds_away: Away win odds (optional)
            
        Returns:
            Dictionary with predictions
        """
        if not self.dc_model or not self.dc_model.fitted:
            raise ValueError("Dixon-Coles model not available")
        
        if home_team not in self.dc_model.teams or away_team not in self.dc_model.teams:
            raise ValueError(f"Teams not in model: {home_team}, {away_team}")
        
        # Get predictions from Dixon-Coles model
        p_home, p_draw, p_away = self.dc_model.predict_wdl(home_team, away_team)
        score_matrix = self.dc_model.predict_match(home_team, away_team)
        top_scorelines = self.dc_model.get_most_likely_scores(home_team, away_team, top_n=10)
        
        # Get team strengths
        strengths = self.dc_model.get_team_strengths()
        home_strength = strengths[strengths['team'] == home_team]['strength'].iloc[0]
        away_strength = strengths[strengths['team'] == away_team]['strength'].iloc[0]
        
        predictions = {
            'match': {
                'home_team': home_team,
                'away_team': away_team,
                'date': date or datetime.now().strftime('%Y-%m-%d')
            },
            'probabilities': {
                'home_win': float(p_home),
                'draw': float(p_draw),
                'away_win': float(p_away)
            },
            'team_strengths': {
                'home_strength': float(home_strength),
                'away_strength': float(away_strength),
                'strength_diff': float(home_strength - away_strength)
            },
            'top_scorelines': top_scorelines.to_dict('records'),
            'fair_odds': {
                'home': float(1 / p_home),
                'draw': float(1 / p_draw),
                'away': float(1 / p_away)
            }
        }
        
        # Market analysis if odds provided
        if all(odds is not None for odds in [odds_home, odds_draw, odds_away]):
            analysis = self.market_analyzer.analyze_match(
                (p_home, p_draw, p_away),
                (odds_home, odds_draw, odds_away),
                home_team, away_team, date
            )
            
            predictions['market_analysis'] = {
                'market_odds': analysis['market_odds'],
                'edges': analysis['edges'],
                'values': analysis['values'],
                'bookmaker_margin': analysis['bookmaker_margin'],
                'value_bets': analysis['value_bets']
            }
            
            # Kelly recommendations
            if analysis['value_bets']:
                kelly_recommendations = []
                for bet in analysis['value_bets']:
                    kelly_calc = self.kelly_calculator.calculate_stake(
                        bet['model_prob'], bet['odds'], bankroll=1000
                    )
                    kelly_recommendations.append({
                        'outcome': bet['outcome'],
                        'kelly_fraction': kelly_calc['kelly_fraction'],
                        'stake_percentage': kelly_calc['stake_percentage'],
                        'expected_return': kelly_calc['expected_return']
                    })
                
                predictions['kelly_recommendations'] = kelly_recommendations
        
        return predictions
    
    def predict_batch(self, input_file: str, output_file: str = None):
        """
        Predict multiple matches from CSV file.
        
        Args:
            input_file: Path to CSV with match data
            output_file: Path to save predictions (optional)
        """
        df = pd.read_csv(input_file)
        required_cols = ['home_team', 'away_team']
        
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Input file must contain columns: {required_cols}")
        
        predictions = []
        
        for _, row in df.iterrows():
            try:
                pred = self.predict_single_match(
                    row['home_team'],
                    row['away_team'],
                    row.get('date'),
                    row.get('odds_home'),
                    row.get('odds_draw'),
                    row.get('odds_away')
                )
                predictions.append(pred)
            except Exception as e:
                logger.error(f"Error predicting {row['home_team']} vs {row['away_team']}: {e}")
                continue
        
        # Save predictions
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(predictions, f, indent=2)
            logger.info(f"Saved {len(predictions)} predictions to {output_file}")
        
        return predictions
    
    def analyze_portfolio(self, predictions: list, bankroll: float = 1000):
        """Analyze betting portfolio from predictions."""
        value_bets = []
        
        for pred in predictions:
            if 'market_analysis' in pred and pred['market_analysis']['value_bets']:
                for bet in pred['market_analysis']['value_bets']:
                    bet_info = {
                        'home_team': pred['match']['home_team'],
                        'away_team': pred['match']['away_team'],
                        'outcome': bet['outcome'],
                        'probability': bet['model_prob'],
                        'odds': bet['odds'],
                        'edge': bet['edge']
                    }
                    value_bets.append(bet_info)
        
        if not value_bets:
            return {'message': 'No value bets found'}
        
        # Calculate portfolio
        portfolio = self.kelly_calculator.optimize_portfolio(
            value_bets, bankroll=bankroll
        )
        
        return portfolio
    
    def backtest(self, results_file: str, start_date: str = None, end_date: str = None):
        """
        Backtest model performance on historical results.
        
        Args:
            results_file: CSV with historical match results
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
        """
        df = pd.read_csv(results_file)
        df['date'] = pd.to_datetime(df['date'])
        
        if start_date:
            df = df[df['date'] >= start_date]
        if end_date:
            df = df[df['date'] <= end_date]
        
        correct_predictions = 0
        total_predictions = 0
        brier_scores = []
        
        for _, row in df.iterrows():
            try:
                pred = self.predict_single_match(
                    row['home_team'], 
                    row['away_team'], 
                    row['date'].strftime('%Y-%m-%d')
                )
                
                # Check accuracy
                actual_result = row['result']  # 'H', 'D', 'A'
                predicted_result = max(pred['probabilities'], key=pred['probabilities'].get)
                
                result_map = {'home_win': 'H', 'draw': 'D', 'away_win': 'A'}
                if result_map[predicted_result] == actual_result:
                    correct_predictions += 1
                
                total_predictions += 1
                
                # Calculate Brier score
                probs = [pred['probabilities']['home_win'], 
                        pred['probabilities']['draw'], 
                        pred['probabilities']['away_win']]
                actual_vec = [1 if actual_result == r else 0 for r in ['H', 'D', 'A']]
                brier = sum((p - a)**2 for p, a in zip(probs, actual_vec))
                brier_scores.append(brier)
                
            except Exception as e:
                logger.error(f"Error in backtest for {row['home_team']} vs {row['away_team']}: {e}")
                continue
        
        accuracy = correct_predictions / max(total_predictions, 1)
        avg_brier = np.mean(brier_scores) if brier_scores else 0
        
        backtest_results = {
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions,
            'average_brier_score': avg_brier,
            'date_range': {
                'start': start_date,
                'end': end_date
            }
        }
        
        return backtest_results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description='Champions League Predictor CLI')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict match outcome')
    predict_parser.add_argument('--home', required=True, help='Home team name')
    predict_parser.add_argument('--away', required=True, help='Away team name')
    predict_parser.add_argument('--date', help='Match date (YYYY-MM-DD)')
    predict_parser.add_argument('--odds-home', type=float, help='Home win odds')
    predict_parser.add_argument('--odds-draw', type=float, help='Draw odds')
    predict_parser.add_argument('--odds-away', type=float, help='Away win odds')
    predict_parser.add_argument('--output', help='Output file for predictions')
    
    # Batch predict command
    batch_parser = subparsers.add_parser('batch', help='Predict multiple matches')
    batch_parser.add_argument('--input', required=True, help='Input CSV file')
    batch_parser.add_argument('--output', help='Output JSON file')
    
    # Portfolio command
    portfolio_parser = subparsers.add_parser('portfolio', help='Analyze betting portfolio')
    portfolio_parser.add_argument('--predictions', required=True, help='Predictions JSON file')
    portfolio_parser.add_argument('--bankroll', type=float, default=1000, help='Bankroll amount')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Backtest model performance')
    backtest_parser.add_argument('--results', required=True, help='Historical results CSV')
    backtest_parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    backtest_parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    
    # Global arguments
    parser.add_argument('--config', default='configs/train.yaml', help='Config file path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize predictor
    predictor = CLIPredictor(args.config)
    predictor.load_models()
    
    try:
        if args.command == 'predict':
            prediction = predictor.predict_single_match(
                args.home, args.away, args.date,
                args.odds_home, args.odds_draw, args.odds_away
            )
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(prediction, f, indent=2)
                print(f"Prediction saved to {args.output}")
            else:
                print(json.dumps(prediction, indent=2))
        
        elif args.command == 'batch':
            predictions = predictor.predict_batch(args.input, args.output)
            print(f"Processed {len(predictions)} matches")
        
        elif args.command == 'portfolio':
            with open(args.predictions, 'r') as f:
                predictions = json.load(f)
            
            portfolio = predictor.analyze_portfolio(predictions, args.bankroll)
            print(json.dumps(portfolio, indent=2))
        
        elif args.command == 'backtest':
            results = predictor.backtest(args.results, args.start_date, args.end_date)
            print(json.dumps(results, indent=2))
        
        else:
            parser.print_help()
    
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
