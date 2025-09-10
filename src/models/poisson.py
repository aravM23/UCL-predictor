"""
Model B: Bivariate Poisson with Dixon-Coles correction for scoreline prediction.
Implements the Dixon-Coles model for realistic football score distributions.
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson
import pickle
from typing import Dict, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class DixonColesModel:
    """Dixon-Coles model for football score prediction."""
    
    def __init__(self, max_goals: int = 6, dc_correction: bool = True):
        self.max_goals = max_goals
        self.dc_correction = dc_correction
        self.teams = []
        self.team_to_idx = {}
        self.idx_to_team = {}
        self.params = None
        self.fitted = False
        
    def _tau(self, x: int, y: int, lambda_home: float, lambda_away: float, rho: float) -> float:
        """
        Dixon-Coles tau correction for low scores.
        Adjusts for the dependency between home and away goals for low-scoring games.
        """
        if not self.dc_correction:
            return 1.0
            
        if x == 0 and y == 0:
            return 1 - lambda_home * lambda_away * rho
        elif x == 0 and y == 1:
            return 1 + lambda_home * rho
        elif x == 1 and y == 0:
            return 1 + lambda_away * rho
        elif x == 1 and y == 1:
            return 1 - rho
        else:
            return 1.0
    
    def _log_likelihood(self, params: np.ndarray, matches: pd.DataFrame) -> float:
        """
        Calculate negative log-likelihood for parameter optimization.
        
        Args:
            params: Parameter vector [attack_strengths, defense_strengths, home_advantage, rho]
            matches: DataFrame with match results
            
        Returns:
            Negative log-likelihood
        """
        n_teams = len(self.teams)
        
        # Unpack parameters
        attack = params[:n_teams]
        defense = params[n_teams:2*n_teams]
        home_advantage = params[2*n_teams]
        rho = params[2*n_teams + 1] if self.dc_correction else 0.0
        
        log_lik = 0.0
        
        for _, match in matches.iterrows():
            home_idx = self.team_to_idx[match['home_team']]
            away_idx = self.team_to_idx[match['away_team']]
            
            # Expected goals
            lambda_home = np.exp(attack[home_idx] - defense[away_idx] + home_advantage)
            lambda_away = np.exp(attack[away_idx] - defense[home_idx])
            
            home_goals = int(match['home_goals'])
            away_goals = int(match['away_goals'])
            
            # Poisson probabilities
            prob_home = poisson.pmf(home_goals, lambda_home)
            prob_away = poisson.pmf(away_goals, lambda_away)
            
            # Dixon-Coles correction
            tau_correction = self._tau(home_goals, away_goals, lambda_home, lambda_away, rho)
            
            # Joint probability
            joint_prob = prob_home * prob_away * tau_correction
            
            # Add to log-likelihood (with small epsilon to avoid log(0))
            log_lik += np.log(max(joint_prob, 1e-10))
        
        return -log_lik  # Return negative for minimization
    
    def fit(self, matches: pd.DataFrame, regularization: Optional[Dict] = None) -> Dict:
        """
        Fit the Dixon-Coles model to match data.
        
        Args:
            matches: DataFrame with columns [home_team, away_team, home_goals, away_goals]
            regularization: Optional regularization parameters
            
        Returns:
            Dictionary with optimization results
        """
        logger.info("Fitting Dixon-Coles model...")
        
        # Get unique teams
        self.teams = sorted(list(set(matches['home_team'].unique()) | set(matches['away_team'].unique())))
        self.team_to_idx = {team: i for i, team in enumerate(self.teams)}
        self.idx_to_team = {i: team for i, team in enumerate(self.teams)}
        
        n_teams = len(self.teams)
        logger.info(f"Fitting model for {n_teams} teams on {len(matches)} matches")
        
        # Initialize parameters
        # Attack and defense strengths (mean-centered)
        attack_init = np.random.normal(0, 0.1, n_teams)
        defense_init = np.random.normal(0, 0.1, n_teams)
        home_advantage_init = 0.3  # Typical home advantage
        rho_init = 0.0 if not self.dc_correction else -0.1  # Small negative correlation
        
        # Combine parameters
        if self.dc_correction:
            params_init = np.concatenate([attack_init, defense_init, [home_advantage_init, rho_init]])
        else:
            params_init = np.concatenate([attack_init, defense_init, [home_advantage_init]])
        
        # Set up constraints (sum of attack = sum of defense = 0 for identifiability)
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x[:n_teams])},  # sum of attacks = 0
            {'type': 'eq', 'fun': lambda x: np.sum(x[n_teams:2*n_teams])}  # sum of defenses = 0
        ]
        
        # Bounds (reasonable ranges for parameters)
        bounds = []
        # Attack strengths: [-2, 2]
        bounds.extend([(-2, 2) for _ in range(n_teams)])
        # Defense strengths: [-2, 2]  
        bounds.extend([(-2, 2) for _ in range(n_teams)])
        # Home advantage: [0, 1]
        bounds.append((0, 1))
        # Rho: [-0.5, 0.5]
        if self.dc_correction:
            bounds.append((-0.5, 0.5))
        
        # Add regularization to objective if specified
        def regularized_objective(params):
            obj = self._log_likelihood(params, matches)
            
            if regularization:
                # L2 regularization on attack/defense strengths
                attack_reg = regularization.get('attack_strength', 0.0)
                defense_reg = regularization.get('defense_strength', 0.0)
                
                if attack_reg > 0:
                    obj += attack_reg * np.sum(params[:n_teams]**2)
                if defense_reg > 0:
                    obj += defense_reg * np.sum(params[n_teams:2*n_teams]**2)
            
            return obj
        
        # Optimize
        result = minimize(
            regularized_objective,
            params_init,
            method='L-BFGS-B',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        
        if not result.success:
            logger.warning(f"Optimization did not converge: {result.message}")
        
        # Store parameters
        self.params = {
            'attack': result.x[:n_teams],
            'defense': result.x[n_teams:2*n_teams], 
            'home_advantage': result.x[2*n_teams],
            'rho': result.x[2*n_teams + 1] if self.dc_correction else 0.0
        }
        
        self.fitted = True
        
        logger.info(f"Model fitted successfully. Log-likelihood: {-result.fun:.2f}")
        logger.info(f"Home advantage: {self.params['home_advantage']:.3f}")
        if self.dc_correction:
            logger.info(f"Rho parameter: {self.params['rho']:.3f}")
        
        return {
            'success': result.success,
            'log_likelihood': -result.fun,
            'n_iterations': result.nit,
            'params': self.params
        }
    
    def predict_match(self, home_team: str, away_team: str) -> np.ndarray:
        """
        Predict score matrix for a single match.
        
        Args:
            home_team: Name of home team
            away_team: Name of away team
            
        Returns:
            Score matrix (max_goals+1, max_goals+1) with probabilities
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if home_team not in self.team_to_idx or away_team not in self.team_to_idx:
            raise ValueError(f"Unknown team(s): {home_team}, {away_team}")
        
        home_idx = self.team_to_idx[home_team]
        away_idx = self.team_to_idx[away_team]
        
        # Expected goals
        lambda_home = np.exp(
            self.params['attack'][home_idx] - 
            self.params['defense'][away_idx] + 
            self.params['home_advantage']
        )
        lambda_away = np.exp(
            self.params['attack'][away_idx] - 
            self.params['defense'][home_idx]
        )
        
        # Build score matrix
        score_matrix = np.zeros((self.max_goals + 1, self.max_goals + 1))
        
        for home_goals in range(self.max_goals + 1):
            for away_goals in range(self.max_goals + 1):
                # Basic Poisson probabilities
                prob_home = poisson.pmf(home_goals, lambda_home)
                prob_away = poisson.pmf(away_goals, lambda_away)
                
                # Dixon-Coles correction
                tau_correction = self._tau(home_goals, away_goals, lambda_home, lambda_away, self.params['rho'])
                
                score_matrix[home_goals, away_goals] = prob_home * prob_away * tau_correction
        
        # Normalize to ensure probabilities sum to 1
        score_matrix = score_matrix / score_matrix.sum()
        
        return score_matrix
    
    def predict_wdl(self, home_team: str, away_team: str) -> Tuple[float, float, float]:
        """
        Predict W/D/L probabilities from score matrix.
        
        Args:
            home_team: Name of home team
            away_team: Name of away team
            
        Returns:
            Tuple of (P_home_win, P_draw, P_away_win)
        """
        score_matrix = self.predict_match(home_team, away_team)
        
        # Sum probabilities for each outcome
        p_home_win = np.sum(np.triu(score_matrix, k=1))  # Home goals > Away goals
        p_draw = np.sum(np.diag(score_matrix))  # Home goals = Away goals
        p_away_win = np.sum(np.tril(score_matrix, k=-1))  # Home goals < Away goals
        
        return p_home_win, p_draw, p_away_win
    
    def get_team_strengths(self) -> pd.DataFrame:
        """Get team attack and defense strengths."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        df = pd.DataFrame({
            'team': self.teams,
            'attack': self.params['attack'],
            'defense': self.params['defense']
        })
        
        # Add overall strength (attack - defense)
        df['strength'] = df['attack'] - df['defense']
        
        return df.sort_values('strength', ascending=False)
    
    def get_most_likely_scores(self, home_team: str, away_team: str, top_n: int = 10) -> pd.DataFrame:
        """
        Get most likely scorelines for a match.
        
        Args:
            home_team: Name of home team
            away_team: Name of away team
            top_n: Number of top scorelines to return
            
        Returns:
            DataFrame with scorelines and probabilities
        """
        score_matrix = self.predict_match(home_team, away_team)
        
        # Find top scorelines
        scorelines = []
        for i in range(self.max_goals + 1):
            for j in range(self.max_goals + 1):
                scorelines.append({
                    'home_goals': i,
                    'away_goals': j,
                    'probability': score_matrix[i, j]
                })
        
        df = pd.DataFrame(scorelines)
        df = df.sort_values('probability', ascending=False).head(top_n)
        df['scoreline'] = df['home_goals'].astype(str) + '-' + df['away_goals'].astype(str)
        
        return df[['scoreline', 'home_goals', 'away_goals', 'probability']].reset_index(drop=True)
    
    def save_model(self, path: str):
        """Save fitted model to disk."""
        if not self.fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'max_goals': self.max_goals,
            'dc_correction': self.dc_correction,
            'teams': self.teams,
            'team_to_idx': self.team_to_idx,
            'idx_to_team': self.idx_to_team,
            'params': self.params,
            'fitted': self.fitted
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load fitted model from disk."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.max_goals = model_data['max_goals']
        self.dc_correction = model_data['dc_correction']
        self.teams = model_data['teams']
        self.team_to_idx = model_data['team_to_idx']
        self.idx_to_team = model_data['idx_to_team']
        self.params = model_data['params']
        self.fitted = model_data['fitted']
        
        logger.info(f"Model loaded from {path}")


def fit_dixon_coles(matches: pd.DataFrame, config: Dict) -> DixonColesModel:
    """
    Main function to fit Dixon-Coles model.
    
    Args:
        matches: DataFrame with match results
        config: Configuration dictionary
        
    Returns:
        Fitted DixonColesModel
    """
    model_config = config.get('modelB', {})
    
    model = DixonColesModel(
        max_goals=model_config.get('max_goals', 6),
        dc_correction=model_config.get('dc_correlation', True)
    )
    
    # Fit model
    regularization = model_config.get('regularization', {})
    result = model.fit(matches, regularization if regularization else None)
    
    # Print team strengths
    strengths = model.get_team_strengths()
    logger.info("Top 10 strongest teams:")
    logger.info(strengths.head(10).to_string(index=False))
    
    return model


if __name__ == "__main__":
    import argparse
    import yaml
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description='Train Dixon-Coles model')
    parser.add_argument('--data', required=True, help='Match data CSV file')
    parser.add_argument('--config', default='configs/train.yaml', help='Config file')
    parser.add_argument('--output', default='models/dixon_coles_model.pkl', help='Output model file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load match data
    matches = pd.read_csv(args.data)
    matches = matches.dropna(subset=['home_goals', 'away_goals'])
    
    # Fit model
    model = fit_dixon_coles(matches, config)
    
    # Save model
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(args.output)
    
    logger.info(f"Dixon-Coles model saved to {args.output}")
    
    # Example prediction
    if len(model.teams) >= 2:
        home_team = model.teams[0]
        away_team = model.teams[1]
        
        print(f"\nExample prediction: {home_team} vs {away_team}")
        p_h, p_d, p_a = model.predict_wdl(home_team, away_team)
        print(f"W/D/L probabilities: {p_h:.3f} / {p_d:.3f} / {p_a:.3f}")
        
        top_scores = model.get_most_likely_scores(home_team, away_team, top_n=5)
        print("Most likely scorelines:")
        print(top_scores.to_string(index=False))
