"""
Parlay optimization with Monte Carlo simulation for correlated outcomes.
Handles multi-leg bet optimization accounting for shared variance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from itertools import combinations
import logging

logger = logging.getLogger(__name__)


class ParlayLeg:
    """Represents a single leg of a parlay bet."""
    
    def __init__(self, home_team: str, away_team: str, outcome: str, 
                 odds: float, score_matrix: np.ndarray, date: str = None):
        self.home_team = home_team
        self.away_team = away_team
        self.outcome = outcome  # 'home', 'draw', 'away'
        self.odds = odds
        self.score_matrix = score_matrix  # From Dixon-Coles model
        self.date = date
        
    def simulate_outcome(self, rng: np.random.Generator) -> bool:
        """
        Simulate match outcome based on score matrix.
        
        Args:
            rng: Random number generator
            
        Returns:
            True if leg wins, False otherwise
        """
        # Sample scoreline from probability matrix
        flat_probs = self.score_matrix.flatten()
        flat_probs = flat_probs / flat_probs.sum()  # Normalize
        
        # Sample index
        sampled_idx = rng.choice(len(flat_probs), p=flat_probs)
        
        # Convert back to (home_goals, away_goals)
        max_goals = self.score_matrix.shape[0] - 1
        home_goals = sampled_idx // (max_goals + 1)
        away_goals = sampled_idx % (max_goals + 1)
        
        # Determine match result
        if home_goals > away_goals:
            match_result = 'home'
        elif home_goals < away_goals:
            match_result = 'away'
        else:
            match_result = 'draw'
        
        return match_result == self.outcome
    
    def get_independent_probability(self) -> float:
        """Calculate independent probability of this leg winning."""
        if self.outcome == 'home':
            return np.sum(np.triu(self.score_matrix, k=1))
        elif self.outcome == 'draw':
            return np.sum(np.diag(self.score_matrix))
        else:  # away
            return np.sum(np.tril(self.score_matrix, k=-1))


class ParlaySimulator:
    """Monte Carlo simulator for parlay betting with correlation."""
    
    def __init__(self, n_simulations: int = 20000, random_seed: int = None):
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(random_seed)
        
    def simulate_parlay(self, legs: List[ParlayLeg], correlation_factor: float = 0.1) -> Dict:
        """
        Simulate parlay outcome with correlation between legs.
        
        Args:
            legs: List of parlay legs
            correlation_factor: Factor controlling correlation between legs (0-1)
            
        Returns:
            Dictionary with simulation results
        """
        if not legs:
            return {'probability': 0, 'expected_odds': 0, 'simulations': 0}
        
        wins = 0
        total_odds = np.prod([leg.odds for leg in legs])
        
        # Store individual leg results for correlation analysis
        leg_results = np.zeros((self.n_simulations, len(legs)), dtype=bool)
        
        for sim in range(self.n_simulations):
            # Generate correlated random seeds for this simulation
            base_seed = self.rng.integers(0, 1000000)
            
            # Simulate each leg with potential correlation
            parlay_wins = True
            for i, leg in enumerate(legs):
                # Create correlated random generator
                leg_seed = base_seed + int(i * (1 - correlation_factor) * 1000)
                leg_rng = np.random.default_rng(leg_seed)
                
                leg_wins = leg.simulate_outcome(leg_rng)
                leg_results[sim, i] = leg_wins
                
                if not leg_wins:
                    parlay_wins = False
                    # In real parlay, we can break early, but we continue for correlation analysis
            
            if parlay_wins:
                wins += 1
        
        # Calculate correlation matrix between legs
        correlation_matrix = np.corrcoef(leg_results.T) if len(legs) > 1 else np.array([[1.0]])
        avg_correlation = np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]) if len(legs) > 1 else 0.0
        
        probability = wins / self.n_simulations
        expected_return = probability * total_odds - 1  # Expected return per unit bet
        
        # Independent probability (naive calculation)
        independent_prob = np.prod([leg.get_independent_probability() for leg in legs])
        
        return {
            'probability': probability,
            'independent_probability': independent_prob,
            'correlation_effect': probability - independent_prob,
            'expected_odds': 1 / probability if probability > 0 else float('inf'),
            'market_odds': total_odds,
            'expected_return': expected_return,
            'average_correlation': avg_correlation,
            'individual_probabilities': [leg.get_independent_probability() for leg in legs],
            'leg_results': leg_results,
            'simulations': self.n_simulations
        }
    
    def find_optimal_parlay_size(self, available_legs: List[ParlayLeg], 
                                max_legs: int = 5, min_probability: float = 0.1) -> Dict:
        """
        Find optimal parlay size by testing different combinations.
        
        Args:
            available_legs: Pool of available legs
            max_legs: Maximum number of legs to consider
            min_probability: Minimum acceptable probability
            
        Returns:
            Dictionary with optimal parlay recommendations
        """
        results = []
        
        for num_legs in range(2, min(len(available_legs) + 1, max_legs + 1)):
            # Test combinations of this size
            for leg_combo in combinations(available_legs, num_legs):
                simulation = self.simulate_parlay(list(leg_combo))
                
                if simulation['probability'] >= min_probability:
                    results.append({
                        'legs': leg_combo,
                        'num_legs': num_legs,
                        'probability': simulation['probability'],
                        'expected_return': simulation['expected_return'],
                        'market_odds': simulation['market_odds'],
                        'correlation_effect': simulation['correlation_effect'],
                        'leg_descriptions': [f"{leg.home_team} vs {leg.away_team} ({leg.outcome})" for leg in leg_combo]
                    })
        
        if not results:
            return {'optimal_parlays': [], 'message': 'No suitable parlays found'}
        
        # Sort by expected return
        results.sort(key=lambda x: x['expected_return'], reverse=True)
        
        return {
            'optimal_parlays': results[:10],  # Top 10
            'best_parlay': results[0] if results else None,
            'total_combinations_tested': len(results)
        }


class ParlayOptimizer:
    """Advanced parlay optimizer with Kelly criterion integration."""
    
    def __init__(self, simulator: ParlaySimulator = None, kelly_cap: float = 0.05):
        self.simulator = simulator or ParlaySimulator()
        self.kelly_cap = kelly_cap
        
    def optimize_parlay_portfolio(self, available_legs: List[ParlayLeg], 
                                 bankroll: float = 1000, max_exposure: float = 0.1,
                                 max_parlays: int = 5) -> Dict:
        """
        Optimize entire parlay portfolio for maximum expected utility.
        
        Args:
            available_legs: Available betting legs
            bankroll: Available bankroll
            max_exposure: Maximum total exposure as fraction of bankroll
            max_parlays: Maximum number of parlays to recommend
            
        Returns:
            Optimized portfolio of parlays
        """
        # Find all viable parlay combinations
        viable_parlays = []
        
        for num_legs in range(2, min(len(available_legs) + 1, 6)):  # Up to 5 legs
            for leg_combo in combinations(available_legs, num_legs):
                simulation = self.simulator.simulate_parlay(list(leg_combo))
                
                if simulation['expected_return'] > 0:  # Positive expected value
                    kelly_fraction = self._calculate_parlay_kelly(
                        simulation['probability'], 
                        simulation['market_odds']
                    )
                    
                    if kelly_fraction > 0:
                        viable_parlays.append({
                            'legs': leg_combo,
                            'probability': simulation['probability'],
                            'odds': simulation['market_odds'],
                            'expected_return': simulation['expected_return'],
                            'kelly_fraction': kelly_fraction,
                            'stake': bankroll * kelly_fraction,
                            'expected_profit': bankroll * kelly_fraction * simulation['expected_return'],
                            'correlation_effect': simulation['correlation_effect'],
                            'utility_score': self._calculate_utility(kelly_fraction, simulation['expected_return'])
                        })
        
        if not viable_parlays:
            return {'optimized_parlays': [], 'total_stake': 0, 'expected_profit': 0}
        
        # Sort by utility score
        viable_parlays.sort(key=lambda x: x['utility_score'], reverse=True)
        
        # Select parlays within exposure constraints
        selected_parlays = []
        total_stake = 0
        max_stake = bankroll * max_exposure
        
        for parlay in viable_parlays:
            if len(selected_parlays) >= max_parlays:
                break
                
            if total_stake + parlay['stake'] <= max_stake:
                selected_parlays.append(parlay)
                total_stake += parlay['stake']
        
        total_expected_profit = sum(p['expected_profit'] for p in selected_parlays)
        
        return {
            'optimized_parlays': selected_parlays,
            'total_stake': total_stake,
            'exposure_percentage': (total_stake / bankroll) * 100,
            'expected_profit': total_expected_profit,
            'expected_yield': (total_expected_profit / total_stake * 100) if total_stake > 0 else 0,
            'num_parlays': len(selected_parlays),
            'average_probability': np.mean([p['probability'] for p in selected_parlays]) if selected_parlays else 0
        }
    
    def _calculate_parlay_kelly(self, probability: float, odds: float) -> float:
        """Calculate Kelly fraction for parlay bet."""
        if probability <= 0 or odds <= 1:
            return 0.0
        
        b = odds - 1  # Net odds
        p = probability
        q = 1 - p
        
        kelly_f = (b * p - q) / b
        
        if kelly_f <= 0:
            return 0.0
        
        return min(kelly_f, self.kelly_cap)
    
    def _calculate_utility(self, kelly_fraction: float, expected_return: float) -> float:
        """Calculate utility score for parlay selection."""
        # Logarithmic utility (Kelly's original assumption)
        if kelly_fraction <= 0:
            return 0.0
        
        # Expected log growth
        utility = kelly_fraction * expected_return
        
        # Penalty for higher variance (larger parlays)
        variance_penalty = kelly_fraction ** 2 * 0.1
        
        return utility - variance_penalty


def create_sample_parlays(matches_data: List[Dict], score_matrices: Dict) -> List[ParlayLeg]:
    """
    Create sample parlay legs from match predictions.
    
    Args:
        matches_data: List of match dictionaries with predictions
        score_matrices: Dictionary mapping match keys to score matrices
        
    Returns:
        List of ParlayLeg objects
    """
    legs = []
    
    for match in matches_data:
        match_key = f"{match['home_team']}_vs_{match['away_team']}"
        
        if match_key in score_matrices:
            score_matrix = score_matrices[match_key]
            
            # Create legs for each outcome if odds are available
            outcomes = ['home', 'draw', 'away']
            odds_keys = ['odds_home', 'odds_draw', 'odds_away']
            
            for outcome, odds_key in zip(outcomes, odds_keys):
                if odds_key in match and match[odds_key] > 1.0:
                    leg = ParlayLeg(
                        home_team=match['home_team'],
                        away_team=match['away_team'],
                        outcome=outcome,
                        odds=match[odds_key],
                        score_matrix=score_matrix,
                        date=match.get('date')
                    )
                    legs.append(leg)
    
    return legs


if __name__ == "__main__":
    # Example usage
    
    # Create sample score matrices (normally from Dixon-Coles model)
    np.random.seed(42)
    sample_matrix1 = np.random.dirichlet(np.ones(49)).reshape(7, 7)  # 7x7 matrix
    sample_matrix2 = np.random.dirichlet(np.ones(49)).reshape(7, 7)
    
    # Create sample legs
    legs = [
        ParlayLeg("Real Madrid", "Man City", "home", 2.1, sample_matrix1),
        ParlayLeg("Barcelona", "PSG", "away", 2.8, sample_matrix2),
        ParlayLeg("Bayern", "Arsenal", "home", 1.9, sample_matrix1)
    ]
    
    # Simulate parlay
    simulator = ParlaySimulator(n_simulations=10000)
    result = simulator.simulate_parlay(legs[:2])  # 2-leg parlay
    
    print("Parlay Simulation Results:")
    print(f"Probability: {result['probability']:.3f}")
    print(f"Independent probability: {result['independent_probability']:.3f}")
    print(f"Correlation effect: {result['correlation_effect']:.3f}")
    print(f"Expected return: {result['expected_return']:.3f}")
    print(f"Market odds: {result['market_odds']:.2f}")
    
    # Optimize parlay portfolio
    optimizer = ParlayOptimizer(simulator)
    portfolio = optimizer.optimize_parlay_portfolio(legs, bankroll=1000)
    
    print(f"\nOptimal Portfolio:")
    print(f"Number of parlays: {portfolio['num_parlays']}")
    print(f"Total stake: ${portfolio['total_stake']:.2f}")
    print(f"Expected profit: ${portfolio['expected_profit']:.2f}")
    print(f"Expected yield: {portfolio['expected_yield']:.1f}%")
