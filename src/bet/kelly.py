"""
Kelly criterion implementation for optimal bet sizing.
Calculates Kelly fractions with safety caps and bankroll management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def kelly_fraction(probability: float, odds: float, cap: float = 0.05) -> float:
    """
    Calculate Kelly criterion fraction for a bet.
    
    Formula: f* = (bp - q) / b
    Where:
    - b = odds - 1 (net odds)
    - p = probability of winning
    - q = probability of losing = 1 - p
    
    Args:
        probability: Model probability of the outcome
        odds: Decimal odds offered by bookmaker
        cap: Maximum fraction to bet (safety cap)
        
    Returns:
        Kelly fraction (0 to cap)
    """
    if probability <= 0 or probability >= 1:
        return 0.0
        
    if odds <= 1.0:
        return 0.0
    
    # Net odds (profit per unit stake)
    b = odds - 1
    p = probability
    q = 1 - p
    
    # Kelly formula
    kelly_f = (b * p - q) / b
    
    # Only bet if Kelly fraction is positive (positive expected value)
    if kelly_f <= 0:
        return 0.0
    
    # Apply safety cap
    return min(kelly_f, cap)


def fractional_kelly(probability: float, odds: float, fraction: float = 0.25, cap: float = 0.05) -> float:
    """
    Calculate fractional Kelly for more conservative betting.
    
    Args:
        probability: Model probability of the outcome
        odds: Decimal odds offered by bookmaker
        fraction: Fraction of full Kelly to use (e.g., 0.25 for quarter Kelly)
        cap: Maximum fraction to bet
        
    Returns:
        Fractional Kelly stake
    """
    full_kelly = kelly_fraction(probability, odds, cap=1.0)  # No cap for calculation
    fractional_stake = full_kelly * fraction
    
    return min(fractional_stake, cap)


class KellyCalculator:
    """Kelly criterion calculator with advanced features."""
    
    def __init__(self, kelly_cap: float = 0.05, min_edge: float = 0.02, 
                 min_probability: float = 0.05, max_probability: float = 0.95):
        self.kelly_cap = kelly_cap
        self.min_edge = min_edge
        self.min_probability = min_probability
        self.max_probability = max_probability
        
    def calculate_stake(self, probability: float, odds: float, 
                       bankroll: float = 1000, method: str = 'full') -> Dict:
        """
        Calculate optimal stake using Kelly criterion.
        
        Args:
            probability: Model probability of winning
            odds: Decimal odds from bookmaker
            bankroll: Total bankroll amount
            method: 'full', 'quarter', 'eighth' Kelly
            
        Returns:
            Dictionary with stake calculations
        """
        # Validate inputs
        if not (self.min_probability <= probability <= self.max_probability):
            return self._no_bet_result("Probability out of valid range")
            
        if odds <= 1.0:
            return self._no_bet_result("Invalid odds")
        
        # Calculate edge
        implied_prob = 1 / odds
        edge = probability - implied_prob
        
        if edge < self.min_edge:
            return self._no_bet_result("Insufficient edge")
        
        # Calculate Kelly fraction based on method
        if method == 'full':
            kelly_f = kelly_fraction(probability, odds, self.kelly_cap)
        elif method == 'quarter':
            kelly_f = fractional_kelly(probability, odds, 0.25, self.kelly_cap)
        elif method == 'eighth':
            kelly_f = fractional_kelly(probability, odds, 0.125, self.kelly_cap)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Calculate stakes
        stake_amount = bankroll * kelly_f
        
        # Expected return calculations
        expected_return = (probability * (odds - 1) - (1 - probability)) * stake_amount
        expected_growth = np.log(1 + kelly_f * (odds - 1)) * probability + np.log(1 - kelly_f) * (1 - probability)
        
        return {
            'recommend_bet': kelly_f > 0,
            'kelly_fraction': kelly_f,
            'stake_amount': stake_amount,
            'stake_percentage': kelly_f * 100,
            'edge': edge,
            'expected_return': expected_return,
            'expected_growth': expected_growth,
            'probability': probability,
            'odds': odds,
            'implied_probability': implied_prob,
            'method': method,
            'risk_level': self._assess_risk_level(kelly_f, edge),
            'confidence': self._assess_confidence(probability, edge)
        }
    
    def calculate_multiple_bets(self, bets: List[Dict], bankroll: float = 1000, 
                               method: str = 'full') -> pd.DataFrame:
        """
        Calculate stakes for multiple independent bets.
        
        Args:
            bets: List of bet dictionaries with 'probability' and 'odds'
            bankroll: Total bankroll
            method: Kelly method to use
            
        Returns:
            DataFrame with bet recommendations
        """
        results = []
        
        for i, bet in enumerate(bets):
            calc = self.calculate_stake(
                bet['probability'], 
                bet['odds'], 
                bankroll, 
                method
            )
            
            calc['bet_id'] = i
            calc['description'] = bet.get('description', f"Bet {i+1}")
            calc['home_team'] = bet.get('home_team', '')
            calc['away_team'] = bet.get('away_team', '')
            calc['outcome'] = bet.get('outcome', '')
            
            results.append(calc)
        
        df = pd.DataFrame(results)
        
        # Calculate total exposure
        total_stake = df['stake_amount'].sum()
        df['total_exposure'] = total_stake
        df['exposure_percentage'] = (total_stake / bankroll) * 100
        
        # Sort by expected return
        df = df.sort_values('expected_return', ascending=False)
        
        return df
    
    def optimize_portfolio(self, bets: List[Dict], bankroll: float = 1000, 
                          max_exposure: float = 0.1) -> Dict:
        """
        Optimize bet portfolio considering correlation and exposure limits.
        
        Args:
            bets: List of bet opportunities
            bankroll: Available bankroll
            max_exposure: Maximum total exposure as fraction of bankroll
            
        Returns:
            Optimized portfolio recommendations
        """
        # Calculate individual Kelly stakes
        bet_calcs = []
        for bet in bets:
            calc = self.calculate_stake(bet['probability'], bet['odds'], bankroll)
            if calc['recommend_bet']:
                bet_calcs.append({
                    **calc,
                    **bet,
                    'individual_stake': calc['stake_amount']
                })
        
        if not bet_calcs:
            return {'optimized_bets': [], 'total_stake': 0, 'expected_return': 0}
        
        # Sort by expected return per unit risk
        bet_calcs.sort(key=lambda x: x['expected_return'] / max(x['stake_amount'], 1), reverse=True)
        
        # Select bets within exposure limit
        total_stake = 0
        max_stake = bankroll * max_exposure
        selected_bets = []
        
        for bet in bet_calcs:
            if total_stake + bet['individual_stake'] <= max_stake:
                selected_bets.append(bet)
                total_stake += bet['individual_stake']
            else:
                # Partial allocation if it fits
                remaining = max_stake - total_stake
                if remaining > 0:
                    bet['stake_amount'] = remaining
                    bet['kelly_fraction'] = remaining / bankroll
                    selected_bets.append(bet)
                    total_stake = max_stake
                break
        
        # Calculate portfolio metrics
        total_expected_return = sum(bet['expected_return'] for bet in selected_bets)
        
        return {
            'optimized_bets': selected_bets,
            'total_stake': total_stake,
            'exposure_percentage': (total_stake / bankroll) * 100,
            'expected_return': total_expected_return,
            'expected_yield': (total_expected_return / total_stake * 100) if total_stake > 0 else 0,
            'num_bets': len(selected_bets),
            'avg_edge': np.mean([bet['edge'] for bet in selected_bets]) if selected_bets else 0
        }
    
    def _no_bet_result(self, reason: str) -> Dict:
        """Return structure for when no bet is recommended."""
        return {
            'recommend_bet': False,
            'kelly_fraction': 0.0,
            'stake_amount': 0.0,
            'stake_percentage': 0.0,
            'edge': 0.0,
            'expected_return': 0.0,
            'expected_growth': 0.0,
            'reason': reason,
            'risk_level': 'none',
            'confidence': 'none'
        }
    
    def _assess_risk_level(self, kelly_fraction: float, edge: float) -> str:
        """Assess risk level of a bet."""
        if kelly_fraction == 0:
            return 'none'
        elif kelly_fraction < 0.01:
            return 'very_low'
        elif kelly_fraction < 0.02:
            return 'low'
        elif kelly_fraction < 0.05:
            return 'medium'
        else:
            return 'high'
    
    def _assess_confidence(self, probability: float, edge: float) -> str:
        """Assess confidence level in a bet."""
        if edge < 0.02:
            return 'low'
        elif edge < 0.05:
            return 'medium'
        elif edge < 0.10:
            return 'high'
        else:
            return 'very_high'


def simulate_kelly_performance(bets_history: pd.DataFrame, initial_bankroll: float = 1000,
                              kelly_cap: float = 0.05) -> Dict:
    """
    Simulate historical performance using Kelly criterion.
    
    Args:
        bets_history: DataFrame with columns [probability, odds, outcome, date]
        initial_bankroll: Starting bankroll
        kelly_cap: Kelly fraction cap
        
    Returns:
        Dictionary with simulation results
    """
    bankroll = initial_bankroll
    bankroll_history = [bankroll]
    bet_results = []
    
    calculator = KellyCalculator(kelly_cap=kelly_cap)
    
    for _, bet in bets_history.iterrows():
        # Calculate stake
        stake_calc = calculator.calculate_stake(
            bet['probability'], 
            bet['odds'], 
            bankroll
        )
        
        if stake_calc['recommend_bet']:
            stake = stake_calc['stake_amount']
            
            # Determine outcome (1 if won, 0 if lost)
            won = bet['outcome'] == 1
            
            if won:
                profit = stake * (bet['odds'] - 1)
                bankroll += profit
            else:
                bankroll -= stake
            
            bet_results.append({
                'date': bet.get('date'),
                'stake': stake,
                'odds': bet['odds'],
                'probability': bet['probability'],
                'won': won,
                'profit': profit if won else -stake,
                'bankroll': bankroll,
                'kelly_fraction': stake_calc['kelly_fraction']
            })
        
        bankroll_history.append(bankroll)
    
    # Calculate performance metrics
    if bet_results:
        total_bets = len(bet_results)
        wins = sum(1 for bet in bet_results if bet['won'])
        total_staked = sum(bet['stake'] for bet in bet_results)
        total_profit = sum(bet['profit'] for bet in bet_results)
        
        win_rate = wins / total_bets
        roi = (total_profit / total_staked) * 100 if total_staked > 0 else 0
        final_return = ((bankroll - initial_bankroll) / initial_bankroll) * 100
        
        # Drawdown calculation
        peak_bankroll = initial_bankroll
        max_drawdown = 0
        for balance in bankroll_history:
            if balance > peak_bankroll:
                peak_bankroll = balance
            drawdown = (peak_bankroll - balance) / peak_bankroll
            max_drawdown = max(max_drawdown, drawdown)
        
        return {
            'initial_bankroll': initial_bankroll,
            'final_bankroll': bankroll,
            'total_return_pct': final_return,
            'total_bets': total_bets,
            'wins': wins,
            'losses': total_bets - wins,
            'win_rate': win_rate,
            'total_staked': total_staked,
            'total_profit': total_profit,
            'roi_pct': roi,
            'max_drawdown_pct': max_drawdown * 100,
            'bankroll_history': bankroll_history,
            'bet_results': bet_results
        }
    
    return {'error': 'No bets placed in simulation'}


if __name__ == "__main__":
    # Example usage
    calculator = KellyCalculator(kelly_cap=0.05, min_edge=0.02)
    
    # Single bet example
    result = calculator.calculate_stake(
        probability=0.45,  # 45% chance
        odds=2.4,          # Decimal odds of 2.4
        bankroll=1000
    )
    
    print("Single Bet Kelly Calculation:")
    print(f"Recommend bet: {result['recommend_bet']}")
    print(f"Kelly fraction: {result['kelly_fraction']:.3f}")
    print(f"Stake amount: ${result['stake_amount']:.2f}")
    print(f"Edge: {result['edge']:.3f}")
    print(f"Expected return: ${result['expected_return']:.2f}")
    print(f"Risk level: {result['risk_level']}")
    
    # Multiple bets example
    bets = [
        {'probability': 0.45, 'odds': 2.4, 'description': 'Real Madrid Win'},
        {'probability': 0.30, 'odds': 4.0, 'description': 'Draw'},
        {'probability': 0.60, 'odds': 1.8, 'description': 'Over 2.5 Goals'}
    ]
    
    portfolio = calculator.calculate_multiple_bets(bets, bankroll=1000)
    print("\nMultiple Bets Portfolio:")
    print(portfolio[['description', 'recommend_bet', 'stake_amount', 'expected_return']].to_string(index=False))
