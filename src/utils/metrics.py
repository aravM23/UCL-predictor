"""
Utility functions for model evaluation and metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional


def calculate_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Calculate Brier score for probabilistic predictions.
    
    Args:
        y_true: True binary outcomes (0 or 1)
        y_prob: Predicted probabilities
        
    Returns:
        Brier score (lower is better)
    """
    return brier_score_loss(y_true, y_prob)


def calculate_multiclass_brier(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Calculate Brier scores for multiclass predictions.
    
    Args:
        y_true: True class labels (0, 1, 2 for H, D, A)
        y_prob: Predicted probabilities (n_samples, n_classes)
        
    Returns:
        Dictionary with Brier scores per class and overall
    """
    n_classes = y_prob.shape[1]
    class_names = ['Home', 'Draw', 'Away']
    
    brier_scores = {}
    
    for i in range(n_classes):
        y_binary = (y_true == i).astype(int)
        brier_scores[f'brier_{class_names[i]}'] = brier_score_loss(y_binary, y_prob[:, i])
    
    # Overall Brier score (sum across classes)
    brier_scores['brier_overall'] = sum(brier_scores.values())
    
    return brier_scores


def calculate_expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, 
                                       n_bins: int = 10) -> float:
    """
    Calculate Expected Calibration Error (ECE).
    
    Args:
        y_true: True binary outcomes
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration
        
    Returns:
        ECE value
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find predictions in this bin
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # Accuracy in this bin
            accuracy_in_bin = y_true[in_bin].mean()
            
            # Average confidence in this bin
            avg_confidence_in_bin = y_prob[in_bin].mean()
            
            # Add to ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def plot_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, 
                          n_bins: int = 10, title: str = "Calibration Plot"):
    """
    Plot calibration curve for probabilistic predictions.
    
    Args:
        y_true: True binary outcomes
        y_prob: Predicted probabilities
        n_bins: Number of bins
        title: Plot title
    """
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins
    )
    
    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.ylabel("Fraction of positives")
    plt.xlabel("Mean predicted probability")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def calculate_profit_curve(predictions: pd.DataFrame, kelly_cap: float = 0.05,
                          initial_bankroll: float = 1000) -> Dict:
    """
    Calculate profit curve for betting strategy.
    
    Args:
        predictions: DataFrame with predictions and actual outcomes
        kelly_cap: Kelly fraction cap
        initial_bankroll: Starting bankroll
        
    Returns:
        Dictionary with profit metrics
    """
    bankroll = initial_bankroll
    bankroll_history = [bankroll]
    bet_history = []
    
    for _, row in predictions.iterrows():
        # Calculate Kelly stake
        prob = row['predicted_prob']
        odds = row['odds']
        
        if prob > 0 and odds > 1:
            # Kelly formula
            kelly_f = ((odds * prob) - 1) / (odds - 1)
            kelly_f = max(0, min(kelly_f, kelly_cap))
            
            stake = bankroll * kelly_f
            
            if stake > 0:
                # Determine outcome
                won = row['outcome'] == 1
                
                if won:
                    profit = stake * (odds - 1)
                    bankroll += profit
                else:
                    bankroll -= stake
                
                bet_history.append({
                    'stake': stake,
                    'profit': profit if won else -stake,
                    'won': won,
                    'odds': odds,
                    'kelly_fraction': kelly_f
                })
        
        bankroll_history.append(bankroll)
    
    # Calculate metrics
    total_bets = len(bet_history)
    if total_bets > 0:
        wins = sum(1 for bet in bet_history if bet['won'])
        total_profit = sum(bet['profit'] for bet in bet_history)
        total_staked = sum(bet['stake'] for bet in bet_history)
        
        return {
            'final_bankroll': bankroll,
            'total_return_pct': ((bankroll - initial_bankroll) / initial_bankroll) * 100,
            'total_bets': total_bets,
            'win_rate': wins / total_bets,
            'total_profit': total_profit,
            'roi_pct': (total_profit / total_staked) * 100 if total_staked > 0 else 0,
            'bankroll_history': bankroll_history,
            'bet_history': bet_history
        }
    
    return {'error': 'No bets placed'}


def evaluate_wdl_model(y_true: np.ndarray, y_prob: np.ndarray, 
                      class_names: List[str] = None) -> Dict:
    """
    Comprehensive evaluation of W/D/L model.
    
    Args:
        y_true: True class labels
        y_prob: Predicted probabilities
        class_names: Names of classes
        
    Returns:
        Dictionary with all evaluation metrics
    """
    if class_names is None:
        class_names = ['Home', 'Draw', 'Away']
    
    n_classes = y_prob.shape[1]
    
    # Basic metrics
    y_pred = np.argmax(y_prob, axis=1)
    accuracy = (y_pred == y_true).mean()
    
    # Log loss
    logloss = log_loss(y_true, y_prob)
    
    # Brier scores
    brier_scores = calculate_multiclass_brier(y_true, y_prob)
    
    # Per-class metrics
    class_metrics = {}
    for i in range(n_classes):
        y_binary = (y_true == i).astype(int)
        y_prob_binary = y_prob[:, i]
        
        # Only calculate if we have positive examples
        if y_binary.sum() > 0:
            class_metrics[class_names[i]] = {
                'accuracy': (y_pred[y_true == i] == i).mean() if (y_true == i).sum() > 0 else 0,
                'brier_score': brier_score_loss(y_binary, y_prob_binary),
                'ece': calculate_expected_calibration_error(y_binary, y_prob_binary)
            }
            
            # ROC AUC if binary classification makes sense
            if len(np.unique(y_binary)) > 1:
                class_metrics[class_names[i]]['roc_auc'] = roc_auc_score(y_binary, y_prob_binary)
    
    return {
        'overall_accuracy': accuracy,
        'log_loss': logloss,
        **brier_scores,
        'class_metrics': class_metrics
    }


def compare_models(model_results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Compare multiple model results.
    
    Args:
        model_results: Dictionary of model names to evaluation results
        
    Returns:
        DataFrame comparing models
    """
    comparison_data = []
    
    for model_name, results in model_results.items():
        row = {
            'model': model_name,
            'accuracy': results.get('overall_accuracy', 0),
            'log_loss': results.get('log_loss', float('inf')),
            'brier_overall': results.get('brier_overall', float('inf'))
        }
        
        # Add class-specific metrics if available
        if 'class_metrics' in results:
            for class_name, metrics in results['class_metrics'].items():
                row[f'{class_name.lower()}_accuracy'] = metrics.get('accuracy', 0)
                row[f'{class_name.lower()}_brier'] = metrics.get('brier_score', float('inf'))
        
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)


def bootstrap_confidence_intervals(y_true: np.ndarray, y_prob: np.ndarray,
                                 metric_func, n_bootstrap: int = 1000,
                                 confidence_level: float = 0.95) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence intervals for a metric.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        metric_func: Function to calculate metric
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        Tuple of (metric_value, lower_bound, upper_bound)
    """
    n_samples = len(y_true)
    bootstrap_scores = []
    
    # Original metric
    original_score = metric_func(y_true, y_prob)
    
    # Bootstrap samples
    np.random.seed(42)
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_score = metric_func(y_true[indices], y_prob[indices])
        bootstrap_scores.append(bootstrap_score)
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_scores, lower_percentile)
    upper_bound = np.percentile(bootstrap_scores, upper_percentile)
    
    return original_score, lower_bound, upper_bound


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate sample data
    n_samples = 1000
    y_true = np.random.choice([0, 1, 2], size=n_samples, p=[0.4, 0.3, 0.3])
    
    # Generate somewhat realistic probabilities
    y_prob = np.random.dirichlet([2, 1.5, 2], size=n_samples)
    
    # Add some noise to make predictions less perfect
    noise = np.random.normal(0, 0.1, y_prob.shape)
    y_prob += noise
    y_prob = np.abs(y_prob)  # Ensure positive
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)  # Normalize
    
    # Evaluate model
    results = evaluate_wdl_model(y_true, y_prob)
    
    print("Model Evaluation Results:")
    print(f"Overall Accuracy: {results['overall_accuracy']:.3f}")
    print(f"Log Loss: {results['log_loss']:.3f}")
    print(f"Overall Brier Score: {results['brier_overall']:.3f}")
    
    print("\nPer-class metrics:")
    for class_name, metrics in results['class_metrics'].items():
        print(f"{class_name}: Accuracy={metrics['accuracy']:.3f}, Brier={metrics['brier_score']:.3f}")
    
    # Bootstrap confidence interval example
    def accuracy_func(y_true, y_prob):
        y_pred = np.argmax(y_prob, axis=1)
        return (y_pred == y_true).mean()
    
    acc_mean, acc_lower, acc_upper = bootstrap_confidence_intervals(
        y_true, y_prob, accuracy_func
    )
    print(f"\nAccuracy: {acc_mean:.3f} (95% CI: {acc_lower:.3f} - {acc_upper:.3f})")
