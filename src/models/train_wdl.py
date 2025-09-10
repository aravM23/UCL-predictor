"""
Model A: LightGBM training for W/D/L prediction.
Handles multiclass classification with proper time-series cross-validation.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, brier_score_loss
import pickle
import yaml
from pathlib import Path
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)


class WDLTrainer:
    """Trainer for Win/Draw/Loss prediction using LightGBM."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.feature_names = None
        self.label_encoder = {'H': 0, 'D': 1, 'A': 2}
        self.label_decoder = {0: 'H', 1: 'D', 2: 'A'}
        
    def prepare_data(self, features_df: pd.DataFrame, targets_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for training.
        
        Args:
            features_df: Feature matrix
            targets_df: Target labels
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        # Merge features with targets
        data = features_df.merge(targets_df[['date', 'home_team', 'away_team', 'result']], 
                               on=['date', 'home_team', 'away_team'], how='inner')
        
        # Sort by date for time-series splits
        data = data.sort_values('date')
        
        # Extract features (exclude meta columns)
        feature_cols = [col for col in data.columns 
                       if col not in ['date', 'home_team', 'away_team', 'result', 'home_goals', 'away_goals']]
        
        X = data[feature_cols].values
        y = data['result'].map(self.label_encoder).values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.feature_names = feature_cols
        logger.info(f"Prepared data: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y, feature_cols
    
    def train_with_cv(self, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> Dict:
        """
        Train model with time-series cross-validation.
        
        Args:
            X: Feature matrix
            y: Target labels  
            n_splits: Number of CV splits
            
        Returns:
            Dictionary with training results
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_results = {
            'train_scores': [],
            'val_scores': [],
            'models': [],
            'feature_importance': []
        }
        
        model_params = self.config['modelA']['params'].copy()
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info(f"Training fold {fold + 1}/{n_splits}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create LightGBM datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # Train model
            model = lgb.train(
                model_params,
                train_data,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'val'],
                callbacks=[
                    lgb.early_stopping(self.config['modelA'].get('early_stopping_rounds', 100)),
                    lgb.log_evaluation(self.config['modelA'].get('verbose_eval', 100))
                ]
            )
            
            # Evaluate
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            
            train_score = log_loss(y_train, train_pred)
            val_score = log_loss(y_val, val_pred)
            
            cv_results['train_scores'].append(train_score)
            cv_results['val_scores'].append(val_score)
            cv_results['models'].append(model)
            cv_results['feature_importance'].append(model.feature_importance(importance_type='gain'))
            
            logger.info(f"Fold {fold + 1} - Train log-loss: {train_score:.4f}, Val log-loss: {val_score:.4f}")
        
        # Select best model (lowest validation score)
        best_fold = np.argmin(cv_results['val_scores'])
        self.model = cv_results['models'][best_fold]
        
        # Average results
        mean_train_score = np.mean(cv_results['train_scores'])
        mean_val_score = np.mean(cv_results['val_scores'])
        std_val_score = np.std(cv_results['val_scores'])
        
        logger.info(f"CV Results - Train: {mean_train_score:.4f}, Val: {mean_val_score:.4f} ± {std_val_score:.4f}")
        logger.info(f"Best model from fold {best_fold + 1}")
        
        return cv_results
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability matrix (n_samples, 3) for [Home, Draw, Away]
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        probs = self.model.predict(X)
        
        # Ensure probabilities sum to 1 (numerical stability)
        probs = probs / probs.sum(axis=1, keepdims=True)
        
        return probs
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """Get feature importance from trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        importance = self.model.feature_importance(importance_type=importance_type)
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def save_model(self, path: str):
        """Save trained model to disk."""
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'label_encoder': self.label_encoder,
            'label_decoder': self.label_decoder,
            'config': self.config
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
            
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model from disk."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
            
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.label_encoder = model_data['label_encoder']
        self.label_decoder = model_data['label_decoder']
        self.config = model_data['config']
        
        logger.info(f"Model loaded from {path}")


def train_wdl_model(features_df: pd.DataFrame, targets_df: pd.DataFrame, config: Dict) -> WDLTrainer:
    """
    Main function to train W/D/L model.
    
    Args:
        features_df: Feature matrix
        targets_df: Target labels
        config: Configuration dictionary
        
    Returns:
        Trained WDLTrainer instance
    """
    trainer = WDLTrainer(config)
    
    # Prepare data
    X, y, feature_names = trainer.prepare_data(features_df, targets_df)
    
    # Train with cross-validation
    cv_results = trainer.train_with_cv(X, y, n_splits=5)
    
    # Print feature importance
    importance_df = trainer.get_feature_importance()
    logger.info("Top 10 most important features:")
    logger.info(importance_df.head(10).to_string(index=False))
    
    return trainer


def evaluate_model(trainer: WDLTrainer, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """
    Evaluate trained model on test set.
    
    Args:
        trainer: Trained WDLTrainer
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Predictions
    probs = trainer.predict_proba(X_test)
    preds = np.argmax(probs, axis=1)
    
    # Accuracy
    accuracy = (preds == y_test).mean()
    
    # Log-loss
    logloss = log_loss(y_test, probs)
    
    # Brier score (per class)
    brier_scores = {}
    for i, class_name in trainer.label_decoder.items():
        y_binary = (y_test == i).astype(int)
        brier_scores[f'brier_{class_name}'] = brier_score_loss(y_binary, probs[:, i])
    
    # Class-wise accuracy
    class_accuracies = {}
    for i, class_name in trainer.label_decoder.items():
        mask = (y_test == i)
        if mask.sum() > 0:
            class_accuracies[f'accuracy_{class_name}'] = (preds[mask] == i).mean()
    
    results = {
        'accuracy': accuracy,
        'log_loss': logloss,
        **brier_scores,
        **class_accuracies
    }
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train LightGBM model for W/D/L prediction')
    parser.add_argument('--features', required=True, help='Features pickle file')
    parser.add_argument('--config', default='configs/train.yaml', help='Config file')
    parser.add_argument('--output', default='models/wdl_model.pkl', help='Output model file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    with open(args.features, 'rb') as f:
        data = pickle.load(f)
        
    features_df = data['features']
    targets_df = data['targets']
    
    # Train model
    trainer = train_wdl_model(features_df, targets_df, config)
    
    # Save model
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output)
    
    logger.info(f"Training completed. Model saved to {args.output}")
