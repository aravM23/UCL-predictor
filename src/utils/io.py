"""
I/O utilities for the Champions League predictor.
Handles file operations, data serialization, and model persistence.
"""

import pickle
import json
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union, List
import logging

logger = logging.getLogger(__name__)


def save_pickle(obj: Any, filepath: Union[str, Path]) -> None:
    """
    Save object to pickle file.
    
    Args:
        obj: Object to save
        filepath: Path to save file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    
    logger.info(f"Saved object to {filepath}")


def load_pickle(filepath: Union[str, Path]) -> Any:
    """
    Load object from pickle file.
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        Loaded object
    """
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    
    logger.info(f"Loaded object from {filepath}")
    return obj


def save_json(obj: Any, filepath: Union[str, Path], indent: int = 2) -> None:
    """
    Save object to JSON file.
    
    Args:
        obj: Object to save (must be JSON serializable)
        filepath: Path to save file
        indent: JSON indentation
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to native Python types
    if hasattr(obj, 'tolist') and hasattr(obj, 'dtype'):
        obj = obj.tolist()
    elif isinstance(obj, dict):
        obj = convert_numpy_types(obj)
    elif isinstance(obj, list):
        obj = [convert_numpy_types(item) if isinstance(item, dict) else item for item in obj]
    
    with open(filepath, 'w') as f:
        json.dump(obj, f, indent=indent, default=numpy_json_serializer)
    
    logger.info(f"Saved JSON to {filepath}")


def load_json(filepath: Union[str, Path]) -> Any:
    """
    Load object from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded object
    """
    with open(filepath, 'r') as f:
        obj = json.load(f)
    
    logger.info(f"Loaded JSON from {filepath}")
    return obj


def save_yaml(obj: Dict, filepath: Union[str, Path]) -> None:
    """
    Save dictionary to YAML file.
    
    Args:
        obj: Dictionary to save
        filepath: Path to save file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        yaml.dump(obj, f, default_flow_style=False, indent=2)
    
    logger.info(f"Saved YAML to {filepath}")


def load_yaml(filepath: Union[str, Path]) -> Dict:
    """
    Load dictionary from YAML file.
    
    Args:
        filepath: Path to YAML file
        
    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r') as f:
        obj = yaml.safe_load(f)
    
    logger.info(f"Loaded YAML from {filepath}")
    return obj


def save_csv(df: pd.DataFrame, filepath: Union[str, Path], **kwargs) -> None:
    """
    Save DataFrame to CSV file.
    
    Args:
        df: DataFrame to save
        filepath: Path to save file
        **kwargs: Additional arguments for pandas.to_csv()
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(filepath, index=False, **kwargs)
    logger.info(f"Saved CSV with {len(df)} rows to {filepath}")


def load_csv(filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Load DataFrame from CSV file.
    
    Args:
        filepath: Path to CSV file
        **kwargs: Additional arguments for pandas.read_csv()
        
    Returns:
        Loaded DataFrame
    """
    df = pd.read_csv(filepath, **kwargs)
    logger.info(f"Loaded CSV with {len(df)} rows from {filepath}")
    return df


def convert_numpy_types(obj: Any) -> Any:
    """
    Convert numpy types to native Python types for JSON serialization.
    
    Args:
        obj: Object that may contain numpy types
        
    Returns:
        Object with numpy types converted
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def numpy_json_serializer(obj: Any) -> Any:
    """
    JSON serializer for numpy types.
    
    Args:
        obj: Object to serialize
        
    Returns:
        Serializable version of object
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def ensure_dir(filepath: Union[str, Path]) -> Path:
    """
    Ensure directory exists for given filepath.
    
    Args:
        filepath: File or directory path
        
    Returns:
        Path object
    """
    filepath = Path(filepath)
    if filepath.suffix:  # It's a file
        filepath.parent.mkdir(parents=True, exist_ok=True)
    else:  # It's a directory
        filepath.mkdir(parents=True, exist_ok=True)
    
    return filepath


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to project root
    """
    current_file = Path(__file__)
    # Go up until we find setup.py, requirements.txt, or README.md
    for parent in current_file.parents:
        if any((parent / name).exists() for name in ['requirements.txt', 'README.md', 'setup.py']):
            return parent
    
    # Fallback to current working directory
    return Path.cwd()


def list_model_files(models_dir: Union[str, Path] = "models") -> List[Path]:
    """
    List all model files in the models directory.
    
    Args:
        models_dir: Path to models directory
        
    Returns:
        List of model file paths
    """
    models_dir = Path(models_dir)
    if not models_dir.exists():
        return []
    
    model_files = []
    for ext in ['.pkl', '.joblib', '.h5', '.pt', '.pth']:
        model_files.extend(models_dir.glob(f"*{ext}"))
    
    return sorted(model_files)


def backup_file(filepath: Union[str, Path], backup_dir: Union[str, Path] = "backups") -> Path:
    """
    Create a backup of a file with timestamp.
    
    Args:
        filepath: Path to file to backup
        backup_dir: Directory to store backups
        
    Returns:
        Path to backup file
    """
    filepath = Path(filepath)
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    backup_filename = f"{filepath.stem}_{timestamp}{filepath.suffix}"
    backup_path = backup_dir / backup_filename
    
    import shutil
    shutil.copy2(filepath, backup_path)
    
    logger.info(f"Created backup: {backup_path}")
    return backup_path


def save_predictions_report(predictions: List[Dict], output_dir: Union[str, Path] = "reports") -> Dict[str, Path]:
    """
    Save predictions in multiple formats for analysis.
    
    Args:
        predictions: List of prediction dictionaries
        output_dir: Directory to save reports
        
    Returns:
        Dictionary of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    saved_files = {}
    
    # Save as JSON
    json_path = output_dir / f"predictions_{timestamp}.json"
    save_json(predictions, json_path)
    saved_files['json'] = json_path
    
    # Convert to DataFrame and save as CSV
    if predictions:
        # Flatten predictions for CSV
        flattened = []
        for pred in predictions:
            flat_pred = {
                'home_team': pred['match']['home_team'],
                'away_team': pred['match']['away_team'],
                'date': pred['match']['date'],
                'prob_home': pred['probabilities']['home_win'],
                'prob_draw': pred['probabilities']['draw'],
                'prob_away': pred['probabilities']['away_win'],
            }
            
            # Add market analysis if available
            if 'market_analysis' in pred:
                flat_pred.update({
                    'edge_home': pred['market_analysis']['edges']['home'],
                    'edge_draw': pred['market_analysis']['edges']['draw'],
                    'edge_away': pred['market_analysis']['edges']['away'],
                    'value_bets_count': len(pred['market_analysis']['value_bets'])
                })
            
            flattened.append(flat_pred)
        
        df = pd.DataFrame(flattened)
        csv_path = output_dir / f"predictions_{timestamp}.csv"
        save_csv(df, csv_path)
        saved_files['csv'] = csv_path
    
    # Save summary statistics
    if predictions:
        summary = {
            'total_predictions': len(predictions),
            'timestamp': timestamp,
            'average_probabilities': {
                'home_win': np.mean([p['probabilities']['home_win'] for p in predictions]),
                'draw': np.mean([p['probabilities']['draw'] for p in predictions]),
                'away_win': np.mean([p['probabilities']['away_win'] for p in predictions])
            }
        }
        
        # Add market analysis summary if available
        market_predictions = [p for p in predictions if 'market_analysis' in p]
        if market_predictions:
            total_value_bets = sum(len(p['market_analysis']['value_bets']) for p in market_predictions)
            summary['market_analysis'] = {
                'predictions_with_odds': len(market_predictions),
                'total_value_bets': total_value_bets,
                'value_bet_rate': total_value_bets / len(market_predictions) if market_predictions else 0
            }
        
        summary_path = output_dir / f"summary_{timestamp}.yaml"
        save_yaml(summary, summary_path)
        saved_files['summary'] = summary_path
    
    logger.info(f"Saved prediction report to {output_dir}")
    return saved_files


def load_latest_model(model_type: str, models_dir: Union[str, Path] = "models") -> Any:
    """
    Load the latest model of a given type.
    
    Args:
        model_type: Type of model ('wdl', 'dixon_coles', etc.)
        models_dir: Directory containing models
        
    Returns:
        Loaded model object
    """
    models_dir = Path(models_dir)
    
    # Find model files matching the type
    pattern = f"*{model_type}*.pkl"
    model_files = list(models_dir.glob(pattern))
    
    if not model_files:
        raise FileNotFoundError(f"No {model_type} model files found in {models_dir}")
    
    # Get the latest file by modification time
    latest_file = max(model_files, key=lambda p: p.stat().st_mtime)
    
    return load_pickle(latest_file)


if __name__ == "__main__":
    # Example usage
    
    # Create sample data
    sample_data = {
        'predictions': [
            {
                'match': {'home_team': 'Real Madrid', 'away_team': 'Barcelona'},
                'probabilities': {'home_win': 0.45, 'draw': 0.30, 'away_win': 0.25}
            }
        ],
        'config': {'model_version': '1.0', 'timestamp': '2024-01-01'}
    }
    
    # Save in different formats
    save_json(sample_data, 'test_output/sample.json')
    save_yaml(sample_data['config'], 'test_output/config.yaml')
    
    # Load back
    loaded_data = load_json('test_output/sample.json')
    loaded_config = load_yaml('test_output/config.yaml')
    
    print("I/O operations completed successfully!")
    print(f"Loaded data keys: {list(loaded_data.keys())}")
    print(f"Loaded config: {loaded_config}")
