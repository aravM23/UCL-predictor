# Champions League Betting Assistant

A comprehensive Champions League match prediction system that combines machine learning, sports analytics, and betting theory to generate:

- **Calibrated probabilities** for home win, draw, away win
- **Full scoreline distributions** via bivariate Poisson with Dixon-Coles correction
- **Market edge detection** vs bookmaker odds + Kelly staking suggestions
- **Parlay optimization** that accounts for correlation using simulated joint scorelines

## 🏗️ Architecture

### Data Layer
- Historical matches + odds from Kaggle (European Soccer Database)
- Team-level ELO ratings with temporal dynamics
- Player availability and injury data (when available)
- Expected goals (xG) data or shots-based proxies

### Feature Engineering
- Rolling team form (goals for/against over 5/10 matches)
- Home/away performance splits
- Schedule fatigue (rest days, consecutive away games, travel)
- ELO deltas and opponent strength differentials
- Odds-derived features and market sentiment

### Dual Modeling Approach

**Model A: Gradient Boosting (LightGBM)**
- Target: W/D/L classification with class weights for draws
- Features: All engineered features + contextual variables
- CV: Time-series grouped splits by season/matchweek
- Output: P_A(home), P_A(draw), P_A(away)

**Model B: Bivariate Poisson with Dixon-Coles**
- Learns team attack_i, defense_i parameters + home advantage
- Applies Dixon-Coles correction for realistic low-score dependencies
- Generates full scoreline matrix P(goals_home, goals_away)
- Derives P_B(W/D/L) by summing appropriate cells

### Meta Layer
- **Probability Calibration**: Isotonic regression for both models
- **Ensemble**: Logistic regression stacking [P_A, P_B, meta_features] → final P(W/D/L)
- **Reconciliation**: Rescale scoreline matrix to match ensemble W/D/L totals

### Betting Intelligence
- **Edge Detection**: model_prob - market_prob after removing bookmaker margin
- **Kelly Criterion**: Optimal stake sizing with safety caps (f_max = 0.05)
- **Parlay Optimization**: Monte Carlo simulation accounting for correlation via joint scoreline sampling

## 🚀 Quick Start

### Installation
```bash
# Clone or download the project
cd UCL_predictor

# Install dependencies
pip install -r requirements.txt

# Download sample data (see data/README.md for Kaggle setup)
python src/data/ingest.py --source kaggle --competition european-soccer-database
```

### Training Models
```bash
# Feature engineering
python src/data/features.py --input data/raw/matches.csv --output data/processed/features.pkl

# Train both models
python src/models/train_wdl.py --config configs/train.yaml
python src/models/poisson.py --config configs/train.yaml

# Calibrate and ensemble
python src/models/calibrate.py
python src/models/ensemble.py
```

### Making Predictions
```bash
# CLI prediction
python cli.py predict --home "Real Madrid" --away "Manchester City" --date "2024-04-17"

# Start Streamlit app locally
streamlit run simple_app.py

# OR visit the live demo
# 🌐 Live Demo: [UCL Predictor App](https://share.streamlit.io) (coming soon)
```

## 📊 Key Features

### Match Prediction
- Ensemble probabilities for W/D/L outcomes
- Top 10 most likely scorelines with probabilities
- Fair odds calculation and market edge detection
- Confidence intervals and scenario analysis

### Betting Analysis
- Kelly fraction calculations with customizable caps
- Historical backtest performance vs closing odds
- Bankroll management recommendations
- Risk-adjusted expected returns

### Parlay Optimization
- Multi-leg correlation modeling via Monte Carlo
- Expected utility maximization
- Optimal leg selection given constraints
- Joint probability estimation with shared variance

## 🏆 What Makes This Unique

1. **Dual Perspective**: Reconciles class probabilities with full scoreline distributions
2. **Realistic Dependencies**: Dixon-Coles correction captures Champions League scoring patterns
3. **Correlation Modeling**: Parlay optimizer uses joint sampling rather than naive independence
4. **Practical Outputs**: Fair odds, edges, Kelly fractions, and scenario toggles
5. **Safety First**: Conservative Kelly caps and educational disclaimers

## 📈 Model Performance

Target metrics tracked:
- **Accuracy**: Brier score and multiclass log-loss for W/D/L
- **Calibration**: Reliability plots and Expected Calibration Error (ECE)
- **Profitability**: Profit curves against historical closing odds
- **Scorelines**: Log-likelihood and over/under hit rates

## ⚠️ Disclaimer

**For educational and research purposes only.** 
- No guarantee of profit in sports betting
- Kelly fractions capped at conservative levels (5% max)
- Always bet responsibly and within your means
- Past performance does not guarantee future results

## 📁 Project Structure

```
UCL_predictor/
├── data/
│   ├── raw/              # Raw Kaggle data
│   └── processed/        # Feature-engineered datasets
├── notebooks/            # Exploratory analysis and prototyping
├── src/
│   ├── data/
│   │   ├── ingest.py     # Data loading and preprocessing
│   │   └── features.py   # Feature engineering pipeline
│   ├── models/
│   │   ├── train_wdl.py  # LightGBM training
│   │   ├── poisson.py    # Dixon-Coles implementation
│   │   ├── calibrate.py  # Probability calibration
│   │   └── ensemble.py   # Model stacking
│   ├── bet/
│   │   ├── market.py     # Odds analysis and edge detection
│   │   ├── kelly.py      # Kelly criterion implementation
│   │   ├── parlay.py     # Multi-leg optimization
│   │   └── simulate.py   # Monte Carlo correlation modeling
│   ├── app/
│   │   └── serve.py      # Streamlit app and REST API
│   └── utils/
│       ├── metrics.py    # Evaluation functions
│       └── io.py         # File I/O utilities
├── configs/
│   ├── features.yaml     # Feature engineering config
│   └── train.yaml        # Model training parameters
├── cli.py                # Command-line interface
└── requirements.txt      # Python dependencies
```

## 🔧 Configuration

Key parameters in `configs/train.yaml`:
- **Features**: Rolling window sizes, max goals for Poisson
- **Model A**: LightGBM hyperparameters and class weights
- **Model B**: Dixon-Coles correlation flag and goal limits
- **Betting**: Kelly caps, minimum edge thresholds

## 📚 Data Sources

- **Primary**: Kaggle European Soccer Database (Champions League subset)
- **Odds**: Bookmaker closing lines (Bet365, Pinnacle, etc.)
- **Live fixtures**: football-data.org API or similar
- **Optional**: Open xG sources (falls back to shots-based proxies)

## 🤝 Contributing

Contributions welcome! Areas for enhancement:
- Player-level injury/availability modeling
- Market-specific odds analysis (Asian handicaps, totals)
- Real-time data pipelines
- Advanced ensemble techniques
- Mobile app interface

## 📄 License

MIT License - see LICENSE file for details.

---

**Made by Arav Mathur** 😎
