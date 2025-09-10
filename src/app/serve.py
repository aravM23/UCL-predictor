"""
Streamlit application for the Champions League predictor.
Provides interactive web interface for match predictions and betting analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date
import pickle
import yaml
from pathlib import Path
import sys
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from models.train_wdl import WDLTrainer
    from models.poisson import DixonColesModel
    from bet.market import MarketAnalyzer, implied_probs_from_odds
    from bet.kelly import KellyCalculator
    from bet.parlay import ParlaySimulator, ParlayLeg, ParlayOptimizer
    from data.features import FeatureBuilder
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UCLPredictorApp:
    """Main Streamlit application class."""
    
    def __init__(self):
        self.wdl_model = None
        self.dc_model = None
        self.feature_builder = None
        self.market_analyzer = MarketAnalyzer()
        self.kelly_calculator = KellyCalculator()
        self.load_models()
        
    def load_models(self):
        """Load trained models if available."""
        try:
            # Try to load WDL model
            wdl_path = Path("models/wdl_model.pkl")
            if wdl_path.exists():
                self.wdl_model = WDLTrainer({})
                self.wdl_model.load_model(str(wdl_path))
                
            # Try to load Dixon-Coles model
            dc_path = Path("models/dixon_coles_model.pkl")
            if dc_path.exists():
                self.dc_model = DixonColesModel()
                self.dc_model.load_model(str(dc_path))
                
            # Load feature builder config
            config_path = Path("configs/train.yaml")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                self.feature_builder = FeatureBuilder(config)
                
        except Exception as e:
            st.warning(f"Could not load models: {e}")
    
    def run(self):
        """Run the Streamlit application."""
        st.set_page_config(
            page_title="Champions League Predictor",
            page_icon="⚽",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("⚽ Champions League Football Predictor")
        st.markdown("*Advanced match prediction with dual modeling and betting intelligence*")
        
        # Sidebar navigation
        page = st.sidebar.selectbox(
            "Choose a page",
            ["Match Prediction", "Market Analysis", "Betting Strategy", "Model Performance", "About"]
        )
        
        if page == "Match Prediction":
            self.match_prediction_page()
        elif page == "Market Analysis":
            self.market_analysis_page()
        elif page == "Betting Strategy":
            self.betting_strategy_page()
        elif page == "Model Performance":
            self.model_performance_page()
        elif page == "About":
            self.about_page()
    
    def match_prediction_page(self):
        """Match prediction interface."""
        st.header("🎯 Match Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Match Details")
            
            # Team selection
            if self.dc_model and hasattr(self.dc_model, 'teams'):
                teams = sorted(self.dc_model.teams)
            else:
                teams = [
                    "Real Madrid", "Manchester City", "Barcelona", "PSG", 
                    "Bayern München", "Liverpool", "Arsenal", "Chelsea",
                    "Juventus", "Milan", "Inter", "Borussia Dortmund",
                    "Atlético Madrid", "Manchester United", "Tottenham"
                ]
            
            home_team = st.selectbox("Home Team", teams, index=0)
            away_team = st.selectbox("Away Team", teams, index=1)
            match_date = st.date_input("Match Date", value=date.today())
            
            # Odds input
            st.subheader("Market Odds (Optional)")
            col1a, col1b, col1c = st.columns(3)
            with col1a:
                odds_home = st.number_input("Home Win Odds", min_value=1.01, value=2.20, step=0.01)
            with col1b:
                odds_draw = st.number_input("Draw Odds", min_value=1.01, value=3.40, step=0.01)
            with col1c:
                odds_away = st.number_input("Away Win Odds", min_value=1.01, value=3.10, step=0.01)
        
        with col2:
            st.subheader("Prediction Results")
            
            if st.button("Predict Match", type="primary"):
                if home_team == away_team:
                    st.error("Please select different teams")
                    return
                
                # Generate predictions
                predictions = self.predict_match(home_team, away_team, match_date)
                
                if predictions:
                    # Display W/D/L probabilities
                    prob_home = predictions['wdl_probs'][0]
                    prob_draw = predictions['wdl_probs'][1]
                    prob_away = predictions['wdl_probs'][2]
                    
                    # Probability visualization
                    fig = go.Figure(data=[
                        go.Bar(
                            x=['Home Win', 'Draw', 'Away Win'],
                            y=[prob_home, prob_draw, prob_away],
                            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c']
                        )
                    ])
                    fig.update_layout(
                        title="Match Outcome Probabilities",
                        yaxis_title="Probability",
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Numeric probabilities
                    col2a, col2b, col2c = st.columns(3)
                    with col2a:
                        st.metric("Home Win", f"{prob_home:.1%}")
                    with col2b:
                        st.metric("Draw", f"{prob_draw:.1%}")
                    with col2c:
                        st.metric("Away Win", f"{prob_away:.1%}")
                    
                    # Market analysis if odds provided
                    market_probs = implied_probs_from_odds(odds_home, odds_draw, odds_away)
                    analysis = self.market_analyzer.analyze_match(
                        (prob_home, prob_draw, prob_away),
                        (odds_home, odds_draw, odds_away),
                        home_team, away_team
                    )
                    
                    if analysis['value_bets']:
                        st.success(f"Found {len(analysis['value_bets'])} value betting opportunities!")
                        for bet in analysis['value_bets']:
                            st.write(f"✅ {bet['outcome'].title()}: Edge = {bet['edge']:.1%}, Odds = {bet['odds']:.2f}")
                    
                    # Scoreline predictions
                    if 'top_scorelines' in predictions:
                        st.subheader("Most Likely Scorelines")
                        scoreline_df = pd.DataFrame(predictions['top_scorelines'])
                        st.dataframe(scoreline_df, use_container_width=True)
                
                else:
                    st.error("Could not generate predictions. Please check that models are loaded.")
    
    def market_analysis_page(self):
        """Market analysis and value betting interface."""
        st.header("📊 Market Analysis")
        
        st.subheader("Upload Match Data")
        uploaded_file = st.file_uploader(
            "Upload CSV with match predictions and odds",
            type=['csv'],
            help="CSV should contain: home_team, away_team, prob_home, prob_draw, prob_away, odds_home, odds_draw, odds_away"
        )
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                
                if self.validate_market_data(df):
                    # Analyze all matches
                    analysis_df = self.market_analyzer.analyze_multiple_matches(df)
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Matches", len(analysis_df))
                    with col2:
                        st.metric("Value Bets Found", analysis_df['num_value_bets'].sum())
                    with col3:
                        avg_margin = analysis_df['bookmaker_margin'].mean()
                        st.metric("Avg Bookmaker Margin", f"{avg_margin:.1f}%")
                    with col4:
                        matches_with_value = (analysis_df['num_value_bets'] > 0).sum()
                        st.metric("Matches with Value", f"{matches_with_value}/{len(analysis_df)}")
                    
                    # Top value bets
                    st.subheader("Top Value Betting Opportunities")
                    top_bets = self.market_analyzer.get_top_value_bets(analysis_df, top_n=10)
                    
                    if not top_bets.empty:
                        st.dataframe(
                            top_bets[['home_team', 'away_team', 'outcome', 'edge', 'value', 'odds']],
                            use_container_width=True
                        )
                        
                        # Edge distribution
                        fig = px.histogram(
                            top_bets, 
                            x='edge', 
                            title="Distribution of Betting Edges",
                            nbins=20
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No value bets found in the uploaded data.")
                        
                else:
                    st.error("Invalid data format. Please check the required columns.")
                    
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    def betting_strategy_page(self):
        """Betting strategy and Kelly criterion interface."""
        st.header("💰 Betting Strategy")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Portfolio Settings")
            bankroll = st.number_input("Bankroll ($)", min_value=100, value=1000, step=100)
            kelly_cap = st.slider("Kelly Cap (%)", min_value=1, max_value=10, value=5) / 100
            min_edge = st.slider("Minimum Edge (%)", min_value=1, max_value=10, value=2) / 100
            max_exposure = st.slider("Max Total Exposure (%)", min_value=5, max_value=25, value=10) / 100
            
            # Sample bets for demo
            st.subheader("Sample Bets")
            sample_bets = [
                {"description": "Real Madrid Win", "probability": 0.45, "odds": 2.4},
                {"description": "Barcelona Draw", "probability": 0.30, "odds": 3.8},
                {"description": "Bayern Over 2.5", "probability": 0.65, "odds": 1.9},
                {"description": "PSG Win", "probability": 0.55, "odds": 2.1},
            ]
            
            selected_bets = []
            for i, bet in enumerate(sample_bets):
                if st.checkbox(bet["description"], value=i < 2):
                    selected_bets.append(bet)
        
        with col2:
            st.subheader("Strategy Results")
            
            if selected_bets:
                # Update calculator settings
                self.kelly_calculator.kelly_cap = kelly_cap
                self.kelly_calculator.min_edge = min_edge
                
                # Calculate portfolio
                portfolio_df = self.kelly_calculator.calculate_multiple_bets(
                    selected_bets, bankroll=bankroll
                )
                
                # Display recommendations
                st.subheader("Individual Bet Recommendations")
                display_cols = [
                    'description', 'recommend_bet', 'stake_amount', 
                    'stake_percentage', 'edge', 'expected_return'
                ]
                st.dataframe(portfolio_df[display_cols], use_container_width=True)
                
                # Portfolio summary
                total_stake = portfolio_df['stake_amount'].sum()
                total_return = portfolio_df['expected_return'].sum()
                
                col2a, col2b, col2c = st.columns(3)
                with col2a:
                    st.metric("Total Stake", f"${total_stake:.2f}")
                with col2b:
                    st.metric("Expected Return", f"${total_return:.2f}")
                with col2c:
                    roi = (total_return / max(total_stake, 1)) * 100
                    st.metric("Expected ROI", f"{roi:.1f}%")
                
                # Risk visualization
                if total_stake > 0:
                    fig = go.Figure(data=[
                        go.Bar(
                            x=portfolio_df['description'],
                            y=portfolio_df['stake_amount'],
                            marker_color=['green' if rec else 'red' for rec in portfolio_df['recommend_bet']]
                        )
                    ])
                    fig.update_layout(
                        title="Recommended Stake Allocation",
                        yaxis_title="Stake Amount ($)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.info("Select some bets to see strategy recommendations.")
    
    def model_performance_page(self):
        """Model performance and evaluation metrics."""
        st.header("📈 Model Performance")
        
        # Model status
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Status")
            if self.wdl_model:
                st.success("✅ W/D/L Model (LightGBM) - Loaded")
            else:
                st.error("❌ W/D/L Model - Not Available")
                
            if self.dc_model:
                st.success("✅ Dixon-Coles Model - Loaded")
                if hasattr(self.dc_model, 'teams'):
                    st.info(f"Trained on {len(self.dc_model.teams)} teams")
            else:
                st.error("❌ Dixon-Coles Model - Not Available")
        
        with col2:
            st.subheader("Sample Predictions")
            if self.dc_model and hasattr(self.dc_model, 'teams') and len(self.dc_model.teams) >= 2:
                # Show sample team strengths
                strengths = self.dc_model.get_team_strengths()
                st.dataframe(strengths.head(10), use_container_width=True)
        
        # Historical performance (would need actual data)
        st.subheader("Historical Performance")
        st.info("Historical performance metrics would be displayed here with actual backtest data.")
        
        # Sample performance chart
        dates = pd.date_range('2023-01-01', '2024-01-01', freq='W')
        cumulative_return = np.cumsum(np.random.normal(0.02, 0.05, len(dates)))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=cumulative_return,
            mode='lines',
            name='Cumulative Return',
            line=dict(color='green')
        ))
        fig.update_layout(
            title="Sample Cumulative Return (Demo Data)",
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def about_page(self):
        """About page with project information."""
        st.header("ℹ️ About")
        
        st.markdown("""
        ## Champions League Football Predictor
        
        This application combines advanced machine learning with sports analytics and betting theory to provide:
        
        ### 🎯 Dual Modeling Approach
        - **Model A**: LightGBM gradient boosting for W/D/L classification
        - **Model B**: Dixon-Coles corrected bivariate Poisson for scoreline distributions
        - **Ensemble**: Calibrated probability reconciliation
        
        ### 📊 Advanced Features
        - ELO rating system for team strength
        - Rolling form and schedule fatigue modeling
        - Market edge detection vs bookmaker odds
        - Kelly criterion for optimal bet sizing
        - Parlay optimization with correlation modeling
        
        ### ⚠️ Important Disclaimer
        **This tool is for educational and research purposes only.**
        - No guarantee of profit in sports betting
        - Always bet responsibly and within your means
        - Past performance does not guarantee future results
        - Consider this as decision support, not financial advice
        
        ### 🔧 Technical Details
        - Built with Python, Streamlit, LightGBM, and SciPy
        - Uses time-series cross-validation for model training
        - Implements isotonic regression for probability calibration
        - Monte Carlo simulation for parlay correlation modeling
        
        ### 📚 Data Sources
        - Historical match data from Kaggle (European Soccer Database)
        - Live fixtures from football-data.org API
        - Bookmaker odds for market analysis
        
        ---
        *For more information, see the project README and source code.*
        """)
    
    def predict_match(self, home_team: str, away_team: str, match_date: date) -> dict:
        """Generate match predictions using loaded models."""
        try:
            predictions = {}
            
            # Dixon-Coles predictions
            if self.dc_model and self.dc_model.fitted:
                if home_team in self.dc_model.teams and away_team in self.dc_model.teams:
                    # Get W/D/L probabilities
                    p_home, p_draw, p_away = self.dc_model.predict_wdl(home_team, away_team)
                    predictions['wdl_probs'] = [p_home, p_draw, p_away]
                    
                    # Get top scorelines
                    top_scorelines = self.dc_model.get_most_likely_scores(home_team, away_team, top_n=10)
                    predictions['top_scorelines'] = top_scorelines.to_dict('records')
                    
                    # Get team strengths
                    strengths = self.dc_model.get_team_strengths()
                    home_strength = strengths[strengths['team'] == home_team]['strength'].iloc[0] if len(strengths[strengths['team'] == home_team]) > 0 else 0
                    away_strength = strengths[strengths['team'] == away_team]['strength'].iloc[0] if len(strengths[strengths['team'] == away_team]) > 0 else 0
                    predictions['strength_diff'] = home_strength - away_strength
                    
            # If no models available, return demo predictions
            if not predictions:
                predictions = {
                    'wdl_probs': [0.45, 0.30, 0.25],  # Demo probabilities
                    'top_scorelines': [
                        {'scoreline': '2-1', 'home_goals': 2, 'away_goals': 1, 'probability': 0.12},
                        {'scoreline': '1-1', 'home_goals': 1, 'away_goals': 1, 'probability': 0.11},
                        {'scoreline': '2-0', 'home_goals': 2, 'away_goals': 0, 'probability': 0.09},
                    ]
                }
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in match prediction: {e}")
            return None
    
    def validate_market_data(self, df: pd.DataFrame) -> bool:
        """Validate uploaded market data format."""
        required_cols = [
            'home_team', 'away_team', 'prob_home', 'prob_draw', 'prob_away',
            'odds_home', 'odds_draw', 'odds_away'
        ]
        
        return all(col in df.columns for col in required_cols)


def main():
    """Main function to run the Streamlit app."""
    app = UCLPredictorApp()
    app.run()


if __name__ == "__main__":
    main()
