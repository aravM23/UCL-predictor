import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
import os
from pathlib import Path
import math
from scipy.stats import poisson
from ucl_fixtures_2025 import UCL_2025_TEAMS, UCL_2025_FIXTURES

# Set page config
st.set_page_config(
    page_title="Champions League Betting Assistant",
    page_icon="⚽",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e3d59;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #007bff;
    }
</style>
""", unsafe_allow_html=True)

def load_sample_data():
    """Load sample data if available"""
    try:
        matches_file = Path("data/raw/sample_matches.csv")
        fixtures_file = Path("data/raw/sample_fixtures.csv")
        
        if matches_file.exists():
            matches = pd.read_csv(matches_file)
            return matches
        else:
            return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Helper functions for UCL 2025 fixtures
def get_available_dates():
    """Get all available match dates from UCL 2025 fixtures"""
    dates = set()
    for fixture in UCL_2025_FIXTURES:
        dates.add(fixture['date'])
    return sorted(list(dates))

def get_fixtures_by_date(date):
    """Get all fixtures for a specific date"""
    target_date = date.strftime("%Y-%m-%d")
    fixtures = []
    
    for fixture in UCL_2025_FIXTURES:
        if fixture['date'] == target_date:
            fixtures.append(fixture)
    
    return fixtures

def get_next_fixtures(limit=10):
    """Get next upcoming fixtures"""
    today = datetime.now().date()
    upcoming_fixtures = []
    
    for fixture in UCL_2025_FIXTURES:
        fixture_date = datetime.strptime(fixture['date'], "%Y-%m-%d").date()
        if fixture_date >= today:
            upcoming_fixtures.append(fixture)
    
    # Sort by date
    upcoming_fixtures.sort(key=lambda x: x['date'])
    return upcoming_fixtures[:limit]

def get_team_strength(team_name):
    """Get team strength rating from UCL 2025 teams"""
    return UCL_2025_TEAMS.get(team_name, {}).get('strength', 70)

def simulate_prediction(home_team, away_team):
    """Generate realistic prediction based on team strengths and historical performance"""
    # Get team strengths from real UCL 2025 data
    home_strength = get_team_strength(home_team)
    away_strength = get_team_strength(away_team)
    
    # Base probabilities with home advantage
    home_advantage = 0.1
    
    # Convert strength to goal expectancy (simplified model)
    home_attack = (home_strength / 100) * 1.5 + home_advantage
    away_attack = (away_strength / 100) * 1.2
    
    # Adjust for defensive quality
    home_defense = 2.0 - (away_strength / 100) * 0.8
    away_defense = 2.0 - (home_strength / 100) * 0.8
    
    # Calculate expected goals using simplified model
    home_xg = max(0.3, min(4.0, home_attack / away_defense))
    away_xg = max(0.3, min(4.0, away_attack / home_defense))
    
    # Generate scoreline probabilities using Poisson distribution
    scorelines = []
    total_prob = 0
    
    for home_goals in range(6):
        for away_goals in range(6):
            # Poisson probability
            prob_home = (home_xg ** home_goals) * np.exp(-home_xg) / math.factorial(home_goals)
            prob_away = (away_xg ** away_goals) * np.exp(-away_xg) / math.factorial(away_goals)
            prob = prob_home * prob_away
            
            if prob > 0.005:  # Only include probable scorelines
                scorelines.append({
                    'scoreline': f"{home_goals}-{away_goals}",
                    'probability': prob,
                    'home_goals': home_goals,
                    'away_goals': away_goals
                })
                total_prob += prob
    
    # Normalize probabilities
    for s in scorelines:
        s['probability'] /= total_prob
    
    # Sort by probability
    scorelines = sorted(scorelines, key=lambda x: x['probability'], reverse=True)[:10]
    
    # Calculate W/D/L probabilities
    home_prob = sum(s['probability'] for s in scorelines if s['home_goals'] > s['away_goals'])
    draw_prob = sum(s['probability'] for s in scorelines if s['home_goals'] == s['away_goals'])
    away_prob = sum(s['probability'] for s in scorelines if s['home_goals'] < s['away_goals'])
    
    # Normalize W/D/L probabilities
    total_wdl = home_prob + draw_prob + away_prob
    home_prob /= total_wdl
    draw_prob /= total_wdl
    away_prob /= total_wdl
    
    return {
        'home_prob': home_prob,
        'draw_prob': draw_prob,
        'away_prob': away_prob,
        'scorelines': scorelines,
        'home_xg': home_xg,
        'away_xg': away_xg,
        'home_strength': home_strength,
        'away_strength': away_strength
    }

def main():
    # Header
    st.markdown('<h1 class="main-header">⚽ Champions League Betting Assistant 2025/26</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎯 Real UCL Fixtures")
    
    # Load teams from real UCL 2025 data
    teams = list(UCL_2025_TEAMS.keys())
    available_dates = get_available_dates()
    
    st.sidebar.success(f"✅ {len(teams)} UCL 2025/26 teams loaded")
    st.sidebar.info(f"📅 {len(available_dates)} match dates available")
    
    # Date selection mode
    prediction_mode = st.sidebar.radio(
        "🔍 Selection Mode:",
        ["📅 Browse by Date", "🆚 Manual Team Selection"]
    )
    
    selected_fixture = None
    home_team = None
    away_team = None
    match_date = None
    
    if prediction_mode == "📅 Browse by Date":
        # Date selection
        st.sidebar.subheader("📅 Select Match Date")
        
        # Convert string dates to date objects for the selectbox
        date_options = []
        for date_str in available_dates:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
            date_options.append(date_obj)
        
        if date_options:
            selected_date = st.sidebar.selectbox(
                "Choose a date:",
                date_options,
                format_func=lambda x: x.strftime("%B %d, %Y (%A)")
            )
            
            # Get fixtures for selected date
            fixtures_on_date = get_fixtures_by_date(selected_date)
            
            if fixtures_on_date:
                st.sidebar.subheader(f"⚽ Matches on {selected_date.strftime('%B %d, %Y')}")
                
                # Create fixture options
                fixture_options = []
                for i, fixture in enumerate(fixtures_on_date):
                    venue_short = fixture['venue'].split(' ')[0] if ' ' in fixture['venue'] else fixture['venue']
                    option_text = f"{fixture['home']} vs {fixture['away']} ({fixture['time']} at {venue_short})"
                    fixture_options.append(option_text)
                
                selected_fixture_idx = st.sidebar.selectbox(
                    "Choose match:",
                    range(len(fixture_options)),
                    format_func=lambda x: fixture_options[x]
                )
                
                selected_fixture = fixtures_on_date[selected_fixture_idx]
                home_team = selected_fixture['home']
                away_team = selected_fixture['away']
                match_date = selected_date
                
                # Display fixture details
                st.sidebar.markdown(f"""
                **Match Details:**
                - 🏠 **Home:** {home_team}
                - ✈️ **Away:** {away_team}
                - ⏰ **Time:** {selected_fixture['time']}
                - 🏟️ **Venue:** {selected_fixture['venue']}
                """)
            else:
                st.sidebar.warning("No matches scheduled for this date")
    
    else:
        # Manual team selection
        st.sidebar.subheader("🆚 Manual Selection")
        home_team = st.sidebar.selectbox("🏠 Home Team", teams, index=0)
        away_team = st.sidebar.selectbox("✈️ Away Team", teams, index=1)
        match_date = st.sidebar.date_input("📅 Match Date", datetime.now() + timedelta(days=7))
    
    # Prediction button
    if st.sidebar.button("🔮 Generate Prediction", type="primary"):
        if home_team and away_team and home_team != away_team:
            # Main content area
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Match header
                if selected_fixture:
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h2>🆚 {home_team} vs {away_team}</h2>
                        <p>📅 {match_date.strftime('%B %d, %Y')} at {selected_fixture['time']}</p>
                        <p>🏟️ {selected_fixture['venue']}</p>
                        <p>🏆 UEFA Champions League 2025/26</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h2>🆚 {home_team} vs {away_team}</h2>
                        <p>📅 {match_date.strftime('%B %d, %Y')}</p>
                        <p>🏆 UEFA Champions League 2025/26</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Generate prediction
                with st.spinner("🤖 Analyzing teams and generating predictions..."):
                    prediction = simulate_prediction(home_team, away_team)
                
                # Win probabilities
                st.subheader("🎯 Win Probabilities")
                prob_col1, prob_col2, prob_col3 = st.columns(3)
                
                with prob_col1:
                    st.metric(
                        f"🏠 {home_team}", 
                        f"{prediction['home_prob']:.1%}",
                        delta=f"Fair Odds: {1/prediction['home_prob']:.2f}"
                    )
                
                with prob_col2:
                    st.metric(
                        "🤝 Draw", 
                        f"{prediction['draw_prob']:.1%}",
                        delta=f"Fair Odds: {1/prediction['draw_prob']:.2f}"
                    )
                
                with prob_col3:
                    st.metric(
                        f"✈️ {away_team}", 
                        f"{prediction['away_prob']:.1%}",
                        delta=f"Fair Odds: {1/prediction['away_prob']:.2f}"
                    )
                
                # Probability chart
                fig = px.bar(
                    x=[f'{home_team} Win', 'Draw', f'{away_team} Win'],
                    y=[prediction['home_prob'], prediction['draw_prob'], prediction['away_prob']],
                    color=['#4CAF50', '#FFC107', '#F44336'],
                    title="Win Probability Distribution",
                    labels={'x': 'Outcome', 'y': 'Probability'}
                )
                fig.update_layout(showlegend=False, yaxis_title="Probability")
                st.plotly_chart(fig, use_container_width=True)
                
                # Expected Goals
                st.subheader("⚽ Expected Goals")
                xg_col1, xg_col2 = st.columns(2)
                with xg_col1:
                    st.metric(f"{home_team} xG", f"{prediction['home_xg']:.2f}")
                with xg_col2:
                    st.metric(f"{away_team} xG", f"{prediction['away_xg']:.2f}")
                
                # Most likely scorelines
                st.subheader("📊 Most Likely Scorelines")
                
                scoreline_data = []
                for s in prediction['scorelines']:
                    scoreline_data.append({
                        'Scoreline': s['scoreline'],
                        'Probability': f"{s['probability']:.1%}",
                        'Prob_Value': s['probability']
                    })
                
                scoreline_df = pd.DataFrame(scoreline_data)
                
                # Scoreline chart
                fig_scorelines = px.bar(
                    scoreline_df, 
                    x='Scoreline', 
                    y='Prob_Value',
                    title="Scoreline Probabilities",
                    labels={'Prob_Value': 'Probability'},
                    color='Prob_Value',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig_scorelines, use_container_width=True)
                
                # Scoreline table
                st.dataframe(scoreline_df[['Scoreline', 'Probability']], use_container_width=True)
            
            with col2:
                st.subheader("📊 Match Analysis")
                
                # Team strengths
                st.markdown("### ⭐ Team Ratings")
                home_rating = prediction['home_strength']
                away_rating = prediction['away_strength']
                
                st.metric("🏠 Home Team", f"{home_rating}/100")
                st.metric("✈️ Away Team", f"{away_rating}/100")
                
                # Strength comparison chart
                fig_strength = px.bar(
                    x=[home_team, away_team],
                    y=[home_rating, away_rating],
                    title="Team Strength Comparison",
                    color=[home_rating, away_rating],
                    color_continuous_scale='RdYlGn'
                )
                fig_strength.update_layout(showlegend=False)
                st.plotly_chart(fig_strength, use_container_width=True)
                
                # Market analysis
                st.markdown("### 💰 Betting Analysis")
                
                # Simulated market odds (with bookmaker margin)
                margin = 1.08  # 8% margin
                market_home = (1/prediction['home_prob']) * margin
                market_draw = (1/prediction['draw_prob']) * margin
                market_away = (1/prediction['away_prob']) * margin
                
                st.markdown(f"""
                **Estimated Market Odds:**
                - 🏠 Home: {market_home:.2f}
                - 🤝 Draw: {market_draw:.2f}
                - ✈️ Away: {market_away:.2f}
                
                **Fair Value Odds:**
                - 🏠 Home: {1/prediction['home_prob']:.2f}
                - 🤝 Draw: {1/prediction['draw_prob']:.2f}
                - ✈️ Away: {1/prediction['away_prob']:.2f}
                """)
                
                # Value detection
                edge_home = (market_home - 1/prediction['home_prob']) / market_home
                edge_draw = (market_draw - 1/prediction['draw_prob']) / market_draw
                edge_away = (market_away - 1/prediction['away_prob']) / market_away
                
                if edge_home > 0.05:
                    st.success(f"✅ Value bet: {home_team} win (+{edge_home:.1%} edge)")
                if edge_away > 0.05:
                    st.success(f"✅ Value bet: {away_team} win (+{edge_away:.1%} edge)")
                if edge_draw > 0.05:
                    st.success(f"✅ Value bet: Draw (+{edge_draw:.1%} edge)")
                
                # Team info
                st.markdown("### � Team Information")
                home_info = UCL_2025_TEAMS.get(home_team, {})
                away_info = UCL_2025_TEAMS.get(away_team, {})
                
                st.markdown(f"""
                **{home_team}**
                - Country: {home_info.get('country', 'Unknown')}
                - Strength: {home_info.get('strength', 70)}/100
                
                **{away_team}**
                - Country: {away_info.get('country', 'Unknown')}  
                - Strength: {away_info.get('strength', 70)}/100
                """)
        else:
            st.sidebar.error("⚠️ Please select different teams")
    
    # Show upcoming fixtures
    st.markdown("---")
    st.subheader("📅 Upcoming UCL Fixtures")
    
    upcoming_fixtures = get_next_fixtures(10)
    if upcoming_fixtures:
        upcoming_df = pd.DataFrame(upcoming_fixtures)
        upcoming_df['Match'] = upcoming_df['home'] + ' vs ' + upcoming_df['away']
        upcoming_df['Date'] = pd.to_datetime(upcoming_df['date']).dt.strftime('%B %d, %Y')
        
        display_df = upcoming_df[['Date', 'Match', 'time', 'venue']].copy()
        display_df.columns = ['Date', 'Match', 'Time', 'Venue']
        
        st.dataframe(display_df, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>⚽ Champions League Betting Assistant 2025/26 - Real Fixtures & Teams</p>
        <p>Predictions based on team strength analysis and statistical modeling</p>
        <p>🔬 This is for educational purposes - always bet responsibly</p>
        <br>
        <p style='font-weight: bold; color: #444;'>Made by Arav Mathur 😎</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
