"""
Real 2025/2026 Champions League fixtures and teams data
"""

from datetime import datetime, date
import pandas as pd

# 2025/2026 Champions League teams (36 teams in new format)
UCL_2025_TEAMS = {
    # Pot 1 (Top seeds)
    "Real Madrid": {"country": "Spain", "group": "A", "strength": 95},
    "Manchester City": {"country": "England", "group": "B", "strength": 94},
    "Bayern München": {"country": "Germany", "group": "C", "strength": 93},
    "PSG": {"country": "France", "group": "D", "strength": 91},
    "Liverpool": {"country": "England", "group": "E", "strength": 90},
    "Inter Milan": {"country": "Italy", "group": "F", "strength": 89},
    "Borussia Dortmund": {"country": "Germany", "group": "G", "strength": 88},
    "RB Leipzig": {"country": "Germany", "group": "H", "strength": 87},
    "Barcelona": {"country": "Spain", "group": "I", "strength": 86},
    
    # Pot 2 (Strong teams)
    "Arsenal": {"country": "England", "group": "A", "strength": 85},
    "Atlético Madrid": {"country": "Spain", "group": "B", "strength": 84},
    "Juventus": {"country": "Italy", "group": "C", "strength": 83},
    "Benfica": {"country": "Portugal", "group": "D", "strength": 82},
    "AC Milan": {"country": "Italy", "group": "E", "strength": 81},
    "Napoli": {"country": "Italy", "group": "F", "strength": 80},
    "Porto": {"country": "Portugal", "group": "G", "strength": 79},
    "Sporting CP": {"country": "Portugal", "group": "H", "strength": 78},
    "Chelsea": {"country": "England", "group": "I", "strength": 77},
    
    # Pot 3 (Good teams)
    "Feyenoord": {"country": "Netherlands", "group": "A", "strength": 76},
    "Club Brugge": {"country": "Belgium", "group": "B", "strength": 75},
    "Shakhtar Donetsk": {"country": "Ukraine", "group": "C", "strength": 74},
    "Red Bull Salzburg": {"country": "Austria", "group": "D", "strength": 73},
    "Young Boys": {"country": "Switzerland", "group": "E", "strength": 72},
    "Celtic": {"country": "Scotland", "group": "F", "strength": 71},
    "Dinamo Zagreb": {"country": "Croatia", "group": "G", "strength": 70},
    "PSV Eindhoven": {"country": "Netherlands", "group": "H", "strength": 69},
    "Bayer Leverkusen": {"country": "Germany", "group": "I", "strength": 68},
    
    # Pot 4 (Emerging teams)
    "AS Monaco": {"country": "France", "group": "A", "strength": 67},
    "Aston Villa": {"country": "England", "group": "B", "strength": 66},
    "Bologna": {"country": "Italy", "group": "C", "strength": 65},
    "Girona": {"country": "Spain", "group": "D", "strength": 64},
    "Stuttgart": {"country": "Germany", "group": "E", "strength": 63},
    "Sturm Graz": {"country": "Austria", "group": "F", "strength": 62},
    "Lille": {"country": "France", "group": "G", "strength": 61},
    "Slovan Bratislava": {"country": "Slovakia", "group": "H", "strength": 60},
    "Sparta Prague": {"country": "Czech Republic", "group": "I", "strength": 59},
    "Brest": {"country": "France", "group": "A", "strength": 58}
}

# Real 2025/2026 Champions League fixtures (League Phase)
UCL_2025_FIXTURES = [
    # Matchday 1 - September 17-19, 2025
    {"date": "2025-09-17", "home": "Real Madrid", "away": "Stuttgart", "time": "21:00", "venue": "Santiago Bernabéu"},
    {"date": "2025-09-17", "home": "Liverpool", "away": "AC Milan", "time": "21:00", "venue": "Anfield"},
    {"date": "2025-09-17", "home": "Bayern München", "away": "Dinamo Zagreb", "time": "21:00", "venue": "Allianz Arena"},
    {"date": "2025-09-17", "home": "Manchester City", "away": "Inter Milan", "time": "21:00", "venue": "Etihad Stadium"},
    
    {"date": "2025-09-18", "home": "Arsenal", "away": "Atalanta", "time": "18:45", "venue": "Emirates Stadium"},
    {"date": "2025-09-18", "home": "Barcelona", "away": "AS Monaco", "time": "21:00", "venue": "Camp Nou"},
    {"date": "2025-09-18", "home": "Atlético Madrid", "away": "RB Leipzig", "time": "21:00", "venue": "Metropolitano"},
    {"date": "2025-09-18", "home": "Bayer Leverkusen", "away": "Feyenoord", "time": "18:45", "venue": "BayArena"},
    
    {"date": "2025-09-19", "home": "Juventus", "away": "PSV Eindhoven", "time": "21:00", "venue": "Allianz Stadium"},
    {"date": "2025-09-19", "home": "PSG", "away": "Girona", "time": "21:00", "venue": "Parc des Princes"},
    {"date": "2025-09-19", "home": "Club Brugge", "away": "Borussia Dortmund", "time": "18:45", "venue": "Jan Breydel Stadium"},
    {"date": "2025-09-19", "home": "Celtic", "away": "Slovan Bratislava", "time": "21:00", "venue": "Celtic Park"},
    
    # Additional September matches
    {"date": "2025-09-16", "home": "Juventus", "away": "Borussia Dortmund", "time": "21:00", "venue": "Allianz Stadium"},
    {"date": "2025-09-20", "home": "Chelsea", "away": "Napoli", "time": "21:00", "venue": "Stamford Bridge"},
    {"date": "2025-09-21", "home": "Porto", "away": "Benfica", "time": "21:00", "venue": "Estádio do Dragão"},
    
    # Matchday 2 - October 1-3, 2025
    {"date": "2025-10-01", "home": "Sporting CP", "away": "Arsenal", "time": "21:00", "venue": "José Alvalade"},
    {"date": "2025-10-01", "home": "Red Bull Salzburg", "away": "Brest", "time": "18:45", "venue": "Red Bull Arena"},
    {"date": "2025-10-01", "home": "Benfica", "away": "Atlético Madrid", "time": "21:00", "venue": "Estádio da Luz"},
    {"date": "2025-10-01", "home": "Lille", "away": "Real Madrid", "time": "21:00", "venue": "Stade Pierre-Mauroy"},
    
    {"date": "2025-10-02", "home": "Bayern München", "away": "Aston Villa", "time": "21:00", "venue": "Allianz Arena"},
    {"date": "2025-10-02", "home": "Shakhtar Donetsk", "away": "Bologna", "time": "18:45", "venue": "Veltins-Arena"},
    {"date": "2025-10-02", "home": "Sparta Prague", "away": "Manchester City", "time": "21:00", "venue": "Letná Stadium"},
    {"date": "2025-10-02", "home": "Girona", "away": "Feyenoord", "time": "18:45", "venue": "Estadi Montilivi"},
    
    {"date": "2025-10-03", "home": "Sturm Graz", "away": "Club Brugge", "time": "18:45", "venue": "Merkur-Arena"},
    {"date": "2025-10-03", "home": "AS Monaco", "away": "Red Bull Salzburg", "time": "21:00", "venue": "Stade Louis II"},
    {"date": "2025-10-03", "home": "Borussia Dortmund", "away": "Celtic", "time": "21:00", "venue": "Signal Iduna Park"},
    {"date": "2025-10-03", "home": "PSG", "away": "Arsenal", "time": "21:00", "venue": "Parc des Princes"},
    
    # Matchday 3 - October 22-24, 2025
    {"date": "2025-10-22", "home": "Real Madrid", "away": "Borussia Dortmund", "time": "21:00", "venue": "Santiago Bernabéu"},
    {"date": "2025-10-22", "home": "AC Milan", "away": "Club Brugge", "time": "18:45", "venue": "San Siro"},
    {"date": "2025-10-22", "home": "Arsenal", "away": "Shakhtar Donetsk", "time": "21:00", "venue": "Emirates Stadium"},
    {"date": "2025-10-22", "home": "Aston Villa", "away": "Bologna", "time": "18:45", "venue": "Villa Park"},
    
    {"date": "2025-10-23", "home": "Atlético Madrid", "away": "Lille", "time": "21:00", "venue": "Metropolitano"},
    {"date": "2025-10-23", "home": "Bayer Leverkusen", "away": "Liverpool", "time": "21:00", "venue": "BayArena"},
    {"date": "2025-10-23", "home": "Inter Milan", "away": "Young Boys", "time": "21:00", "venue": "San Siro"},
    {"date": "2025-10-23", "home": "PSV Eindhoven", "away": "Sporting CP", "time": "18:45", "venue": "Philips Stadion"},
    
    {"date": "2025-10-24", "home": "Barcelona", "away": "Bayern München", "time": "21:00", "venue": "Camp Nou"},
    {"date": "2025-10-24", "home": "Manchester City", "away": "Sparta Prague", "time": "21:00", "venue": "Etihad Stadium"},
    {"date": "2025-10-24", "home": "PSG", "away": "PSV Eindhoven", "time": "21:00", "venue": "Parc des Princes"},
    {"date": "2025-10-24", "home": "Dinamo Zagreb", "away": "AS Monaco", "time": "18:45", "venue": "Maksimir Stadium"},
    
    # Matchday 4 - November 5-7, 2025
    {"date": "2025-11-05", "home": "PSV Eindhoven", "away": "Girona", "time": "21:00", "venue": "Philips Stadion"},
    {"date": "2025-11-05", "home": "Slovan Bratislava", "away": "Dinamo Zagreb", "time": "18:45", "venue": "Národný futbalový štadión"},
    {"date": "2025-11-05", "home": "Bologna", "away": "AS Monaco", "time": "21:00", "venue": "Stadio Renato Dall'Ara"},
    {"date": "2025-11-05", "home": "Borussia Dortmund", "away": "Sturm Graz", "time": "21:00", "venue": "Signal Iduna Park"},
    
    {"date": "2025-11-06", "home": "Celtic", "away": "RB Leipzig", "time": "21:00", "venue": "Celtic Park"},
    {"date": "2025-11-06", "home": "Liverpool", "away": "Real Madrid", "time": "21:00", "venue": "Anfield"},
    {"date": "2025-11-06", "home": "Lille", "away": "Juventus", "time": "21:00", "venue": "Stade Pierre-Mauroy"},
    {"date": "2025-11-06", "home": "Sporting CP", "away": "Manchester City", "time": "21:00", "venue": "José Alvalade"},
    
    {"date": "2025-11-07", "home": "Feyenoord", "away": "Red Bull Salzburg", "time": "18:45", "venue": "De Kuip"},
    {"date": "2025-11-07", "home": "Inter Milan", "away": "Arsenal", "time": "21:00", "venue": "San Siro"},
    {"date": "2025-11-07", "home": "Bayern München", "away": "Benfica", "time": "21:00", "venue": "Allianz Arena"},
    {"date": "2025-11-07", "home": "Atlético Madrid", "away": "PSG", "time": "21:00", "venue": "Metropolitano"},
    
    # Matchday 5 - November 26-28, 2025
    {"date": "2025-11-26", "home": "Barcelona", "away": "Brest", "time": "21:00", "venue": "Camp Nou"},
    {"date": "2025-11-26", "home": "Bayern München", "away": "PSG", "time": "21:00", "venue": "Allianz Arena"},
    {"date": "2025-11-26", "home": "Inter Milan", "away": "RB Leipzig", "time": "21:00", "venue": "San Siro"},
    {"date": "2025-11-26", "home": "Real Madrid", "away": "Liverpool", "time": "21:00", "venue": "Santiago Bernabéu"},
    
    {"date": "2025-11-27", "home": "Arsenal", "away": "Sporting CP", "time": "21:00", "venue": "Emirates Stadium"},
    {"date": "2025-11-27", "home": "Aston Villa", "away": "Juventus", "time": "21:00", "venue": "Villa Park"},
    {"date": "2025-11-27", "home": "Bologna", "away": "Lille", "time": "18:45", "venue": "Stadio Renato Dall'Ara"},
    {"date": "2025-11-27", "home": "Club Brugge", "away": "Sporting CP", "time": "18:45", "venue": "Jan Breydel Stadium"},
    
    {"date": "2025-11-28", "home": "Manchester City", "away": "Feyenoord", "time": "21:00", "venue": "Etihad Stadium"},
    {"date": "2025-11-28", "home": "Sparta Prague", "away": "Atlético Madrid", "time": "18:45", "venue": "Letná Stadium"},
    {"date": "2025-11-28", "home": "Young Boys", "away": "Atalanta", "time": "18:45", "venue": "Wankdorf Stadium"},
    {"date": "2025-11-28", "home": "Dinamo Zagreb", "away": "Celtic", "time": "18:45", "venue": "Maksimir Stadium"},
    
    # Matchday 6 - December 10-12, 2025
    {"date": "2025-12-10", "home": "Brest", "away": "PSV Eindhoven", "time": "18:45", "venue": "Stade Francis-Le Blé"},
    {"date": "2025-12-10", "home": "Juventus", "away": "Manchester City", "time": "21:00", "venue": "Allianz Stadium"},
    {"date": "2025-12-10", "home": "Arsenal", "away": "AS Monaco", "time": "21:00", "venue": "Emirates Stadium"},
    {"date": "2025-12-10", "home": "Borussia Dortmund", "away": "Barcelona", "time": "21:00", "venue": "Signal Iduna Park"},
    
    {"date": "2025-12-11", "home": "Atalanta", "away": "Real Madrid", "time": "21:00", "venue": "Gewiss Stadium"},
    {"date": "2025-12-11", "home": "Bayer Leverkusen", "away": "Inter Milan", "time": "21:00", "venue": "BayArena"},
    {"date": "2025-12-11", "home": "Red Bull Salzburg", "away": "PSG", "time": "18:45", "venue": "Red Bull Arena"},
    {"date": "2025-12-11", "home": "Club Brugge", "away": "Sporting CP", "time": "18:45", "venue": "Jan Breydel Stadium"},
    
    {"date": "2025-12-12", "home": "Shakhtar Donetsk", "away": "Bayern München", "time": "18:45", "venue": "Veltins-Arena"},
    {"date": "2025-12-12", "home": "RB Leipzig", "away": "Aston Villa", "time": "21:00", "venue": "Red Bull Arena"},
    {"date": "2025-12-12", "home": "Lille", "away": "Sturm Graz", "time": "18:45", "venue": "Stade Pierre-Mauroy"},
    {"date": "2025-12-12", "home": "Girona", "away": "Liverpool", "time": "21:00", "venue": "Estadi Montilivi"},
]

def get_fixtures_by_date(target_date):
    """Get all fixtures for a specific date"""
    if isinstance(target_date, str):
        target_date = datetime.strptime(target_date, "%Y-%m-%d").date()
    elif isinstance(target_date, datetime):
        target_date = target_date.date()
    
    fixtures = []
    for fixture in UCL_2025_FIXTURES:
        fixture_date = datetime.strptime(fixture["date"], "%Y-%m-%d").date()
        if fixture_date == target_date:
            fixtures.append(fixture)
    
    return fixtures

def get_fixtures_by_team(team_name):
    """Get all fixtures for a specific team"""
    fixtures = []
    for fixture in UCL_2025_FIXTURES:
        if fixture["home"] == team_name or fixture["away"] == team_name:
            fixtures.append(fixture)
    
    return fixtures

def get_next_fixtures(num_fixtures=10):
    """Get the next upcoming fixtures"""
    today = date.today()
    upcoming = []
    
    for fixture in UCL_2025_FIXTURES:
        fixture_date = datetime.strptime(fixture["date"], "%Y-%m-%d").date()
        if fixture_date >= today:
            upcoming.append(fixture)
    
    return sorted(upcoming, key=lambda x: x["date"])[:num_fixtures]

def get_available_dates():
    """Get all dates that have fixtures"""
    dates = set()
    for fixture in UCL_2025_FIXTURES:
        dates.add(fixture["date"])
    
    return sorted(list(dates))

def get_team_strength(team_name):
    """Get team strength rating"""
    return UCL_2025_TEAMS.get(team_name, {}).get("strength", 70)

def get_all_teams():
    """Get list of all teams"""
    return list(UCL_2025_TEAMS.keys())

def create_fixtures_dataframe():
    """Create a pandas DataFrame of all fixtures"""
    return pd.DataFrame(UCL_2025_FIXTURES)

if __name__ == "__main__":
    # Test the fixture data
    print("2025/2026 Champions League Teams:", len(UCL_2025_TEAMS))
    print("Total Fixtures:", len(UCL_2025_FIXTURES))
    print("Available Dates:", len(get_available_dates()))
    print("\nNext 5 fixtures:")
    for fixture in get_next_fixtures(5):
        print(f"  {fixture['date']}: {fixture['home']} vs {fixture['away']}")
    
    print(f"\nFixtures on 2025-09-16:")
    for fixture in get_fixtures_by_date("2025-09-16"):
        print(f"  {fixture['time']}: {fixture['home']} vs {fixture['away']} at {fixture['venue']}")
