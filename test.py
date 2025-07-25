from sleeper_helpers import get_league_dict, get_players_df, get_traded_picks, get_user_df
from import_r_packages import ffscrapr

all_leagues = get_league_dict("DanCoulton", 2025)

for league in all_leagues:
    conn = ffscrapr.sleeper_connect(season=2025, league_id=league["league_id"])

    traded_picks = get_traded_picks(league["league_id"])
    print(type(traded_picks))