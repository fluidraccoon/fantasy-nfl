import pandas as pd
from sleeper_helpers import (
    get_league_dict,
    get_players_df
)
from standings_helpers import (
    get_matchup_schedule_with_pp
)

df_players = get_players_df()

username = "DanCoulton"
season = 2024
gameweek_end = 1

all_leagues = get_league_dict(username, season)
df_matchup_schedule = []
for league in all_leagues:
    df_matchup_schedule.append(get_matchup_schedule_with_pp(league, df_players, gameweek_end))

df_matchup_schedule = pd.concat(df_matchup_schedule)
df_matchup_schedule.to_csv('data/df_matchup_schedule.csv', index=False)