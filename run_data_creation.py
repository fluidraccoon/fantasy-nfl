import pandas as pd
from sleeper_helpers import (
    get_league_dict,
    get_players_df
)
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from standings_helpers import get_matchup_schedule_with_pp
from ffs_latest_rankings import (
    ffs_latest_rankings,
    get_ff_rosters,
    get_ff_starter_positions,
    ffs_adp_outcomes,
    ffs_generate_projections,
    ffs_score_rosters,
    ffs_optimise_lineups,
    ffs_summarise_week
)
from import_r_packages import ffscrapr

df_players = get_players_df()

username = "DanCoulton"
season = 2024
sims = 1000
current_gameweek = 13

all_leagues = get_league_dict(username, season)
df_matchup_schedule_all_leagues = []
df_summary_week = []
df_summary_season = []
for league in all_leagues:
    # if not league["name"]=="House of Hards Knocks":
    #     continue
    conn = ffscrapr.sleeper_connect(season = season, league_id = league["league_id"])
    print(league["league_id"])

    df_matchup_schedule, df_rosters, df_matchups = get_matchup_schedule_with_pp(league, df_players)
    df_matchup_schedule_all_leagues.append(df_matchup_schedule)
    scoring_history = pandas2ri.rpy2py(ffscrapr.ff_scoringhistory(conn, season = ro.IntVector(range(2016, 2024)))) # TODO get DST/DEF in
    
    latest_rankings_draft = ffs_latest_rankings("draft")
    bye_weeks = (latest_rankings_draft[['team', 'bye']]
            .drop_duplicates()
            .loc[latest_rankings_draft['team'] != 'FA'].reset_index(drop=True)
            .replace({'KCC': 'KC', 'TBB': 'TB', 'SFO': 'SF', 'GBP': 'GB', 'LVR': 'LV', 'NOS': 'NO', 'NEP': 'NE'}))
    latest_rankings_weekly = ffs_latest_rankings("week").merge(bye_weeks, on="team", how="left")
    
    ff_rosters = get_ff_rosters(df_rosters)
    lineup_constraints = get_ff_starter_positions(league)
    adp_outcomes = ffs_adp_outcomes(scoring_history, seasons = range(2016, 2024), pos_filter=["QB", "RB", "WR", "TE", "K", "DST"])
    
    projected_scores = ffs_generate_projections(adp_outcomes, latest_rankings_draft, sims, weeks=range(1, df_matchups["gameweek"].max()), rosters=ff_rosters)
    roster_scores = ffs_score_rosters(projected_scores, ff_rosters)
    optimal_scores = ffs_optimise_lineups(
        roster_scores, lineup_constraints, lineup_efficiency_mean = 0.775, lineup_efficiency_sd = 0.05,
        best_ball = False, pos_filter = ["QB","RB","WR","TE","K"] # TODO make this dynamic
    )
    
    schedules = pd.concat([df_matchup_schedule.loc[:, ["gameweek", "roster_id", "opponent_id"]].assign(season=i) for i in range(1, sims + 1)], ignore_index=True)
    summary_week = ffs_summarise_week(optimal_scores, schedules, df_matchup_schedule, current_gameweek)
    summary_week.to_csv(f'data/df_summary_week_{league["name"]}.csv', index=False)
    
    summary_season = summary_week.groupby(["season", "manager", "roster_id"]).agg(
        h2h_wins=('selected_win', 'sum'),
        points_for=('selected_pts', 'sum')
    ).reset_index()
    summary_season["league"] = league["name"]
    df_summary_season.append(summary_season)

df_matchup_schedule_all_leagues = pd.concat(df_matchup_schedule_all_leagues)
df_matchup_schedule_all_leagues.to_csv('data/df_matchup_schedule.csv', index=False)

# df_summary_week = pd.concat(df_summary_week)
# df_summary_week.to_csv('data/df_summary_week.csv', index=False)

df_summary_season = pd.concat(df_summary_season)
df_summary_season.to_csv('data/df_summary_season.csv', index=False)