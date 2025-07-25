import os
import platform

# Set R environment variables based on operating system
if platform.system() == 'Windows':
    # Windows configuration
    os.environ['R_HOME'] = r'C:\Program Files\R\R-4.3.1'
    os.environ['R_USER'] = os.environ.get('USERNAME', 'default')
    os.environ['R_LIBS_USER'] = os.path.join(os.environ.get('USERPROFILE', ''), 'Documents', 'R', 'win-library', '4.3')
    
    # Add R to PATH for Windows
    r_bin_path = r'C:\Program Files\R\R-4.3.1\bin\x64'
    if r_bin_path not in os.environ['PATH']:
        os.environ['PATH'] = r_bin_path + ';' + os.environ['PATH']
    
    # Set additional environment variables for Windows compatibility
    os.environ['SHELL'] = 'cmd'
    os.environ['COMSPEC'] = 'cmd.exe'
    os.environ['R_ARCH'] = '/x64'
else:
    # Linux/Unix configuration (for GitHub Actions)
    # Check if R_HOME is already set by the workflow, otherwise use default apt-get install location
    if 'R_HOME' not in os.environ:
        os.environ['R_HOME'] = '/usr/lib/R'
    
    # Set LD_LIBRARY_PATH to help rpy2 find libR.so
    if 'LD_LIBRARY_PATH' not in os.environ:
        os.environ['LD_LIBRARY_PATH'] = '/usr/lib/R/lib'
    else:
        os.environ['LD_LIBRARY_PATH'] = f"/usr/lib/R/lib:{os.environ['LD_LIBRARY_PATH']}"
    
    # Set R library paths for Linux
    home_dir = os.path.expanduser('~')
    os.environ['R_LIBS_USER'] = os.path.join(home_dir, 'R', 'libs')

import pandas as pd
from sleeper_helpers import get_league_dict, get_players_df, get_traded_picks, get_user_df
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
    ffs_summarise_week,
)
from import_r_packages import ffscrapr

df_players = get_players_df()

username = "DanCoulton"
season = 2024
sims = 10
current_gameweek = 6

all_leagues = get_league_dict(username, season)
df_matchup_schedule_all_leagues = []
df_summary_week = []
df_summary_season = []
df_draft_picks_all_leagues = []  # For storing draft picks data

for league in all_leagues:
    # if not league["name"] == "NFL Dynasty":
    #     continue
    conn = ffscrapr.sleeper_connect(season=season, league_id=league["league_id"])
    print(league["league_id"])

    # Get draft picks for dynasty leagues
    if "Dynasty" in league["name"]:
        try:
            print(f"Fetching draft picks for {league['name']}")
            traded_picks = get_traded_picks(league["league_id"])
            users_df = get_user_df(league["league_id"])
            
            if not traded_picks.empty:
                # Convert owner_id columns to string to ensure consistent data types
                traded_picks['owner_id'] = traded_picks['owner_id'].astype(str)
                traded_picks['previous_owner_id'] = traded_picks['previous_owner_id'].astype(str)
                users_df['owner_id'] = users_df['owner_id'].astype(str)
                
                # Add league information as first columns
                traded_picks['league_name'] = league["name"]
                traded_picks['league_id'] = league["league_id"]
                
                # Reorder columns to put league_name first
                cols = ['league_name'] + [col for col in traded_picks.columns if col != 'league_name']
                traded_picks = traded_picks[cols]
                
                df_draft_picks_all_leagues.append(traded_picks)
                
                print(f"Added draft picks for {league['name']} to combined dataset")
            else:
                print(f"No traded picks found for {league['name']}")
        except Exception as e:
            print(f"Error fetching draft picks for {league['name']}: {e}")

    df_matchup_schedule, df_rosters, df_matchups = get_matchup_schedule_with_pp(
        league, df_players
    )
    df_matchup_schedule_all_leagues.append(df_matchup_schedule)
    scoring_history = pandas2ri.rpy2py(
        ffscrapr.ff_scoringhistory(conn, season=ro.IntVector(range(2016, 2024)))
    )  # TODO get DST/DEF in

    latest_rankings_draft = ffs_latest_rankings("draft")
    bye_weeks = (
        latest_rankings_draft[["team", "bye"]]
        .drop_duplicates()
        .loc[latest_rankings_draft["team"] != "FA"]
        .reset_index(drop=True)
        .replace(
            {
                "KCC": "KC",
                "TBB": "TB",
                "SFO": "SF",
                "GBP": "GB",
                "LVR": "LV",
                "NOS": "NO",
                "NEP": "NE",
            }
        )
    )
    latest_rankings_weekly = ffs_latest_rankings("week").merge(
        bye_weeks, on="team", how="left"
    )

    ff_rosters = get_ff_rosters(df_rosters)
    lineup_constraints = get_ff_starter_positions(league)
    adp_outcomes = ffs_adp_outcomes(
        scoring_history,
        seasons=range(2016, 2024),
        pos_filter=["QB", "RB", "WR", "TE", "K", "DST"],
    )

    projected_scores = ffs_generate_projections(
        adp_outcomes,
        latest_rankings_draft,
        sims,
        weeks=range(1, df_matchups["gameweek"].max() + 1),
        rosters=ff_rosters,
    )
    roster_scores = ffs_score_rosters(projected_scores, ff_rosters)
    optimal_scores = ffs_optimise_lineups(
        roster_scores,
        lineup_constraints,
        lineup_efficiency_mean=0.775,
        lineup_efficiency_sd=0.05,
        best_ball=False,
        pos_filter=["QB", "RB", "WR", "TE", "K"],  # TODO make this dynamic
    )

    schedules = pd.concat(
        [
            df_matchup_schedule.loc[:, ["gameweek", "roster_id", "opponent_id"]].assign(
                season=i
            )
            for i in range(1, sims + 1)
        ],
        ignore_index=True,
    )
    summary_week = ffs_summarise_week(
        optimal_scores, schedules, df_matchup_schedule, current_gameweek
    )
    summary_week.to_csv(f'data/df_summary_week_{league["name"]}.csv', index=False)

    summary_season = (
        summary_week.groupby(["season", "manager", "roster_id"])
        .agg(h2h_wins=("selected_win", "sum"), points_for=("selected_pts", "sum"))
        .reset_index()
    )
    summary_season["league"] = league["name"]
    df_summary_season.append(summary_season)

df_matchup_schedule_all_leagues = pd.concat(df_matchup_schedule_all_leagues)
df_matchup_schedule_all_leagues.to_csv("data/df_matchup_schedule.csv", index=False)

# Save combined draft picks for all dynasty leagues
if df_draft_picks_all_leagues:
    df_draft_picks_combined = pd.concat(df_draft_picks_all_leagues, ignore_index=True)
    df_draft_picks_combined.to_csv("data/draft_picks.csv", index=False)
    print(f"Saved {len(df_draft_picks_combined)} draft picks from {len(df_draft_picks_all_leagues)} dynasty leagues to draft_picks.csv")
else:
    print("No draft picks data to save")

# df_summary_week = pd.concat(df_summary_week)
# df_summary_week.to_csv('data/df_summary_week.csv', index=False)

df_summary_season = pd.concat(df_summary_season)
df_summary_season.to_csv("data/df_summary_season.csv", index=False)
