import pandas as pd
import numpy as np
import requests
from io import StringIO
import requests
import re
import itertools
import os
from scoring_history import dp_playerids
from import_r_packages import ffscrapr
from rpy2.robjects import pandas2ri, r
from scipy.stats import norm, binom
from scipy.optimize import linprog

def load_ff_rankings(rank_type="draft"):
    # Ensure the argument is one of the allowed values
    if rank_type not in ["draft", "week", "all"]:
        raise ValueError("Type must be one of 'draft', 'week', or 'all'")

    url_mapping = {
        "draft": "https://github.com/dynastyprocess/data/raw/master/files/db_fpecr_latest.csv",
        "week": "https://github.com/dynastyprocess/data/raw/master/files/fp_latest_weekly.csv",
        "all": "https://github.com/dynastyprocess/data/raw/master/files/db_fpecr.csv"
    }
    url = url_mapping[rank_type]

    # Load the data from the URL
    rankings_data = pd.read_csv(StringIO(requests.get(url).text))
    
    if rankings_data is not None:
        print(f"Data successfully loaded for {rank_type} rankings.")
    else:
        print(f"Failed to load {rank_type} rankings data.")

    return rankings_data

def ffs_latest_rankings(rank_type="draft"):
    # Ensure the argument is one of the allowed values
    if rank_type not in ["draft", "week"]:
        raise ValueError("Type must be either 'draft' or 'week'")

    # Handle "draft" type
    if rank_type == "draft":
        fp_latest = load_ff_rankings()
        
        # Apply filter similar to the R code
        condition = (fp_latest['ecr_type'] == 'rp') & (fp_latest['page_type'].str.split('-').str.get(1) == fp_latest['pos'].str.lower())
        fp_cleaned = fp_latest[condition][['player', 'id', 'pos', 'tm', 'bye', 'ecr', 'sd', 'sportsdata_id', 'scrape_date']]

        # Rename columns as per the R code
        fp_cleaned.rename(columns={
            'id': 'fantasypros_id',
            'tm': 'team',
            'sportsdata_id': 'sportradar_id'
        }, inplace=True)

        # Set `bye` to 0 if all values are NaN
        if fp_cleaned['bye'].isna().all():
            fp_cleaned['bye'] = 0

    # Handle "week" type
    elif rank_type == "week":
        fp_latest = load_ff_rankings("week")
        
        # Select and rename columns as per the R code
        fp_cleaned = fp_latest[['player_name', 'fantasypros_id', 'pos', 'team', 'ecr', 'sd', 'scrape_date']]
        fp_cleaned = fp_cleaned.rename(columns={'player_name': 'player'})

    return fp_cleaned

def get_ff_rosters(df_rosters):
    ff_rosters = df_rosters.merge(
        (dp_playerids()[['sleeper_id', 'fantasypros_id']]).dropna().astype(int).astype(str),
        left_on='player_id', right_on='sleeper_id', how='left'
    )
    
    return ff_rosters


# Function to simulate ff_starter_positions.sleeper_conn in Python
def get_ff_starter_positions(league):
    roster_positions = pd.DataFrame(league['roster_positions'], columns=['pos'])
    
    # Process the data
    df_positions = (roster_positions
        .loc[roster_positions['pos'] != 'BN']
        .groupby('pos').size()
        .reset_index(name='min')
        .set_index("pos")
        # .reindex(all_positions['pos'])  # Ensure all positions are included
        .fillna(0)  # Fill missing positions with 0
        .assign(total_starters=lambda x: x['min'].sum()))  # Calculate total starters

    def get_value_from_index(df, index_value, default_value=0):
        if index_value in df.index:
            return df.loc[index_value, 'min']
        else:
            return default_value
    
    # Calculate the flex values
    flex = get_value_from_index(df_positions, 'FLEX')
    wrrb_flex = get_value_from_index(df_positions, 'WRRB_FLEX')
    rec_flex = get_value_from_index(df_positions, 'REC_FLEX')
    super_flex = get_value_from_index(df_positions, 'SUPER_FLEX')
    idp_flex = get_value_from_index(df_positions, 'IDP_FLEX')

    # Define the conditions and calculate new columns
    def calculate_max(row):
        pos = row.name  # Get the index value
        if pos == 'QB':
            return int(row['min'] + super_flex)
        elif pos == 'RB':
            return int(row['min'] + wrrb_flex + super_flex + flex)
        elif pos == 'WR':
            return int(row['min'] + wrrb_flex + rec_flex + super_flex + flex)
        elif pos == 'TE':
            return int(row['min'] + rec_flex + super_flex + flex)
        elif pos in ['DL', 'LB', 'DB']:
            return int(row['min'] + idp_flex)
        else:
            return int(row['min'])

    # Apply the function to each row in the DataFrame
    df_positions['max'] = df_positions.apply(calculate_max, axis=1)

    # Calculate total, offense, defense, and kdef starters
    total_starters = df_positions['min'].sum()

    offense_starters = df_positions.loc[
        df_positions.index.isin(['QB', 'RB', 'WR', 'TE', 'FLEX', 'WRRB_FLEX', 'REC_FLEX', 'SUPER_FLEX']),
        'min'
    ].sum()

    defense_starters = df_positions.loc[
        df_positions.index.isin(['IDP_FLEX', 'DL', 'LB', 'DB']),
        'min'
    ].sum()

    kdef = df_positions.loc[
        df_positions.index.isin(['K', 'DEF']),
        'min'
    ].sum()

    # Add calculated columns to DataFrame
    df_positions['total_starters'] = total_starters
    df_positions['offense_starters'] = offense_starters
    df_positions['defense_starters'] = defense_starters
    df_positions['kdef'] = kdef

    # Filter out positions containing 'FLEX' and rows where 'max' <= 0
    df_filtered = df_positions[
        ~df_positions.index.str.contains('FLEX', na=False) & (df_positions['max'] > 0)
    ].reset_index()

    # Select relevant columns
    df_final = df_filtered[['pos', 'min', 'max', 'offense_starters', 'defense_starters', 'total_starters']]

    return df_final

def fetch_fp_rankings(page, season):
    try:
        # Call the fp_rankings function
        result = r['fp_rankings'](page, year=season)
        print(f"Completed {page} {season}")
        return pandas2ri.rpy2py(result)
    except Exception as e:
        print(f"Error fetching rankings for {page}, {season}: {e}")
        return pd.DataFrame()

# Function to clean player names (mimicking nflreadr::clean_player_names)
def clean_player_names(player_name):
    return player_name.strip().title()

# Function to clean and process position (pos) and pages
def process_position(page_pos):
    page_pos_clean = re.sub(r'cheatsheets|^ppr|\-', '', page_pos).upper().strip()
    return page_pos_clean

# Process data for a combination of pages and seasons
def process_fp_rankings(seasons):
    pages = [
        "qb-cheatsheets",
        "ppr-rb-cheatsheets",
        "ppr-wr-cheatsheets",
        "ppr-te-cheatsheets",
        "k-cheatsheets",
        "dst-cheatsheets"
    ]
    combinations = itertools.product(pages, seasons)
    
    all_rankings = []
    for page, season in combinations:
        try:
            rankings = fetch_fp_rankings(page, season)  # Fetch the rankings using page and season
            rankings['page_pos'] = process_position(page)  # Clean page position
            rankings['season'] = season
            rankings['player_name'] = rankings['player_name'].apply(clean_player_names)
            
            # Filter the rows where page_pos matches pos
            rankings = rankings[rankings['page_pos'] == rankings['pos']]
            
            all_rankings.append(rankings)
        
        except Exception as e:
            print(f"Error processing {page} for season {season}: {e}")
    
    if all_rankings:
        return pd.concat(all_rankings, ignore_index=True)
    else:
        return pd.DataFrame()

def ff_rank_expand(x):
    # Create a sequence from (x - 2) to (x + 2)
    sequence = list(range(x - 2, x + 3))
    
    # Replace values <= 0 with 1
    return [1 if i <= 0 else i for i in sequence]

def ff_apply_gp_model(adp_outcomes, model_type, fp_injury_table=None):
    if model_type == "none":
        # Set the 'prob_gp' column to 1
        adp_outcomes['prob_gp'] = 1
    
    elif model_type == "simple" and fp_injury_table is not None:
        # Merge adp_outcomes with fp_injury_table on the columns 'pos' and 'rank'
        adp_outcomes = pd.merge(adp_outcomes, fp_injury_table, on=['pos', 'rank'], how='left')
    
    return adp_outcomes

def ffs_adp_outcomes(scoring_history, seasons, gp_model="none", pos_filter=["QB", "RB", "WR", "TE"]):
    assert gp_model in ["simple", "none"], "Invalid gp_model choice"
    assert isinstance(pos_filter, list), "pos_filter must be a list"
    assert isinstance(scoring_history, pd.DataFrame), "scoring_history must be a DataFrame"
    # assert all(col in scoring_history.columns for col in ["gsis_id", "team", "season", "points"]), "scoring_history must contain required columns" # TODO reinstate but remove gsis_id?
    
    if not os.path.isfile("fp_rankings_history.csv"):
        fp_rankings_history = process_fp_rankings(seasons)
        fp_rankings_history.to_csv("fp_rankings_history.csv")
    else:
        fp_rankings_history = pd.read_csv("fp_rankings_history.csv")
        if fp_rankings_history["season"].max() < max(seasons):
            fp_rankings_history = process_fp_rankings(seasons)

    # Filter scoring_history for non-null gsis_id and week <= 17
    # sh = scoring_history[(scoring_history['gsis_id'].notna()) & (scoring_history['week'] <= 17)][["gsis_id", "team", "season", "points"]] # TODO which id to use?
    sh = scoring_history[(scoring_history['sleeper_id'].notna()) & (scoring_history['week'] <= 17)][["sleeper_id", "team", "season", "points"]]

    # Load necessary datasets
    fp_rh = fp_rankings_history.drop(columns=["page_pos"])
    dp_id = pd.DataFrame(dp_playerids()).dropna(subset=["sleeper_id", "fantasypros_id"])[["fantasypros_id", "sleeper_id"]]

    # Merge fp_rh and dp_id on fantasypros_id
    ao = pd.merge(fp_rh, dp_id, on="fantasypros_id", how="inner")
    ao["sleeper_id"] = ao["sleeper_id"].astype(int).astype(str)

    # Filter for pos_filter and non-null gsis_id
    ao = ao[(ao['sleeper_id'].notna()) & (ao['pos'].isin(pos_filter))]

    # Merge with scoring_history
    ao = pd.merge(ao, sh, on=["season", "sleeper_id"], how="inner")

    # Group by and apply transformations similar to data.table
    ao_grouped = ao.groupby(["season", "pos", "rank", "fantasypros_id", "player_name"]).agg(
        week_outcomes=("points", lambda x: list(x)),
        games_played=("sleeper_id", "size")
    ).reset_index()

    # Replicate the R behavior of expanding rows and applying .ff_rank_expand logic
    ao_expanded = pd.DataFrame({
        "season": np.repeat(ao_grouped['season'], 5),
        "pos": np.repeat(ao_grouped['pos'], 5),
        "fantasypros_id": np.repeat(ao_grouped['fantasypros_id'], 5),
        "player_name": np.repeat(ao_grouped['player_name'], 5),
        "games_played": np.repeat(ao_grouped['games_played'], 5),
        "week_outcomes": np.repeat(ao_grouped['week_outcomes'], 5),
        "rank": np.concatenate([ff_rank_expand(int(r)) for r in ao_grouped['rank']])
    })

    # Apply GP model
    ao_expanded = ff_apply_gp_model(ao_expanded, gp_model)

    # Final grouping and cleaning
    ao_final = ao_expanded.groupby(["pos", "rank", "prob_gp"]).agg(
        # week_outcomes=("week_outcomes", lambda x: list(*x)),
        week_outcomes=("week_outcomes", lambda x: sum(x, [])),
        player_name=("player_name", list),  # No need for lambda, list() works directly
        fantasypros_id=("fantasypros_id", list)  # No need for lambda, list() works directly
    ).reset_index()

    ao_final = ao_final[ao_final['fantasypros_id'].notna()].sort_values(by=["pos", "rank"]).reset_index(drop=True)

    return ao_final

def ffs_generate_projections_week(adp_outcomes, latest_rankings, sims, rosters=None):
    
    # Validate inputs
    if not isinstance(sims, int) or sims < 1:
        raise ValueError("`n` must be an integer greater than or equal to 1")

    if not isinstance(adp_outcomes, pd.DataFrame):
        raise ValueError("`adp_outcomes` must be a pandas DataFrame")

    adp_outcomes = adp_outcomes[["pos", "rank", "week_outcomes"]]
    
    if not isinstance(latest_rankings, pd.DataFrame):
        raise ValueError("`latest_rankings` must be a pandas DataFrame")

    latest_rankings = latest_rankings[["player", "pos", "team", "ecr", "sd", "fantasypros_id", "scrape_date"]]
    
    # Handle the rosters DataFrame (if not provided, use fantasypros_id from latest_rankings)
    if rosters is None:
        rosters = latest_rankings[["fantasypros_id"]]
    if not isinstance(rosters, pd.DataFrame):
        raise ValueError("`rosters` must be a pandas DataFrame")
    
    # Filter rankings by rosters
    rankings = latest_rankings[latest_rankings['fantasypros_id'].isin(rosters['fantasypros_id'])]
    
    # Expand rankings for `n` projections and calculate rank
    expanded_rankings = rankings.loc[np.repeat(rankings.index, sims)]
    expanded_rankings['week'] = np.tile(np.arange(1, sims + 1), len(rankings))
    expanded_rankings['rank'] = np.round(norm.rvs(loc=expanded_rankings['ecr'], scale=expanded_rankings['sd'] / 2)).astype(int)
    expanded_rankings['rank'] = expanded_rankings['rank'].replace(0, 1)
    
    # Merge with adp_outcomes on "pos" and "rank"
    ps = pd.merge(expanded_rankings, adp_outcomes, on=["pos", "rank"], how='inner')
    
    # Remove NA values in 'ecr' column
    ps = ps[~ps['ecr'].isna()]
    
    # Sample week_outcomes for projections
    ps['projected_score'] = ps.apply(lambda row: np.random.choice(row['week_outcomes']), axis=1)
    
    # Add additional columns: projection, gp_model, season
    ps['projection'] = ps['projected_score']
    ps['gp_model'] = 1
    ps['season'] = 1
    
    # Return the final DataFrame
    return ps

def ffs_generate_projections(adp_outcomes, latest_rankings, sims=100, weeks=range(1, 15), rosters=None):
    
    # Validate input for number of seasons and weeks
    if not isinstance(sims, int) or sims < 1:
        raise ValueError("`n_seasons` must be an integer greater than or equal to 1")

    if not isinstance(weeks, (list, range)) or not all(isinstance(w, int) for w in weeks):
        raise ValueError("`weeks` must be a list or range of integers")
    weeks = list(set(weeks))  # Ensure unique weeks
    n_weeks = len(weeks)

    # Validate adp_outcomes DataFrame
    if not isinstance(adp_outcomes, pd.DataFrame):
        raise ValueError("`adp_outcomes` must be a pandas DataFrame")
    adp_outcomes = adp_outcomes[["pos", "rank", "prob_gp", "week_outcomes"]]

    # Validate latest_rankings DataFrame
    if not isinstance(latest_rankings, pd.DataFrame):
        raise ValueError("`latest_rankings` must be a pandas DataFrame")
    latest_rankings = latest_rankings[["player", "pos", "team", "ecr", "sd", "bye", "fantasypros_id", "scrape_date"]]

    # Handle rosters
    if rosters is None:
        rosters = latest_rankings[["fantasypros_id"]]
    if not isinstance(rosters, pd.DataFrame):
        raise ValueError("`rosters` must be a pandas DataFrame")
    latest_rankings["fantasypros_id"] = latest_rankings["fantasypros_id"].astype(str)

    # Filter rankings by rosters
    rankings = latest_rankings[latest_rankings['fantasypros_id'].isin(rosters['fantasypros_id'])]

    # Expand rankings for each season and calculate rank
    expanded_rankings = rankings.loc[np.repeat(rankings.index, sims)]
    expanded_rankings['season'] = np.tile(np.arange(1, sims + 1), len(rankings))
    expanded_rankings['rank'] = np.round(norm.rvs(loc=expanded_rankings['ecr'], scale=expanded_rankings['sd'] / 2)).astype(int)
    expanded_rankings['rank'] = expanded_rankings['rank'].replace(0, 1)

    # Merge rankings with adp_outcomes on "pos" and "rank"
    ps = pd.merge(expanded_rankings, adp_outcomes, on=["pos", "rank"], how='inner')

    # Filter out rows where ecr or prob_gp are NaN
    ps = ps[~ps['ecr'].isna() & ~ps['prob_gp'].isna()]

    # Generate projections and gp_model for each week
    def sample_week_outcomes(row, n_weeks):
        return np.random.choice(row['week_outcomes'], size=n_weeks, replace=True)

    def sample_gp_model(prob_gp, n_weeks):
        return binom.rvs(n=1, p=prob_gp, size=n_weeks)

    # Apply the sampling for each row
    ps['week'] = [weeks] * len(ps)
    ps['projection'] = ps.apply(lambda row: sample_week_outcomes(row, n_weeks), axis=1)
    ps['gp_model'] = ps.apply(lambda row: sample_gp_model(row['prob_gp'], n_weeks), axis=1)

    # Explode the DataFrame for week, projection, and gp_model
    ps = ps.explode(['week', 'projection', 'gp_model'], ignore_index=True)

    # Calculate projected score, considering the player's bye week
    ps['projected_score'] = ps.apply(
        lambda row: row['projection'] * row['gp_model'] * (row['week'] != row['bye']), axis=1
    )

    # Return the final DataFrame
    return ps

def ffs_score_rosters(projected_scores, rosters):
    # Validate input DataFrames
    if not isinstance(projected_scores, pd.DataFrame):
        raise ValueError("`projected_scores` must be a pandas DataFrame")
    if not isinstance(rosters, pd.DataFrame):
        raise ValueError("`rosters` must be a pandas DataFrame")

    # Select only the necessary columns in projected_scores
    projected_scores = projected_scores[[
        "fantasypros_id", "ecr", "rank", "projection", "gp_model", 
        "season", "week", "projected_score", "scrape_date"
    ]]

    # Merge the rosters with projected scores based on fantasypros_id
    # `how='inner'` ensures only matching rows are kept (equivalent to `all = FALSE` in R)
    roster_scores = pd.merge(
        rosters, projected_scores, on="fantasypros_id", how="inner"
    )

    # Rank players by projected_score within each group (equivalent to R's data.table ranking)
    roster_scores['pos_rank'] = roster_scores.groupby(
        ['roster_id', 'position', 'season', 'week']
    )['projected_score'].rank(method='first', ascending=False).astype(int)

    # Return the resulting DataFrame
    return roster_scores

import pandas as pd
import numpy as np
from scipy.optimize import linprog

def ff_optimise_one_lineup(franchise_scores, lineup_constraints):
    min_req = lineup_constraints['min'].sum()

    # Handle player IDs and scores, filling with NAs and zeros where needed
    player_ids = list(franchise_scores['player_id']) + [None] * min_req
    player_scores = list(franchise_scores['projected_score']) + [0] * min_req
    player_scores = np.nan_to_num(player_scores)

    # Binary position identifiers
    pos_ids = []
    for pos in lineup_constraints['pos']:
        pos_ids += list((franchise_scores['pos'] == pos).astype(int)) + [1] * min_req

    # Building the constraints matrix
    num_constraints = 2 * len(lineup_constraints) + 2
    constraints_matrix = np.zeros((num_constraints, len(player_scores)))

    # Adding position constraints for minimum and maximum positions
    row = 0
    for i, pos in enumerate(lineup_constraints['pos']):
        # Min constraints for each position
        constraints_matrix[row, :] = np.concatenate([np.where(franchise_scores['pos'] == pos, 1, 0), [1] * min_req])
        row += 1
    for i, pos in enumerate(lineup_constraints['pos']):
        # Max constraints for each position
        constraints_matrix[row, :] = np.concatenate([np.where(franchise_scores['pos'] == pos, 1, 0), [1] * min_req])
        row += 1

    # Offense starter constraints (for QB, RB, WR, TE)
    constraints_matrix[row, :] = np.concatenate([np.where(franchise_scores['pos'].isin(["QB", "RB", "WR", "TE"]), 1, 0), [1] * min_req])
    row += 1

    # Total starter constraints
    constraints_matrix[row, :] = np.ones(len(player_scores))

    # Constraints directions
    constraints_dir = ['>='] * len(lineup_constraints) + ['<='] * len(lineup_constraints) + ['<=', '<=']

    # Right-hand side (RHS) of constraints
    constraints_rhs = np.concatenate([lineup_constraints['min'], lineup_constraints['max'], 
                                      [lineup_constraints['offense_starters'].iloc[0]], 
                                      [lineup_constraints['total_starters'].iloc[0]]])

    # Solving the linear programming problem
    res = linprog(c=-np.array(player_scores), A_ub=constraints_matrix, b_ub=constraints_rhs, 
                  bounds=(0, 1), method='highs', options={"disp": False})

    if res.success:
        selected_players = np.where(res.x > 0.5)[0]  # Since the solution is binary, 0 or 1
        optimal_score = -res.fun  # Because we negated the objective
        optimal_player_ids = [player_ids[i] for i in selected_players]
        optimal_player_scores = [player_scores[i] for i in selected_players]

        optimals = {
            'optimal_score': optimal_score,
            'optimal_player_id': optimal_player_ids,
            'optimal_player_score': optimal_player_scores
        }
    else:
        optimals = {
            'optimal_score': 0,
            'optimal_player_id': [],
            'optimal_player_score': []
        }

    return optimals

def ffs_optimise_lineups(
    roster_scores, lineup_constraints, lineup_efficiency_mean=0.775, 
    lineup_efficiency_sd=0.05, best_ball=False, pos_filter=("QB", "RB", "WR", "TE")
):
    # Validate inputs
    assert 0 <= lineup_efficiency_mean <= 1, "lineup_efficiency_mean must be between 0 and 1"
    assert 0 <= lineup_efficiency_sd <= 0.25, "lineup_efficiency_sd must be between 0 and 0.25"
    assert isinstance(best_ball, bool), "best_ball must be a boolean"
    
    # Filter roster_scores based on pos_filter and convert to DataFrame if necessary
    roster_scores = roster_scores[roster_scores['position'].isin(pos_filter)]
    
    # Merge the roster_scores and lineup_constraints on "pos" for max lineup constraints
    max_lineup_constraints = lineup_constraints[['pos', 'max']]
    optimal_scores = pd.merge(roster_scores, max_lineup_constraints, left_on='position', right_on='pos')

    # Filter roster based on position rank and lineup constraints
    optimal_scores = optimal_scores[optimal_scores['pos_rank'] <= optimal_scores['max']]

    # Apply the lineup optimization for each franchise and week
    result = []
    group_cols = ['roster_id', 'season', 'week']
    
    for _, group in optimal_scores.groupby(group_cols):
        franchise_scores = group[['player_id', 'pos', 'projected_score']]
        optimals = ff_optimise_one_lineup(franchise_scores, lineup_constraints)
        optimals.update({
            'roster_id': group['roster_id'].iloc[0],
            'season': group['season'].iloc[0],
            'week': group['week'].iloc[0]
        })
        result.append(optimals)
    
    # Create a DataFrame from the result
    optimal_scores_df = pd.DataFrame(result)

    # Apply best ball logic for lineup efficiency
    if best_ball:
        optimal_scores_df['lineup_efficiency'] = 1
    else:
        optimal_scores_df['lineup_efficiency'] = np.random.normal(lineup_efficiency_mean, lineup_efficiency_sd, len(optimal_scores_df))

    # Calculate actual score
    optimal_scores_df['actual_score'] = optimal_scores_df['optimal_score'] * optimal_scores_df['lineup_efficiency']

    return optimal_scores_df


def ffs_summarise_week(optimal_scores, schedules, df_matchup_schedule, current_gameweek):    
    # Create a copy of optimal_scores to avoid modifying the original DataFrame
    team = optimal_scores.copy()

    # Calculate allplay_wins and allplay_games
    team['allplay_wins'] = team.groupby(['season', 'week'])['actual_score'].rank(method="min") - 1
    team['allplay_games'] = team.groupby(['season', 'week'])['actual_score'].transform('count') - 1
    team['allplay_pct'] = (team['allplay_wins'] / team['allplay_games']).round(3)

    # Create opponent DataFrame
    opponent = team.copy()
    team.rename(columns={'actual_score': 'team_score'}, inplace=True)
    opponent.rename(columns={
        'actual_score': 'opponent_score', 
        'roster_id': 'opponent_id',
        'optimal_score': 'optimal_opponent_score'
    }, inplace=True)
    
    # Merge schedules, team, and opponent DataFrames
    summary_week = pd.merge(schedules, team, left_on=["roster_id", "season", "gameweek"], right_on=["roster_id", "season", "week"], how="inner")
    summary_week = pd.merge(summary_week, opponent[['opponent_score', 'optimal_opponent_score', 'opponent_id', 'season', 'week']], 
                            on=["opponent_id", "season", "week"], how="inner")
    
    # Calculate the result of the match
    summary_week['result'] = np.select(
        [summary_week['team_score'] > summary_week['opponent_score'], summary_week['team_score'] < summary_week['opponent_score'], summary_week['team_score'] == summary_week['opponent_score']],
        ['W', 'L', 'T'], 
        default="Error"
    )
    
    # Round scores and efficiencies
    summary_week['team_score'] = summary_week['team_score'].round(2)
    summary_week['optimal_score'] = summary_week['optimal_score'].round(2)
    summary_week['opponent_score'] = summary_week['opponent_score'].round(2)
    summary_week['lineup_efficiency'] = summary_week['lineup_efficiency'].round(3)
    
    # Select the final columns for the summary
    final_columns = [
        'season', 'week', 'optimal_score', 'lineup_efficiency',
        'team_score', 'opponent_score', 'optimal_opponent_score', 'result', 'opponent_id', 'allplay_wins',
        'allplay_games', 'allplay_pct', 'roster_id', 'optimal_player_id', 'optimal_player_score'
    ]
    
    summary_week = summary_week[final_columns]
    
    # Sort the summary_week DataFrame by season, week, and roster_id
    summary_week.sort_values(by=['season', 'week', 'roster_id'], inplace=True)
    
    summary_week = summary_week.merge(
        df_matchup_schedule[["roster_id", "gameweek", "manager", "points", "points_opponent", "win", "all_play",
                            "pp_points", "pp_points_opponent", "pp_win"]],
        left_on=["roster_id", "week"], right_on=["roster_id", "gameweek"], how="left"
    )
    summary_week['selected_pts'] = np.where(summary_week['week'] > current_gameweek, summary_week['team_score'], summary_week['points'])
    summary_week['selected_win'] = np.where(summary_week['week'] > current_gameweek, np.where(summary_week['result'] == 'W', 1, np.where(summary_week['result'] == 'T', 0.5, 0)), summary_week['win'])
    summary_week['selected_all_play'] = np.where(summary_week['week'] > current_gameweek, summary_week['allplay_wins'], summary_week['all_play'])
    summary_week['selected_pp'] = np.where(summary_week['week'] > current_gameweek, summary_week['optimal_score'], summary_week['pp_points'])
    summary_week['selected_pp_opponent'] = np.where(summary_week['week'] > current_gameweek, summary_week['optimal_opponent_score'], summary_week['pp_points_opponent'])
    summary_week['selected_pp_win'] = np.where(
        summary_week['week'] > current_gameweek,
        np.where(summary_week['optimal_score'] > summary_week['optimal_opponent_score'], 1, np.where(summary_week['optimal_score'] == summary_week['optimal_opponent_score'], 0.5, 0)),
        summary_week['pp_win']
    )
    summary_week['nxt_wk_win'] = np.where(summary_week['week'] == current_gameweek + 1, summary_week['selected_win'], 0)
    
    return summary_week
