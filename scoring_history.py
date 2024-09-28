import pandas as pd
import nfl_data_py as nfl
import requests
import re

def nflverse_player_stats_long(season):
    selected_columns = [
        "season", "week", "player_id",
        "attempts", "carries", "completions", "interceptions", "passing_2pt_conversions", 
        "passing_first_downs", "passing_tds", "passing_yards", "receiving_2pt_conversions", 
        "receiving_first_downs", "receiving_fumbles", "receiving_fumbles_lost", 
        "receiving_tds", "receiving_yards", "receptions", "rushing_2pt_conversions", 
        "rushing_first_downs", "rushing_fumbles", "rushing_fumbles_lost", "rushing_tds", 
        "rushing_yards", "sack_fumbles", "sack_fumbles_lost", "sack_yards", "sacks", 
        "special_teams_tds", "targets"
    ]
    
    ps = nfl.import_weekly_data(season, selected_columns)
    ps = ps[selected_columns]
    
    # Melt the dataframe to long format, with 'season', 'week', and 'player_id' as id_vars
    ps_long = ps.melt(id_vars=["season", "week", "player_id"], var_name="metric")
    
    return ps_long

def nflverse_kicking_long(season):
    psk = pd.read_csv("data/player_stats/player_stats_kicking.csv")
    psk = psk[psk["season"].isin(season)]

    selected_columns = [
        "season", "week", "player_id",
        "fg_att", "fg_blocked", "fg_made", "fg_made_0_19", "fg_made_20_29", "fg_made_30_39",
        "fg_made_40_49", "fg_made_50_59", "fg_made_60_", "fg_made_distance",
        "fg_missed", "fg_missed_0_19", "fg_missed_20_29", "fg_missed_30_39",
        "fg_missed_40_49", "fg_missed_50_59", "fg_missed_60_", "fg_missed_distance",
        "fg_pct", "pat_att", "pat_blocked", "pat_made", "pat_missed", "pat_pct"
    ]

    psk = psk[selected_columns]

    # Melt the dataframe (convert from wide to long format)
    psk_long = pd.melt(psk, id_vars=["season", "week", "player_id"], var_name="metric")

    return psk_long

def dp_playerids():
    url_query = "https://github.com/DynastyProcess/data/raw/master/files/db_playerids.csv"
    
    # Send a GET request to fetch the CSV file
    response = requests.get(url_query, headers={"Accept": "text/csv"})

    # Check if the request was successful
    if response.status_code != 200:
        raise Exception(f"GitHub request failed with error: <{response.status_code}> while calling <{url_query}>")
    
    # Convert the CSV content into a pandas DataFrame
    from io import StringIO
    csv_data = StringIO(response.text)
    player_ids_df = pd.read_csv(csv_data)
    
    return player_ids_df

def nflverse_roster(season):
    # Load the roster data for the given season
    ros = nfl.import_seasonal_rosters(season)

    # Replace "HB" and "FB" with "RB" in the "position" column
    ros['position'] = ros['position'].replace({'HB': 'RB', 'FB': 'RB'})

    # Select specific columns
    selected_columns = {
    'season': 'season',
    'gsis_it_id': 'gsis_id',
    'sportradar_id': 'sportradar_id',
    'player_name': 'player_name',
    'position': 'pos',
    'team': 'team'
    }
    ros = ros[list(selected_columns.keys())].rename(columns=selected_columns)

    # Group by 'season', 'gsis_id', 'sportradar_id' and take the last entry in each group
    ros = ros.groupby(['season', 'gsis_id', 'sportradar_id']).last().reset_index().drop(columns=['gsis_id'])

    # Load player IDs and select the relevant columns
    player_ids = dp_playerids()
    player_ids = player_ids[['sportradar_id', 'mfl_id', 'gsis_id', 'sleeper_id', 'espn_id', 'fleaflicker_id']]

    # Perform a left join with the player IDs data
    ros = ros.merge(player_ids, on='sportradar_id', how='left')

    return ros

def sleeper_rule_mapping(league):
    # Extract the scoring settings from the response
    scoring_settings = league["scoring_settings"]
    
    # Convert the scoring settings to a pandas DataFrame
    scoring_rules = pd.DataFrame(list(scoring_settings.items()), columns=['event', 'points'])

    # Define the `pos` column using conditional logic equivalent to dplyr::case_when
    def assign_pos(event):
        if re.search(r"idp", event):
            return ["DL", "LB", "DB"]
        elif re.search(r"def|allow", event):
            return ["DEF"]
        elif event in [
            "qb_hit", "sack", "sack_yd", "safe", "int", "int_ret_yd", "fum_ret_yd", "fg_ret_yd",
            "tkl", "tkl_loss", "tkl_ast", "tkl_solo", "ff", "blk_kick", "blk_kick_ret_yd"
        ]:
            return ["DEF"]
        elif event in [
            "st_td", "st_ff", "st_fum_rec", "st_tkl_solo", "pr_td", "pr_yd", "kr_td", "kr_yd",
            "fum", "fum_lost", "fum_rec_td"
        ]:
            return ["QB", "RB", "WR", "TE", "DL", "LB", "DB", "K"]
        elif re.search(r"qb", event):
            return ["QB"]
        elif re.search(r"rb", event):
            return ["RB"]
        elif re.search(r"wr", event):
            return ["WR"]
        elif re.search(r"te", event):
            return ["TE"]
        else:
            return ["QB", "RB", "WR", "TE", "K"]

    # # Apply the logic to create the `pos` column
    scoring_rules['pos'] = scoring_rules['event'].apply(assign_pos)
    
    # # Unnest the `pos` column (explode in pandas)
    scoring_rules = scoring_rules.explode('pos')
    
    # # Select only the 'event' and 'pos' columns, dropping 'points'
    scoring_rules = scoring_rules[['event', 'pos']]
    
    return scoring_rules

def ff_scoring_sleeper_conn(league, df_sleeper_rule_mapping):
    # Extract the scoring settings from the response
    scoring_settings = league["scoring_settings"]
    
    # Convert the scoring settings to a pandas DataFrame
    scoring_rules = pd.DataFrame(list(scoring_settings.items()), columns=['event', 'points'])
    
    # # Perform an inner join with the sleeper_rule_mapping DataFrame
    scoring_rules = pd.merge(scoring_rules, df_sleeper_rule_mapping, on='event', how='inner')
    
    # Select the desired columns: 'pos', 'event', and 'points'
    scoring_rules = scoring_rules[['pos', 'event', 'points']]
    
    return scoring_rules

def get_league_rules(scoring_rules):
    stat_mapping = pd.read_csv("data\\player_stats\\stat_mapping.csv")

    # Filter nflfastr_stat_mapping for "sleeper" platform
    nflfastr_stat_mapping_sleeper = stat_mapping[stat_mapping['platform'] == 'sleeper']

    # Perform a left join on 'event' and 'ff_event'
    league_rules = pd.merge(scoring_rules, nflfastr_stat_mapping_sleeper, left_on='event', right_on='ff_event', how='left')

    # Drop the 'ff_event' column if not needed (optional)
    league_rules = league_rules.drop(columns=['ff_event'])
    
    return league_rules

def get_scoring_history(league, start_year, end_year):
    ps = nflverse_player_stats_long(range(start_year, end_year + 1))
    psk = nflverse_kicking_long(range(start_year, end_year + 1))
    ros = nflverse_roster(range(start_year, end_year + 1))
    
    ps = pd.concat([ps, psk]) if "K" in league["roster_positions"] else ps
    
    df_sleeper_rule_mapping = sleeper_rule_mapping(league)
    scoring_rules = ff_scoring_sleeper_conn(league, df_sleeper_rule_mapping)
    
    league_rules = get_league_rules(scoring_rules)
    
    # Inner join between `ros` and `ps` on 'gsis_id' and 'season'
    merged_data = pd.merge(ros, ps, left_on=['gsis_id', 'season'], right_on=['player_id', 'season'], how='inner')

    # Inner join between the merged data and `league_rules` on 'metric' and 'pos'
    merged_data = pd.merge(merged_data, league_rules, left_on=['metric', 'pos'], right_on=['nflfastr_event', 'pos'], how='inner')

    # Calculate the points by multiplying `value` and `points` columns
    merged_data['points'] = merged_data['value'] * merged_data['points']

    # # Group by 'season', 'week', 'gsis_id', 'sportradar_id' and sum up points
    merged_data['points'] = merged_data.groupby(['season', 'week', 'player_id'])['points'].transform(lambda x: round(x.sum(), 2))

    # # Unpivot the data (pivot wider in R) - pivoting the 'metric' column into new columns
    final_data = merged_data.pivot_table(
        index=['season', 'week', 'player_id', 'sleeper_id', 'player_name', 'pos', 'team', 'points'],
        columns='metric',
        values='value',
        fill_value=0,
        aggfunc='max'
    ).reset_index()

    # # Optionally, remove the extra index column created by pivot_table
    final_data.columns.name = None
    
    return final_data