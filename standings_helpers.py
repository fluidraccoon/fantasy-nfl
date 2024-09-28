import pandas as pd
import numpy as np
import itertools
import pulp
from sleeper_helpers import (
    get_user_df,
    get_roster_df,
    get_matchup_df
)


def get_matchup_schedule_df(df_matchups):
        matchup_id = df_matchups.loc[:, ["roster_id", "matchup_id", "gameweek"]].drop_duplicates(ignore_index=True)
        matchup_id = matchup_id.merge(matchup_id, on=["matchup_id", "gameweek"], how="left", suffixes=(None, "_opponent"))
        matchup_id = matchup_id[matchup_id["roster_id"]!=matchup_id["roster_id_opponent"]].reset_index(drop=True)
        matchup_id = matchup_id.rename(columns = {"roster_id_opponent": "opponent_id"})

        return matchup_id

def get_potential_points(df_matchups, roster_positions, league_size, gameweek_end):
    df_lineup_list = []
    for roster_id, gameweek in itertools.product(range(1, league_size + 1), range(1, gameweek_end + 1)):
        df_lineup = df_matchups[df_matchups["roster_id"].eq(roster_id) & df_matchups["gameweek"].eq(gameweek)].reset_index(drop=True)
        num_players = len(roster_positions) - roster_positions.count("BN")
        num_qb = [roster_positions.count("QB"), roster_positions.count("QB") + roster_positions.count("SUPER_FLEX")]
        num_rb = [roster_positions.count("RB"), roster_positions.count("RB") + roster_positions.count("FLEX") + roster_positions.count("SUPER_FLEX")]
        num_wr = [roster_positions.count("WR"), roster_positions.count("WR") + roster_positions.count("FLEX") + roster_positions.count("SUPER_FLEX")]
        num_te = [roster_positions.count("TE"), roster_positions.count("TE") + roster_positions.count("FLEX") + roster_positions.count("SUPER_FLEX")]
        num_k = roster_positions.count("K")
        num_def = roster_positions.count("DEF")

        ## Set variables
        x = pulp.LpVariable.dict("player", range(0, len(df_lineup)), 0,1, cat=pulp.LpInteger)
        prob = pulp.LpProblem("Fantasy", pulp.LpMaximize)
        prob += pulp.lpSum(df_lineup["points"][i] * x[i] for i in range(0, len(df_lineup)))
        prob += sum(x[i] for i in range(0, len(df_lineup))) ==  num_players

        prob  += sum(x[i] for i in range(0, len(df_lineup)) if df_lineup["position"][i] == "QB") >= num_qb[0]
        prob  += sum(x[i] for i in range(0, len(df_lineup)) if df_lineup["position"][i] == "QB") <= num_qb[1]
        prob  += sum(x[i] for i in range(0, len(df_lineup)) if df_lineup["position"][i] == "RB") >= num_rb[0]
        prob  += sum(x[i] for i in range(0, len(df_lineup)) if df_lineup["position"][i] == "RB") <= num_rb[1]
        prob  += sum(x[i] for i in range(0, len(df_lineup)) if df_lineup["position"][i] == "WR") >= num_wr[0]
        prob  += sum(x[i] for i in range(0, len(df_lineup)) if df_lineup["position"][i] == "WR") <= num_wr[1]
        prob  += sum(x[i] for i in range(0, len(df_lineup)) if df_lineup["position"][i] == "TE") >= num_te[0]
        prob  += sum(x[i] for i in range(0, len(df_lineup)) if df_lineup["position"][i] == "TE") <= num_te[1]
        prob  += sum(x[i] for i in range(0, len(df_lineup)) if df_lineup["position"][i] == "K") <= num_k
        prob  += sum(x[i] for i in range(0, len(df_lineup)) if df_lineup["position"][i] == "DEF") <= num_def
        prob.solve()

        for i in range(0, len(df_lineup)):
            df_lineup.loc[i, "pp_starter"] = bool(pulp.value(x[i]))
        
        df_lineup["pp_points"] = df_lineup["points"] * df_lineup["pp_starter"]
        df_lineup["starter_points"] = df_lineup["points"] * df_lineup["starter"]
        df_lineup_list.append(df_lineup)

    return pd.concat(df_lineup_list)


def add_pp_data(df_matchups, df_matchup_schedule, df_rosters):
    df_rosters = df_rosters.loc[:, ["roster_id", "manager"]].drop_duplicates().reset_index(drop=True)
    df_matchup_schedule_pp = df_matchups.groupby(["gameweek", "roster_id"]).agg(
        pp_points=("pp_points", "sum"),
        points=("starter_points", "sum")
    ).reset_index().merge(df_rosters, on="roster_id", how="left")
    df_matchup_schedule_pp = df_matchup_schedule_pp\
        .merge(df_matchup_schedule[["gameweek", "roster_id", "opponent_id"]], on=["gameweek", "roster_id"], how="left")
    df_matchup_schedule_pp = df_matchup_schedule_pp\
        .merge(df_matchup_schedule_pp[["gameweek", "opponent_id", "pp_points", "points"]], left_on=["gameweek", "roster_id"], right_on=["gameweek", "opponent_id"],
            how="left", suffixes=[None, "_opponent"])\
        .drop(columns=['opponent_id_opponent'])
    df_matchup_schedule_pp["pp_win"] = np.where(df_matchup_schedule_pp["pp_points"] > df_matchup_schedule_pp["pp_points_opponent"], 1,
                                            np.where(df_matchup_schedule_pp["pp_points"] < df_matchup_schedule_pp["pp_points_opponent"], 0, 0.5))
    df_matchup_schedule_pp["win"] = np.where(df_matchup_schedule_pp["points"] > df_matchup_schedule_pp["points_opponent"], 1,
                                            np.where(df_matchup_schedule_pp["points"] < df_matchup_schedule_pp["points_opponent"], 0, 0.5))
    df_matchup_schedule_pp["all_play"] = df_matchup_schedule_pp.groupby('gameweek')['points'].rank() - 1
    
    return df_matchup_schedule_pp


def get_matchup_schedule_with_pp(league, df_players):
    league_size = league["settings"]["num_teams"]
    final_gameweek = 14 if league_size == 12 else 15

    df_users = get_user_df(league["league_id"])
    df_rosters = get_roster_df(league["league_id"])\
        .merge(df_players[["player_id", "full_name", "position", "team", "age"]], on="player_id", how="left")\
        .merge(df_users[["owner_id", "manager", "team_name"]], on="owner_id", how="left")
    df_matchups = get_matchup_df(league["league_id"], final_gameweek)\
        .merge(df_players[["player_id", "full_name", "position", "team"]], on="player_id", how="left")
    df_matchups["full_name"] = np.where(df_matchups["position"].eq("DEF"), df_matchups["team"], df_matchups["full_name"])
    df_matchup_schedule = get_matchup_schedule_df(df_matchups)

    df_matchups = get_potential_points(df_matchups, league["roster_positions"], league_size, final_gameweek)

    df_matchup_schedule = add_pp_data(df_matchups, df_matchup_schedule, df_rosters)
    df_matchup_schedule["league"] = league["name"]

    return df_matchup_schedule, df_rosters, df_matchups