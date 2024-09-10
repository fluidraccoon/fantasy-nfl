import pandas as pd
from dotmap import DotMap
from sleeperpy import (
    User,
    Leagues,
    Players
)


def get_league_dict(username, season, sport="nfl"):
    all_leagues = []
    account = User.get_user(username)
    leagues = Leagues.get_all_leagues(account['user_id'], sport, season)
    for league in leagues:
        all_leagues.append(DotMap(league))
    
    return all_leagues


def get_user_df(league_id):
    users = Leagues.get_users(league_id)

    user_list = []
    for index, owner in enumerate(users):
        df_users = pd.DataFrame({
            "manager": owner["display_name"],
            "owner_id": owner["user_id"],
            "team_name": owner["metadata"]["team_name"] if "team_name" in owner["metadata"].keys() else None
        }, index=[index])

        user_list.append(df_users)
        
    return pd.concat(user_list)


def get_roster_df(league_id):
    rosters = Leagues.get_rosters(league_id)

    roster_list = []
    for team in rosters:
        df_roster = pd.DataFrame({
            "roster_id": team["roster_id"],
            "owner_id": team["owner_id"],
            "player_id": team["players"]
        })
        roster_list.append(df_roster)

    return pd.concat(roster_list)


def get_matchup_df(league_id, upper_week):
    matchup_list = []
    for week in range(1, upper_week + 1):
        matchups = Leagues.get_matchups(league_id, week)

        for matchup in matchups:
            df_matchup = pd.DataFrame({
                "roster_id": matchup["roster_id"],
                "matchup_id": matchup["matchup_id"],
                "player_id": matchup["players"],
                "points": matchup["players_points"].values()
            })
            df_matchup["starter"] = df_matchup["player_id"].isin(matchup["starters"])
            df_matchup["gameweek"] = week
            matchup_list.append(df_matchup)

    return pd.concat(matchup_list).reset_index(drop=True)

def get_players_df():
    dict_players = Players.get_all_players()
    df_players = pd.DataFrame(dict_players).transpose()

    return df_players