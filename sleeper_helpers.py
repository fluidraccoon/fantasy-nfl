import pandas as pd
import requests
from sleeperpy import User, Leagues, Players, Avatars


def get_league_dict(username, season, sport="nfl"):
    all_leagues = []
    account = User.get_user(username)
    leagues = Leagues.get_all_leagues(account["user_id"], sport, season)
    for league in leagues:
        all_leagues.append(league)

    return all_leagues


def get_user_df(league_id):
    users = Leagues.get_users(league_id)

    user_list = []
    for index, owner in enumerate(users):
        df_users = pd.DataFrame(
            {
                "manager": owner["display_name"],
                "owner_id": owner["user_id"],
                "team_name": (
                    owner["metadata"]["team_name"]
                    if "team_name" in owner["metadata"].keys()
                    else None
                ),
            },
            index=[index],
        )

        user_list.append(df_users)

    return pd.concat(user_list)


def get_roster_df(league_id):
    rosters = Leagues.get_rosters(league_id)

    roster_list = []
    for team in rosters:
        df_roster = pd.DataFrame(
            {
                "roster_id": team["roster_id"],
                "owner_id": team["owner_id"],
                "player_id": team["players"],
                "division": (
                    team["settings"]["division"]
                    if "division" in team["settings"]
                    else 1
                ),
            }
        )
        roster_list.append(df_roster)

    return pd.concat(roster_list)


def get_matchup_df(league_id, final_gameweek):
    matchup_list = []
    for week in range(1, final_gameweek + 1):
        matchups = Leagues.get_matchups(league_id, week)

        for matchup in matchups:
            df_matchup = pd.DataFrame(
                {
                    "roster_id": matchup["roster_id"],
                    "matchup_id": matchup["matchup_id"],
                    "player_id": matchup["players"],
                    "points": matchup["players_points"].values(),
                }
            )
            if matchup["starters"]:
                df_matchup["starter"] = df_matchup["player_id"].isin(
                    matchup["starters"]
                )
            else:
                df_matchup["starter"] = 0
            df_matchup["gameweek"] = week
            matchup_list.append(df_matchup)

    return pd.concat(matchup_list).reset_index(drop=True)


def get_players_df():
    dict_players = Players.get_all_players()
    df_players = pd.DataFrame(dict_players).transpose()

    return df_players


def get_traded_picks(league_id):
    """Get traded draft picks for a league."""
    try:
        picks_raw = requests.get(f"https://api.sleeper.app/v1/league/{league_id}/traded_picks")
        picks_raw.raise_for_status()  # Raises an exception for bad status codes
        picks_df = pd.DataFrame(picks_raw.json())
        
        return picks_df
    
    except Exception as e:
        print(f"Error getting traded picks: {e}")
        return pd.DataFrame()


def get_draft_order_by_potential_points(league_id, season_standings):
    """
    Calculate draft order based on potential points (lowest gets first pick).
    
    Args:
        league_id: Sleeper league ID
        season_standings: DataFrame with current standings including potential points
    
    Returns:
        DataFrame with draft order and pick ownership
    """
    try:
        # Get traded picks
        traded_picks_df = get_traded_picks(league_id)
        
        # Get users to map roster_id to manager names
        users_df = get_user_df(league_id)
        
        # Sort standings by potential points (ascending - worst team gets pick 1)
        draft_order = season_standings.copy()
        
        # If pp_points column exists, use it; otherwise use regular points
        sort_column = 'pp_points' if 'pp_points' in draft_order.columns else 'points'
        draft_order = draft_order.sort_values(sort_column, ascending=True).reset_index(drop=True)
        
        # Add draft pick numbers
        draft_order['original_pick'] = range(1, len(draft_order) + 1)
        
        # Create draft picks DataFrame for multiple rounds
        all_picks = []
        num_teams = len(draft_order)
        
        for round_num in range(1, 5):  # Show first 4 rounds
            round_picks = draft_order.copy()
            round_picks['round'] = round_num
            round_picks['pick_number'] = round_picks['original_pick']
            round_picks['overall_pick'] = (round_num - 1) * num_teams + round_picks['pick_number']
            round_picks['current_owner'] = round_picks['manager']  # Default owner
            
            # Apply traded picks for this round
            if not traded_picks_df.empty:
                traded_this_round = traded_picks_df[
                    (traded_picks_df['season'] == 2026) & 
                    (traded_picks_df['round'] == round_num)
                ]
                
                for _, trade in traded_this_round.iterrows():
                    # Find the original pick holder and update current owner
                    original_roster_mask = round_picks['roster_id'] == trade['previous_owner_id']
                    if original_roster_mask.any():
                        # Get new owner name
                        new_owner = users_df[users_df['owner_id'] == trade['owner_id']]['manager']
                        if not new_owner.empty:
                            round_picks.loc[original_roster_mask, 'current_owner'] = new_owner.iloc[0]
                            round_picks.loc[original_roster_mask, 'traded'] = True
            
            # Add traded indicator
            if 'traded' not in round_picks.columns:
                round_picks['traded'] = False
            
            all_picks.append(round_picks)
        
        return pd.concat(all_picks, ignore_index=True)
    
    except Exception as e:
        print(f"Error calculating draft order: {e}")
        return season_standings.copy()  # Return original if error
