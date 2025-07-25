import numpy as np


def simulate_table(d):
    # Grouping by 'season' and 'Manager' and aggregating
    table_sim = d.df_summary_week.groupby(
        ["season", "manager", "division"], as_index=False
    ).agg(
        wins=("selected_win", "sum"),
        points=("selected_pts", "sum"),
        potential_wins=("selected_pp_win", "sum"),
        potential_points=("selected_pp", "sum"),
        upcoming=("nxt_wk_win", "sum"),
    )

    # Sorting within each group by 'Points' (descending)
    table_sim["position"] = (
        table_sim.sort_values(
            ["season", "division", "wins", "points"],
            ascending=[True, True, False, False],
        )
        .groupby(["season", "division"])
        .cumcount()
        + 1
    )

    # Adding 'Playoff' column based on condition (if Position <= 6)
    if d.selected_league == "Super Flex Keeper":
        table_sim_wc = table_sim.copy()
        table_sim_wc["wins"] = np.where(
            table_sim_wc["position"] <= 2, 0, table_sim_wc["wins"]
        )
        table_sim_wc["points"] = np.where(
            table_sim_wc["position"] <= 2, 0, table_sim_wc["points"]
        )
        table_sim["overall_position"] = (
            table_sim_wc.sort_values(
                ["season", "wins", "points"], ascending=[True, False, False]
            )
            .groupby(["season"])
            .cumcount()
            + 1
        )
        table_sim["playoff"] = np.where(
            (table_sim["position"] <= 2) | (table_sim["overall_position"] <= 2), 1, 0
        )
    else:
        table_sim["playoff"] = np.where(
            table_sim["position"]
            <= (6 if d.league_size == 12 else 4 if d.league_size == 10 else 1),
            1,
            0,
        )

    # Adding 'Bye' column based on condition (if Position <= 2)
    if d.selected_league == "Super Flex Keeper":
        table_sim["bye"] = np.where(table_sim["position"] == 1, 1, 0)
    else:
        table_sim["bye"] = np.where(table_sim["position"] <= 2, 1, 0)

    # Re-arranging again, sorting by 'Playoff' and 'PPoints' within each 'season'
    table_sim = table_sim.sort_values(
        ["season", "playoff", "potential_points"], ascending=[True, False, False]
    )

    # Adding 'Draft_Pos' column based on new ranking after re-arranging
    table_sim["draft_pos"] = table_sim.groupby("season").cumcount() + 1

    # Removing the groupby index (like ungroup in R)
    table_sim = table_sim.reset_index(drop=True)

    return table_sim
