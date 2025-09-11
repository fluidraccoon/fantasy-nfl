import streamlit as st
import pandas as pd
import altair as alt


def show_current_standings_page(d):
    st.title("Current Standings")

    tabs = st.tabs(["Standings", "Wins over Expectation"])

    with tabs[0]:
        current_standings(d)

    with tabs[1]:
        wins_over_expectation(d)


def current_standings(d):
    st.markdown(
        """
        The hunt for the playoffs as it stands. All-play is the record if all teams were to play each other every week.
        xWins is the number of wins you would have expected so far based on the all-play record. Accuracy is the number of
        points compared to the maximum possible points. The playoff{} % is calculated by calculating player scores 
        since 2016 based on their rank and sampling these to give a score for each simulation. The optimal lineup is 
        calculated and then an efficiency score is calculated to give the starting lineup score. {} different seasons have
        been simulated using the wins to date and the remaining fixtures for each team. Strength of roster is taken into account
        in these calculations.
        """.format(
            " and bye" if d.league_size == 12 else "", d.sims
        )
    )

    df_standings = d.df_standings.copy()

    def determine_super_flex_playoff_status(df_standings_full):
        """Determine playoff status for Super Flex Keeper league based on current standings"""
        # Get division winners (top of each division)
        division_winners = []
        for division in [1, 2]:
            div_teams = df_standings_full[df_standings_full["division"] == division]
            if not div_teams.empty:
                div_winner = div_teams.sort_values(["wins", "points"], ascending=False).iloc[0]
                division_winners.append(div_winner["manager"])
        
        # Get wildcard candidates (everyone except division winners)
        wildcard_candidates = df_standings_full[~df_standings_full["manager"].isin(division_winners)].copy()
        
        # Sort wildcards by overall record
        wildcard_candidates = wildcard_candidates.sort_values(["wins", "points"], ascending=False)
        
        # Top 4 wildcards make playoffs
        wildcard_playoff_teams = wildcard_candidates.head(4)["manager"].tolist()
        
        return {
            "division_winners": division_winners,
            "wildcard_teams": wildcard_playoff_teams,
            "playoff_teams": division_winners + wildcard_playoff_teams
        }

    def set_background_color(x, league_size, is_super_flex=False, manager_name=None, 
                           division_position=None, playoff_status=None):
        if is_super_flex and playoff_status is not None and manager_name is not None:
            # For Super Flex Keeper, use actual playoff structure
            if manager_name in playoff_status["division_winners"]:
                # Division winners get blue (bye)
                color = "#0080ff"
            elif manager_name in playoff_status["wildcard_teams"]:
                # Wildcard playoff teams get green
                color = "#79c973"
            else:
                # Non-playoff teams get red
                color = "#ff6666"
        elif league_size == 12:
            color = (
                "#0080ff" if x.name <= 2 else "#79c973" if x.name <= 6 else "#ff6666"
            )
        elif league_size == 10:
            color = "#79c973" if x.name <= 4 else "#ff6666"
        return [f"background-color: {color}" for i in x]

    config_columns = {
        "manager": st.column_config.TextColumn(
            "Manager", help="Username of team manager"
        ),
        "wins": st.column_config.NumberColumn("Wins", help="Number of wins so far"),
        "points": st.column_config.NumberColumn(
            "Points", help="Number of points so far"
        ),
        "all_play_display": st.column_config.TextColumn(
            "All-Play", help="Wins and losses if you played every team each week"
        ),
        "xwins": st.column_config.TextColumn(
            "xWins", help="Expected number of wins based on the all-play record"
        ),
        "accuracy": st.column_config.ProgressColumn(
            "Accuracy",
            help="Accuracy of team selection compared to maximum points",
            format="%.1f %%",
            min_value=0,
            max_value=100,
        ),
        "playoff": st.column_config.NumberColumn(
            "Playoff %", help="% chance of team making the playoffs"
        ),
        "bye": st.column_config.NumberColumn(
            "Bye %", help="% chance of team getting a first-round bye"
        ),
    }
    if d.league_size == 10:
        del config_columns["bye"]
        df_standings = df_standings.drop(columns=["bye"])

    # Check if this is Super Flex Keeper league with divisions
    if d.selected_league == "Super Flex Keeper" and "division" in df_standings.columns:
        # Determine playoff status for the entire league
        playoff_status = determine_super_flex_playoff_status(df_standings)
        
        # Display separate tables for each division
        divisions = sorted(df_standings["division"].unique())
        division_names = {1: "North Division", 2: "South Division"}
        
        for division in divisions:
            division_name = division_names.get(division, f"Division {division}")
            st.subheader(division_name)
            
            # Filter standings for this division
            division_standings = df_standings[df_standings["division"] == division].copy()
            
            # Sort by wins, then points within division
            division_standings = division_standings.sort_values(
                by=["wins", "points"], ascending=False
            ).reset_index(drop=True)
            division_standings.index = division_standings.index + 1
            
            # Remove division column for display
            display_standings = division_standings.drop(columns=["division"])
            
            # Create a custom styling function for this division
            def apply_super_flex_colors(row):
                return set_background_color(
                    row, d.league_size, is_super_flex=True, 
                    manager_name=row["manager"], playoff_status=playoff_status
                )
            
            st.dataframe(
                display_standings.style.format("{:.0f}", subset=["wins"])
                .format("{:.2f}", subset=["points"])
                .format("{:.1f}", subset=["xwins"])
                .format(
                    "{:.1%}",
                    subset=(
                        ["accuracy", "playoff", "bye"]
                        if d.league_size == 12
                        else ["accuracy", "playoff"]
                    ),
                )
                .apply(apply_super_flex_colors, axis=1)
                .apply(lambda x: [f"color: white" for i in x], axis=1),
                column_config=config_columns,
                height=35 * len(display_standings) + 38,
                use_container_width=False,
            )
            
            # Add some space between divisions
            st.write("")
    else:
        # Standard single table display
        st.dataframe(
            df_standings.style.format("{:.0f}", subset=["wins"])
            .format("{:.2f}", subset=["points"])
            .format("{:.1f}", subset=["xwins"])
            .format(
                "{:.1%}",
                subset=(
                    ["accuracy", "playoff", "bye"]
                    if d.league_size == 12
                    else ["accuracy", "playoff"]
                ),
            )
            .apply(lambda x: set_background_color(x, d.league_size), axis=1)
            .apply(lambda x: [f"color: white" for i in x], axis=1),
            column_config=config_columns,
            height=35 * len(df_standings) + 38,
            use_container_width=False,
        )


def wins_over_expectation(d):
    df_standings = d.df_standings.copy()
    # Calculate angle based on xWins and wins
    df_standings.loc[df_standings.xwins > df_standings.wins, "angle"] = "angle-cat-one"
    df_standings.loc[df_standings.xwins <= df_standings.wins, "angle"] = "angle-cat-two"
    df_standings["manager"] = pd.Categorical(
        df_standings["manager"],
        categories=df_standings.sort_values(by=["wins", "xwins"], ascending=False)[
            "manager"
        ],
    )

    base = alt.Chart(df_standings).encode(
        y=alt.Y(
            "manager:N",
            sort=alt.EncodingSortField(field="wins", op="sum", order="descending"),
            title=None,
        ),
        color=alt.Color("manager:N", legend=None),
    )

    win_circle = base.mark_point(filled=True, size=80, opacity=0.8).encode(
        x=alt.X(
            "wins:Q",
            title="Wins",
            scale=alt.Scale(nice=False),
            axis=alt.Axis(
                tickCount=(df_standings["wins"].max() - df_standings["wins"].min() + 1),
                tickMinStep=1,
            ),
        ),
        tooltip=alt.value(None),
    )

    segments = base.mark_rule(opacity=0.75, strokeWidth=2, strokeCap="round").encode(
        x="wins:Q", x2="xwins:Q"
    )

    xwin_arrow = base.mark_point(
        shape="triangle", size=100, filled=True, opacity=0.75
    ).encode(
        x="xwins:Q",
        angle=alt.Angle(
            "angle:N",
            scale=alt.Scale(domain=["angle-cat-one", "angle-cat-two"], range=[90, 270]),
        ),
        tooltip=alt.value(None),
    )

    chart = (
        (segments + win_circle + xwin_arrow)
        .encode(
            tooltip=[
                alt.Tooltip("wins", title="Wins"),
                alt.Tooltip("xwins", title="xWins"),
                alt.Tooltip("manager", title="Manager"),
            ]
        )
        .properties(
            title={
                "text": "Schedule Luck",
                "subtitle": "Difference between H2H Wins and xWins based on All-Play. The arrow shows where you deserve to be.",
            }
        )
        .configure_axis(grid=False)
        .configure_title(anchor="start")
    )

    st.markdown(
        """
        Wins over expectation (WOE) looks at the relationship between actual wins and all-play wins.\
        This shows how lucky or unlucky a team has been with the schedule.
        """
    )

    luck_col1, luck_col2, luck_col3 = st.columns([2, 0.3, 0.7])

    with luck_col1:
        st.altair_chart(chart, use_container_width=True)

    with luck_col3:
        for i in range(4):
            st.write("")
        df_standings["WOE"] = df_standings["wins"] - df_standings["xwins"]
        max_woe_managers = df_standings[
            df_standings["WOE"] == df_standings["WOE"].max()
        ]["manager"].tolist()
        min_woe_managers = df_standings[
            df_standings["WOE"] == df_standings["WOE"].min()
        ]["manager"].tolist()
        st.write(
            """
            <style>
            [data-testid="stMetricDelta"] svg {
                display: none;
            }
            [data-testid="stMetricValue"] {
                font-size: 15px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.metric(
            "Luckiest Manager(s) 🍀",
            value=", ".join(max_woe_managers),
            delta=f"{round(df_standings['WOE'].max(), 1)} more wins than expected",
            delta_color="normal",
        )
        st.metric(
            "Unluckiest Manager(s) 🐈‍⬛",
            value=", ".join(min_woe_managers),
            delta=f"{-round(df_standings['WOE'].min(), 1)} fewer wins than expected",
            delta_color="inverse",
        )
