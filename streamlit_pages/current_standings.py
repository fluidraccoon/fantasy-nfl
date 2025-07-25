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

    def set_background_color(x, league_size):
        if league_size == 12:
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
