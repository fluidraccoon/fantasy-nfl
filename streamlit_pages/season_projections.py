import streamlit as st
import altair as alt


def show_season_projections_page(d):
    st.title("Current Standings")

    tabs = st.tabs(["Win Projections", "Projected Season Rank"])

    with tabs[0]:
        win_projections(d)

    with tabs[1]:
        projected_season_rank(d)


def win_projections(d):
    st.markdown(
        """
            ## Win Projections
            The following chart shows the distribution of total expected wins over the season. Any wins so far
            have been included in the calculation, and you would expect the spread to reduce as the season progresses
            and the win totals become more certain.
            """
    )

    step = 20
    overlap = 0
    # chart_width = 400

    chart = (
        alt.Chart(d.df_summary_season, height=step)
        .transform_joinaggregate(mean_of_metric="mean(h2h_wins)", groupby=["manager"])
        .transform_bin(
            "binned_wins", "h2h_wins", bin=alt.Bin(step=1, extent=[0, d.max_gameweek])
        )
        .transform_aggregate(
            value="count()", groupby=["manager", "mean_of_metric", "binned_wins"]
        )
        .transform_impute(
            impute="value",
            groupby=["manager", "mean_of_metric"],
            key="binned_wins",
            value=0,
        )
        .mark_area(
            interpolate="monotone",
            fillOpacity=0.8,
            stroke="lightgray",
            strokeWidth=0.5,
        )
        .encode(
            alt.X("binned_wins:Q", bin="binned", title="Season Wins"),
            alt.Y("value:Q", scale=alt.Scale(range=[step, -step * overlap]), axis=None),
            alt.Fill(
                "mean_of_metric:Q",
                legend=None,
                scale=alt.Scale(scheme="redyellowgreen"),
            ),
            tooltip=[alt.Tooltip("mean_of_metric:Q", title="Average Expected Wins", format=".1f")],
        )
        .facet(
            row=alt.Row(
                "manager:N",
                title=None,
                header=alt.Header(labelAngle=0, labelAlign="left"),
                sort=alt.SortField(field="mean_of_metric", order="descending"),
            )
        )
        .properties(
            title={
                "text": f"Distribution of Season Win Totals - {d.sims} Simulated Seasons",
                "subtitle": f"{d.selected_league}",
            },
            bounds="flush",
        )
        .configure_facet(spacing=0)
        .configure_view(stroke=None)
    )

    st.altair_chart(chart, use_container_width=True)


def projected_season_rank(d):
    st.markdown(
        """
            ## Projected Season Rank
            Using the win totals from the chart above, the following chart shows the likelihood of each team finishing in a given position.
            """
    )
    # Group by season, arrange, and mutate Position
    d.df_summary_season["position"] = (
        d.df_summary_season.groupby("season")
        .apply(
            lambda x: (x["h2h_wins"] + x["points_for"] / 10000).rank(
                method="first", ascending=False
            )
        )
        .reset_index(drop=True)
    )

    d.df_summary_season["avg_position"] = d.df_summary_season.groupby("manager")[
        "position"
    ].transform("mean")
    manager_order = (
        d.df_summary_season.groupby("manager")["avg_position"]
        .mean()
        .reset_index()
        .sort_values("avg_position")
    )

    # Step 2: Create Altair Chart
    chart = (
        alt.Chart(d.df_summary_season)
        .mark_bar()
        .encode(
            x=alt.X(
                "manager:N",
                axis=None,
                sort=alt.Sort(manager_order["manager"].tolist()),
            ),
            y=alt.Y("probability:Q", axis=None),
            color=alt.Color(
                "manager:N",
                legend=alt.Legend(title="Manager", orient="right"),
                sort=manager_order["manager"].tolist(),
                scale=alt.Scale(scheme="paired"),
            ),
            tooltip=[
                alt.Tooltip("manager:N", title="Manager"),
                alt.Tooltip("probability:Q", title="Probability (%)", format=".1f"),
            ],
        )
        .transform_aggregate(
            count="count()",  # Aggregate to count the number of rows
            groupby=["manager"],  # Group by 'category'
        )
        .transform_calculate(
            probability=f"datum.count / {d.sims} * 100"  # Calculate count()/1000
        )
        .properties(width=100, height=60)
        .facet(facet=alt.Facet("position:Q", title="Position"), columns=4, spacing=10)
        .configure_axis(labelAngle=0)
        .configure_view(stroke=None)
        .properties(
            title={
                "text": [f"Final Season Rank - {d.sims} Simulated Seasons"],
                "subtitle": [f"{d.selected_league}"],
                "anchor": "start",
                "fontSize": 16,
            }
        )
    )

    st.altair_chart(chart, use_container_width=True)
