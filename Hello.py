import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

st.set_page_config(
    page_title="NFL Fantasy Dashboard",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

df_matchup_schedule = pd.read_csv("data/df_matchup_schedule.csv")
df_summary_week = pd.read_csv("data/df_summary_week.csv")
df_summary_season = pd.read_csv("data/df_summary_season.csv")
sims = df_summary_season["season"].max()

with st.sidebar:
    st.title("Fantasy NFL Dashboard 🏈")
    users = list(set(df_matchup_schedule["manager"]))
    users.sort()
    user_selection = st.selectbox("Select your username", users, index=users.index("DanCoulton"))
    leagues = list(set(df_matchup_schedule[df_matchup_schedule["manager"]==user_selection]["league"]))
    leagues.sort()
    league_selection = st.selectbox("Select a league", leagues)

df_matchup_schedule = df_matchup_schedule[df_matchup_schedule["league"]==league_selection]\
    .drop(columns = ["league"]).reset_index(drop=True)
df_summary_week = df_summary_week[df_summary_week["league"]==league_selection]\
    .drop(columns = ["league"]).reset_index(drop=True)
df_summary_season = df_summary_season[df_summary_season["league"]==league_selection]\
    .drop(columns = ["league"]).reset_index(drop=True)

max_gameweek = max(df_matchup_schedule["gameweek"])
# with st.sidebar:
#     gameweek_start, gameweek_end = st.slider("Select gameweeks", 1, 2, (1, 2))
gameweek_start = 1
gameweek_end = 5
df_matchup_schedule = df_matchup_schedule[
    (df_matchup_schedule["gameweek"] >= gameweek_start) & (df_matchup_schedule["gameweek"] <= gameweek_end)
]

league_size = df_matchup_schedule[df_matchup_schedule["gameweek"]==1]["gameweek"].count()

def simulate_table():
    # Grouping by 'season' and 'Manager' and aggregating
    table_sim = (df_summary_week.groupby(['season', 'manager'], as_index=False)
                .agg(Wins=('selected_win', 'sum'),
                    Points=('selected_pts', 'sum'),
                    PWins=('selected_pp_win', 'sum'),
                    PPoints=('selected_pp', 'sum'),
                    Upcoming=('nxt_wk_win', 'sum'))
                )

    # Sorting within each group by 'Points' (descending)
    table_sim = table_sim.sort_values(['season', 'Points'], ascending=[True, False])

    # Adding 'Position' column based on rank within each 'season'
    table_sim['Position'] = table_sim.groupby('season')['Points'].rank(method='first', ascending=False).astype(int)

    # Adding 'Playoff' column based on condition (if Position <= 6)
    table_sim['Playoff'] = np.where(table_sim['Position'] <= (6 if league_size==12 else 4 if league_size==10 else 1), 1, 0)

    # Adding 'Bye' column based on condition (if Position <= 2)
    table_sim['Bye'] = np.where(table_sim['Position'] <= 2, 1, 0)

    # Re-arranging again, sorting by 'Playoff' and 'PPoints' within each 'season'
    table_sim = table_sim.sort_values(['season', 'Playoff', 'PPoints'], ascending=[True, False, False])

    # Adding 'Draft_Pos' column based on new ranking after re-arranging
    table_sim['Draft_Pos'] = table_sim.groupby('season').cumcount() + 1

    # Removing the groupby index (like ungroup in R)
    table_sim = table_sim.reset_index(drop=True)

    return table_sim

def part1(df_matchup_schedule):
    st.markdown("## Current Standings")

    st.write(
        "The hunt for the playoffs as it stands. All-play is the record if all teams were to play each other every week.\
        xWins is the number of wins you would have expected so far based on the all-play record. Accuracy is the number of\
        points compared to the maximum possible points."
    )

    df_matchup_schedule["all_play"] = df_matchup_schedule.groupby("gameweek")["points"].rank("max") - 1
    
    playoff_chances = simulate_table().groupby(["manager"]).agg(
        playoff=("Playoff", "mean"),
        bye=("Bye", "mean")
    ).reset_index()

    df_standings = df_matchup_schedule.groupby(["manager"]).agg(
        wins=("win", "sum"),
        points=("points", "sum"),
        pp_points=("pp_points", "sum"),
        all_play=("all_play", "sum"),
    ).reset_index()
    df_standings = df_standings.sort_values(by=["wins", "points"], ascending=False).reset_index(drop=True)
    df_standings["all_play_display"] = df_standings["all_play"].apply(lambda x: f"{x:.0f}") + "-" + (gameweek_end * (league_size - 1) - df_standings["all_play"]).apply(lambda x: f"{x:.0f}") 
    df_standings["xwins"] = round(df_standings["all_play"] / (league_size - 1), 1)
    df_standings["accuracy"] = df_standings["points"] / df_standings["pp_points"]
    df_standings = df_standings.drop(columns=["pp_points", "all_play"])
    df_standings = df_standings.merge(playoff_chances, on="manager", how="left")
    df_standings.index = df_standings.index + 1
    # df_standings = df_standings.drop(columns=["bye"])
    

    def set_background_color(x, league_size):
        if league_size == 12:
            color = "#0080ff" if x.name <=2 else "#79c973" if x.name <=6 else "#ff6666"
        elif league_size == 10:
            color = "#79c973" if x.name <=4 else "#ff6666"

        return [f"background-color: {color}" for i in x]
    
    config_columns = {
        "manager": st.column_config.TextColumn("Manager", help="Username of team manager"),
        "wins": st.column_config.NumberColumn("Wins", help="Number of wins so far"),
        "points": st.column_config.NumberColumn("Points", help="Number of points so far"),
        "all_play_display": st.column_config.TextColumn("All-Play", help="Wins and losses if you played every team each week"),
        "xwins": st.column_config.TextColumn("xWins", help="Expected number of wins based on the all-play record"),
        "accuracy": st.column_config.ProgressColumn("Accuracy", help="Accuracy of team selection compared to maximum points", min_value=0, max_value=1),
        "playoff": st.column_config.NumberColumn("Playoff %", help="% chance of team making the playoffs"),
        "bye": st.column_config.NumberColumn("Bye %", help="% chance of team getting a first-round bye"),
    }
    if league_size==10:
        del config_columns["bye"]
        df_standings = df_standings.drop(columns=["bye"])

    st.dataframe(
        df_standings.style\
            .format("{:.0f}", subset=["wins"])\
            .format("{:.2f}", subset=["points"])\
            .format("{:.1f}", subset=["xwins"])\
            .format("{:.1%}", subset=["accuracy", "playoff", "bye"] if league_size==12 else ["accuracy", "playoff"])\
            .apply(lambda x: set_background_color(x, league_size), axis=1)\
            .apply(lambda x: [f"color: white" for i in x], axis=1),
        column_config=config_columns,
        height=35*len(df_standings)+38
    )

    return df_standings

def part2(df_standings):
    df_standings.loc[df_standings.xwins > df_standings.wins , 'angle'] = 'angle-cat-one'
    df_standings.loc[df_standings.xwins <= df_standings.wins , 'angle'] = 'angle-cat-two'
    df_standings['manager'] = pd.Categorical(df_standings['manager'], categories=df_standings.sort_values(by=['wins', 'xwins'], ascending=False)['manager'])

    base = alt.Chart(df_standings).encode(
        y=alt.Y('manager:N', sort=alt.EncodingSortField(field='wins', op='sum', order='descending'), title=None),
        color=alt.Color('manager:N', legend=None)
    )

    win_circle = base.mark_point(filled=True, size=80, opacity=0.8).encode(
        x=alt.X('wins:Q', title='Wins'),
        tooltip=alt.value(None)
    )

    # Segments between Wins and xWins
    segments = base.mark_rule(opacity=0.75, strokeWidth=2, strokeCap='round').encode(
        x='wins:Q',
        x2='xwins:Q'
    )

    xwin_arrow = base.mark_point( 
        shape='triangle', 
        size=100, 
        filled=True, 
        opacity=0.75
    ).encode(
        x='xwins:Q',
        angle=alt.Angle('angle:N', scale=alt.Scale(domain=['angle-cat-one', 'angle-cat-two'], range=[90, 270])),
        tooltip=alt.value(None)
    )

    chart = (segments + win_circle + xwin_arrow).encode(tooltip=[
        alt.Tooltip("wins", title="Wins"),
        alt.Tooltip("xwins", title="xWins"),
        alt.Tooltip("manager", title="Manager")
    ]).properties(
        title={
            'text': 'Schedule Luck',
            'subtitle': 'Difference between H2H Wins and xWins based on All-Play. The arrow shows where you deserve to be.'
        }
    ).configure_axis(grid=False).configure_title(anchor='start')
    
    st.markdown(
        """
        ## Wins over Expectation
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
        max_woe_managers = df_standings[df_standings["WOE"]==df_standings["WOE"].max()]["manager"].tolist()
        min_woe_managers = df_standings[df_standings["WOE"]==df_standings["WOE"].min()]["manager"].tolist()
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

        st.metric('Luckiest Manager(s) 🍀', value=', '.join(max_woe_managers), delta=f"{round(df_standings["WOE"].max(), 1)} more wins than expected", delta_color='normal')
        st.metric('Unluckiest Manager(s) 🐈‍⬛', value=', '.join(min_woe_managers), delta=f"{-round(df_standings["WOE"].min(), 1)} fewer wins than expected", delta_color='inverse')

df_standings = part1(df_matchup_schedule)
part2(df_standings)

def part3(df_summary_season):
    step = 20
    overlap = 0
    # chart_width = 400
    
    chart = alt.Chart(df_summary_season, height=step).transform_joinaggregate(
        mean_of_metric='mean(h2h_wins)', groupby=['manager']
    ).transform_bin(
        'binned_wins', 'h2h_wins', bin=alt.Bin(step=1, extent=[0, max_gameweek])
    ).transform_aggregate(
        value='count()', groupby=['manager', 'mean_of_metric', 'binned_wins']
    ).transform_impute(
        impute='value', groupby=['manager', 'mean_of_metric'], key='binned_wins', value=0
    ).mark_area(
        interpolate='monotone',
        fillOpacity=0.8,
        stroke='lightgray',
        strokeWidth=0.5
    ).encode(
        alt.X("binned_wins:Q", bin="binned", title="Season Wins"),
        alt.Y("value:Q", scale=alt.Scale(range=[step, -step * overlap]), axis=None),
        alt.Fill("mean_of_metric:Q", legend=None, scale=alt.Scale(scheme="redyellowgreen")),
        tooltip=[alt.Tooltip("mean_of_metric:Q", title="Average Expected Wins")]
    ).facet(
        row=alt.Row(
            'manager:N',
            title=None,
            header = alt.Header(labelAngle=0, labelAlign='left'),
            sort=alt.SortField(field='mean_of_metric', order='descending')
        )
    ).properties(
        title={
            'text': f'Distribution of Season Win Totals - {sims} Simulated Seasons',
            'subtitle': f'{league_selection}'
        },
        bounds='flush'
    ).configure_facet(
        spacing=0
    ).configure_view(
        stroke=None
    )
    
    chart

def part4(df_summary_season):
    # Group by season, arrange, and mutate Position
    df_summary_season['position'] = df_summary_season.groupby('season').apply(
        lambda x: (x['h2h_wins'] + x['points_for']/10000).rank(method='first', ascending=False)
    ).reset_index(drop=True)
    
    df_summary_season['avg_position'] = df_summary_season.groupby('manager')['position'].transform('mean')
    manager_order = df_summary_season.groupby('manager')['avg_position'].mean().reset_index().sort_values('avg_position')
    
    # Step 2: Create Altair Chart
    chart2 = alt.Chart(df_summary_season).mark_bar().encode(
        x=alt.X('manager:N', axis=None, sort=alt.Sort(manager_order['manager'].tolist())),
        y=alt.Y('count():Q', axis=None),
        color=alt.Color('manager:N', legend=alt.Legend(
            title="Manager",
            orient="right"
        ), sort=manager_order['manager'].tolist())
    ).properties(
        width=100,
        height=60
    ).facet(
        facet='position:Q',
        columns=4,
        spacing=10
    ).configure_axis(
        labelAngle=0
    ).configure_view(
        stroke=None
    ).properties(
        title={
            'text': [f'Final Season Rank - {sims} Simulated Seasons'],
            'subtitle': [f'{league_selection}'],
            'anchor': 'start',  # Align title to the start
            'fontSize': 16
        }
    )
    
    chart2

part3(df_summary_season)
part4(df_summary_season)