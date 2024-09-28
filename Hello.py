import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(
    page_title="NFL Fantasy Dashboard",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

df_matchup_schedule = pd.read_csv("data/df_matchup_schedule.csv")

with st.sidebar:
    st.title("Fantasy NFL Dashboard 🏈")
    users = list(set(df_matchup_schedule["manager"]))
    users.sort()
    user_selection = st.selectbox("Select a username", users, index=users.index("DanCoulton"))
    leagues = list(set(df_matchup_schedule[df_matchup_schedule["manager"]==user_selection]["league"]))
    leagues.sort()
    league_selection = st.selectbox("Select a league", leagues)

df_matchup_schedule = df_matchup_schedule[df_matchup_schedule["league"]==league_selection]
df_matchup_schedule.drop(columns = ["league"])

max_gameweek = max(df_matchup_schedule["gameweek"])
# with st.sidebar:
#     gameweek_start, gameweek_end = st.slider("Select gameweeks", 1, 2, (1, 2))
gameweek_start = 1
gameweek_end = 3
df_matchup_schedule = df_matchup_schedule[
    (df_matchup_schedule["gameweek"] >= gameweek_start) & (df_matchup_schedule["gameweek"] <= gameweek_end)
]

league_size = df_matchup_schedule[df_matchup_schedule["gameweek"]==1]["gameweek"].count()

def part1(df_matchup_schedule):
    st.markdown("## Current Standings")

    st.write(
        "The hunt for the playoffs as it stands. All-play is the record if all teams were to play each other every week.\
        xWins is the number of wins you would have expected so far based on the all-play record. Accuracy is the number of\
        points compared to the maximum possible points."
    )

    df_matchup_schedule["all_play"] = df_matchup_schedule.groupby("gameweek")["points"].rank("max") - 1

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
    df_standings.index = df_standings.index + 1

    def set_background_color(x, league_size):
        if league_size == 12:
            color = "#0080ff" if x.name <=2 else "#79c973" if x.name <=6 else "#ff6666"
        elif league_size == 10:
            color = "#79c973" if x.name <=4 else "#ff6666"

        return [f"background-color: {color}" for i in x]

    st.dataframe(
        df_standings.style\
            .format("{:.0f}", subset=["wins"])\
            .format("{:.2f}", subset=["points"])\
            .format("{:.1f}", subset=["xwins"])\
            .format("{:.1%}", subset=["accuracy"])\
            .apply(lambda x: set_background_color(x, league_size), axis=1)\
            .apply(lambda x: [f"color: white" for i in x], axis=1),
        column_config={
            "manager": st.column_config.TextColumn("Manager", help="Username of team manager"),
            "wins": st.column_config.NumberColumn("Wins", help="Number of wins so far"),
            "points": st.column_config.NumberColumn("Points", help="Number of points so far"),
            "all_play_display": st.column_config.TextColumn("All-Play", help="Wins and losses if you played every team each week"),
            "xwins": st.column_config.TextColumn("xWins", help="Expected number of wins based on the all-play record"),
            "accuracy": st.column_config.ProgressColumn("Accuracy", help="Accuracy of team selection compared to maximum points", min_value=0, max_value=1)
        },
        height=35*len(df_standings)+38
    )

    return df_standings

def part2(df_standings):
    df_standings.loc[df_standings.xwins > df_standings.wins , 'angle'] = 'angle-cat-one'
    df_standings.loc[df_standings.xwins <= df_standings.wins , 'angle'] = 'angle-cat-two'
    df_standings['manager'] = pd.Categorical(df_standings['manager'], categories=df_standings.sort_values(by=['wins', 'xwins'], ascending=False)['manager'])

    base = alt.Chart(df_standings).encode(
        y=alt.Y('manager:N', sort=alt.EncodingSortField(field='wins', op='sum', order='descending'), title="Manager"),
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
                font-size: 20px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.metric('Luckiest Manager(s) 🍀', value=', '.join(max_woe_managers), delta=f"{df_standings["WOE"].max()} more wins than expected", delta_color='normal')
        st.metric('Unluckiest Manager(s) 🐈‍⬛', value=', '.join(min_woe_managers), delta=f"{-df_standings["WOE"].min()} fewer wins than expected", delta_color='inverse')

df_standings = part1(df_matchup_schedule)
part2(df_standings)

