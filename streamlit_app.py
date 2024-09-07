import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

def part1():
    st.title("Fantasy NFL Dashboard 🏈")

    col1, col2 = st.columns([1, 2])
    option = col1.selectbox("Choose a league", ("NFL Dynasty", "UK Dynasty League "))
    # option = st.sidebar.selectbox("Choose a league", ("NFL Dynasty", "UK Dynasty League "))

    st.write(
        "The hunt for the playoffs as it stands. All-play is the record if all teams were to play each other every week. xWins\
        is the number of wins you would have expected so far based on the all-play record. The playoff and bye % is calculated \
        by calculating player scores since 2013 based on their rank and sampling these to give a score for each simulation. The \
        optimal lineup is calculated and then an efficiency score is calculated to give the starting lineup score. 1000 different \
        seasons have been simulated using the wins to date and the remaining fixtures for each team. Strength of roster is taken \
        into account in these calculations."
    )

    league_size = 12 # TODO link up to league
    gameweek_end = 13 # TODO link up, too
    league_name = option

    # df_matchup_schedule = pd.read_csv("Data/df_matchup_schedule.csv")
    df_matchup_schedule1 = pd.read_csv("Data/df_matchup_schedule1.csv")
    df_matchup_schedule2 = pd.read_csv("Data/df_matchup_schedule2.csv")
    df_matchup_schedule = pd.concat([df_matchup_schedule1, df_matchup_schedule2])
    df_matchup_schedule = df_matchup_schedule[df_matchup_schedule["league"]==league_name]
    df_matchup_schedule.drop(columns = ["league"])

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
    # Example Data (modify it with your actual data)
    data = pd.DataFrame({
        'Manager': ['Manager A', 'Manager B', 'Manager C'],
        'Wins': [15, 12, 9],
        'xWins': [13, 10, 11]
    })

    # Reorder managers by Wins * 100 + xWins
    data['Manager'] = pd.Categorical(data['Manager'], categories=data.sort_values(by=['Wins', 'xWins'], ascending=False)['Manager'])

    # Base chart
    base = alt.Chart(data).encode(
        y=alt.Y('Manager:N', sort=alt.EncodingSortField(field='Wins', op='sum', order='descending')),
        color='Manager:N'
    )

    # Points for Wins
    points = base.mark_point(filled=True, size=80, opacity=0.8).encode(
        x=alt.X('Wins:Q', title='Wins')
    )

    # Segments between Wins and xWins
    segments = base.mark_rule(opacity=0.75, strokeWidth=2, strokeCap='round').encode(
        x='Wins:Q',
        x2='xWins:Q'
    )

    # Arrow indicating direction
    arrows = base.mark_point( 
        shape='triangle', 
        size=100, 
        filled=True, 
        opacity=0.75, 
        angle=30 
    ).encode(
        x='Wins:Q',
        x2='xWins:Q'
    )

    # Text annotations (difference between Wins and xWins)
    text = base.mark_text(align='center', dy=-10).encode(
        x=alt.X('(Wins + xWins)/2:Q'),
        text=alt.Text('round(Wins - xWins, 1):Q')
    )

    # Combine layers
    chart = (segments + points + text).properties(
        title={
            'text': 'Schedule Luck',
            'subtitle': 'Difference between H2H Wins and xWins based on All-Play. The arrow shows where you deserve to be.'
        },
        width=600,
        height=300
    ).configure_axis(
        grid=False
    ).configure_title(
        anchor='start'
    )

    chart.display()

    st.altair_chart(chart, use_container_width=True)



df_standings = part1()
part2(df_standings)