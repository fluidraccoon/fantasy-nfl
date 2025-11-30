import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import types
from streamlit_pages.current_standings import show_current_standings_page
from streamlit_pages.season_projections import show_season_projections_page
from streamlit_pages.playoff_scenarios import show_playoff_scenarios_page
from streamlit_pages.rookie_draft import show_rookie_draft_page
from simulate_table import simulate_table


def create_standings_data(d):
    """Create standings dataframe and attach it to d object"""
    d.df_matchup_schedule["all_play"] = (
        d.df_matchup_schedule.groupby("gameweek")["points"].rank("max") - 1
    )

    playoff_chances = (
        simulate_table(d)
        .groupby(["manager"])
        .agg(playoff=("playoff", "mean"), bye=("bye", "mean"))
        .reset_index()
    )

    df_standings = (
        d.df_matchup_schedule.groupby(["manager"])
        .agg(
            wins=("win", "sum"),
            points=("points", "sum"),
            pp_points=("pp_points", "sum"),
            all_play=("all_play", "sum"),
        )
        .reset_index()
    )
    
    # Add division information for Super Flex Keeper league
    if d.selected_league == "Super Flex Keeper" and hasattr(d, 'df_summary_week'):
        # Get division mapping from summary week data
        division_mapping = d.df_summary_week[['manager', 'division']].drop_duplicates()
        df_standings = df_standings.merge(division_mapping, on='manager', how='left')
    
    df_standings = df_standings.sort_values(
        by=["wins", "points"], ascending=False
    ).reset_index(drop=True)
    df_standings["all_play_display"] = (
        df_standings["all_play"].apply(lambda x: f"{x:.0f}")
        + "-"
        + (d.gameweek_end * (d.league_size - 1) - df_standings["all_play"]).apply(
            lambda x: f"{x:.0f}"
        )
    )
    df_standings["xwins"] = round(df_standings["all_play"] / (d.league_size - 1), 1)
    df_standings["accuracy"] = df_standings["points"] / df_standings["pp_points"] * 100
    df_standings = df_standings.drop(columns=["pp_points", "all_play"])
    df_standings = df_standings.merge(playoff_chances, on="manager", how="left")
    df_standings.index = df_standings.index + 1
    
    d.df_standings = df_standings


def render_sidebar(df_matchup_schedule):
    """Render sidebar with persistent dropdown"""
    st.sidebar.title("Fantasy NFL Dashboard 🏈")

    # Filter out NaN values before sorting
    user_options = sorted(set(df_matchup_schedule["manager"].dropna()))
    selected_user = st.sidebar.selectbox(
        label="Select your username",
        options=user_options,
        key="selected_user",
    )

    league_options = sorted(
        set(
            df_matchup_schedule[df_matchup_schedule["manager"] == selected_user][
                "league"
            ]
        )
    )
    selected_league = st.sidebar.selectbox(
        label="Select a league",
        options=league_options,
        key="selected_league",
    )

    # st.sidebar.markdown("---")

    return selected_user, selected_league


def initialize_session_state():
    if "selected_user" not in st.session_state:
        st.session_state.selected_user = "DanCoulton"

    if "selected_league" not in st.session_state:
        st.session_state.selected_league = "House of Hard Knocks"


def main():
    st.set_page_config(
        page_title="NFL Fantasy Dashboard",
        page_icon="🏈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    d = types.SimpleNamespace()

    # Use attribute assignment with SimpleNamespace
    d.df_matchup_schedule = pd.read_csv("data/df_matchup_schedule.csv")
    d.df_summary_season = pd.read_csv("data/df_summary_season.csv")
    d.sims = d.df_summary_season["season"].max()

    # Initial sidebar render to get user selections
    d.selected_user, d.selected_league = render_sidebar(d.df_matchup_schedule)

    # Filter data by selected league
    d.df_matchup_schedule = (
        d.df_matchup_schedule[d.df_matchup_schedule["league"] == d.selected_league]
        .drop(columns=["league"])
        .reset_index(drop=True)
    )
    d.df_summary_week = pd.read_csv(f"data/df_summary_week_{d.selected_league}.csv")
    d.df_summary_season = (
        d.df_summary_season[d.df_summary_season["league"] == d.selected_league]
        .drop(columns=["league"])
        .reset_index(drop=True)
    )

    d.max_gameweek = max(d.df_matchup_schedule["gameweek"])
    # with st.sidebar:
    #     gameweek_start, gameweek_end = st.slider("Select gameweeks", 1, 2, (1, 2))
    d.season = 2025
    d.gameweek_start = 1
    d.gameweek_end = 12
    d.df_matchup_schedule = d.df_matchup_schedule[
        (d.df_matchup_schedule["gameweek"] >= d.gameweek_start)
        & (d.df_matchup_schedule["gameweek"] <= d.gameweek_end)
    ]

    d.league_size = d.df_matchup_schedule[d.df_matchup_schedule["gameweek"] == 1][
        "gameweek"
    ].count()

    # Create standings data that will be used across multiple pages
    create_standings_data(d)
    
    # Add playoff probabilities to sidebar now that standings data is available
    user_standings = d.df_standings[d.df_standings["manager"] == d.selected_user]
    if not user_standings.empty:
        playoff_prob = user_standings["playoff"].iloc[0]
        st.sidebar.markdown("### Your Playoff Outlook 🎯")
        
        # Create placeholder containers that can be updated by playoff scenarios page
        playoff_metric_container = st.sidebar.empty()
        
        # Store containers in session state for playoff scenarios page to update
        st.session_state.playoff_metric_container = playoff_metric_container
        
        # Set initial metrics (will be replaced if on playoff scenarios page)
        with playoff_metric_container.container():
            st.markdown("<style>div[data-testid='stMetric'] div { font-size: 15px; }</style>", unsafe_allow_html=True)
            if d.league_size == 12:
                # Two columns for playoff and bye (12-team leagues)
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="Playoff %",
                        value=f"{playoff_prob:.1%}",
                        help="Your chance of making the playoffs based on current performance and remaining schedule"
                    )
                
                with col2:
                    bye_prob = user_standings["bye"].iloc[0]
                    st.metric(
                        label="Bye %",
                        value=f"{bye_prob:.1%}",
                        help="Your chance of getting a first-round bye in the playoffs"
                    )
            else:
                # Single column for playoff only (10-team leagues)
                st.metric(
                    label="Playoff %",
                    value=f"{playoff_prob:.1%}",
                    help="Your chance of making the playoffs based on current performance and remaining schedule"
                )
    
        
        # st.sidebar.markdown("---")

    initialize_session_state()
    pages = [
        st.Page(
            lambda: show_current_standings_page(d),
            title="Current Standings",
            icon="🏅",
            url_path="current-standings",
        ),
        st.Page(
            lambda: show_season_projections_page(d),
            title="Season Projections",
            icon="📈",
            url_path="season-projections",
        ),
        st.Page(
            lambda: show_playoff_scenarios_page(d),
            title="Playoff Scenarios",
            icon="❓",  # Question mark for "scenarios" or uncertainty
            url_path="playoff-scenarios",
        ),
    ]
    
    # Only add rookie draft page for dynasty leagues
    if "Dynasty" in d.selected_league:
        pages.append(
            st.Page(
                lambda: show_rookie_draft_page(d),
                title=f"{d.season + 1} Rookie Draft",
                icon="🎯",  # Target icon for draft strategy
                url_path="rookie-draft-2026",
            )
        )

    pg = st.navigation(pages)

    pg.run()


if __name__ == "__main__":
    main()
