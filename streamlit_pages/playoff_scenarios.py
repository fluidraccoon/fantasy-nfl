import streamlit as st
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

def show_playoff_scenarios_page(d):
    st.title("Playoff Scenarios")
    
    playoff_scenarios(d)

# Data processing functions
# Add caching decorator for expensive operations
@st.cache_data
def prepare_remaining_matchups_cached(df_summary_week_hash, gameweek_end):
    """Cached version of prepare_remaining_matchups to avoid recomputation."""
    # Reconstruct the actual dataframe from hash (this is a simplified approach)
    # In practice, you'd pass the actual dataframe but use hash for cache key
    return prepare_remaining_matchups_internal(df_summary_week_hash, gameweek_end)


def prepare_remaining_matchups_internal(df_summary_week, gameweek_end):
    """Internal function for preparing remaining matchups data."""
    # Select only required columns early to reduce memory footprint
    required_columns = [
        "season", "matchup_id", "roster_id", "opponent_id", 
        "team_score", "opponent_score", "manager", "week", 
        "division", "selected_win", "selected_pts"
    ]
    
    # Filter and select in one operation
    df_remaining = df_summary_week[
        (df_summary_week["week"] > gameweek_end) & 
        df_summary_week[required_columns].notna().all(axis=1)
    ][required_columns]
    
    # Use more efficient pivot operation
    df_matchups = (df_remaining[["matchup_id", "week", "manager"]]
                   .drop_duplicates()
                   .assign(idx=lambda x: x.groupby(["matchup_id", "week"]).cumcount())
                   .pivot_table(index=["matchup_id", "week"], 
                               columns="idx", 
                               values="manager", 
                               aggfunc='first')
                   .reset_index()
                   .assign(adjusted_winner=None))
    
    return df_remaining, df_matchups


def prepare_remaining_matchups(df_summary_week, gameweek_end):
    """Prepare and structure remaining matchups data with caching."""
    # Create a simple hash of the relevant data for caching
    cache_key = f"{len(df_summary_week)}_{gameweek_end}_{df_summary_week['week'].max()}"
    
    # For now, call internal function directly (in production, you'd implement proper caching)
    return prepare_remaining_matchups_internal(df_summary_week, gameweek_end)


def calculate_adjusted_playoff_chance(df, league_config):
    """Calculate playoff chances based on adjusted wins - optimized version."""
    # Use vectorized operations instead of multiple groupby calls
    table_sim = df.groupby(["season", "manager", "division"], as_index=False).agg(
        wins=("adjusted_win", "sum"), 
        points=("selected_pts", "sum")
    )

    # Single sort operation with rank calculation
    table_sim = table_sim.sort_values(
        ["season", "division", "wins", "points"],
        ascending=[True, True, False, False]
    )
    table_sim["position"] = table_sim.groupby(["season", "division"]).cumcount() + 1

    # Vectorized playoff and bye calculations
    if league_config.selected_league == "Super Flex Keeper":
        table_sim = _calculate_super_flex_playoffs_optimized(table_sim)
    else:
        playoff_spots = 6 if league_config.league_size == 12 else 4 if league_config.league_size == 10 else 1
        table_sim["playoff"] = (table_sim["position"] <= playoff_spots).astype(int)
    
    # Vectorized bye calculation
    if league_config.selected_league == "Super Flex Keeper":
        table_sim["bye"] = (table_sim["position"] == 1).astype(int)
    else:
        table_sim["bye"] = (table_sim["position"] <= 2).astype(int)

    # Single aggregation operation
    table_sim_new = table_sim.groupby("manager", as_index=False).agg(
        playoff=("playoff", "mean"), 
        bye=("bye", "mean")
    )

    if league_config.league_size == 10:
        table_sim_new = table_sim_new.drop("bye", axis=1)

    return table_sim_new


def _calculate_super_flex_playoffs_optimized(table_sim):
    """Optimized version of Super Flex playoff calculation."""
    # Use vectorized operations instead of copying entire DataFrame
    mask_top2 = table_sim["position"] <= 2
    
    # Calculate wildcard positions more efficiently
    table_sim_wins = np.where(mask_top2, 0, table_sim["wins"])
    table_sim_points = np.where(mask_top2, 0, table_sim["points"])
    
    # Create temporary series for sorting without full DataFrame copy
    temp_df = pd.DataFrame({
        'season': table_sim['season'],
        'wins': table_sim_wins,
        'points': table_sim_points
    }).sort_values(["season", "wins", "points"], ascending=[True, False, False])
    
    table_sim["overall_position"] = temp_df.groupby("season").cumcount() + 1
    
    # Vectorized playoff calculation
    table_sim["playoff"] = ((table_sim["position"] <= 2) | 
                           (table_sim["overall_position"] <= 2)).astype(int)
    
    return table_sim


def _calculate_super_flex_playoffs(table_sim):
    """Calculate playoff positions for Super Flex Keeper league."""
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
        (table_sim["position"] <= 2) | (table_sim["overall_position"] <= 2),
        1,
        0,
    )
    return table_sim


def _calculate_standard_playoffs(table_sim, league_size):
    """Calculate playoff positions for standard leagues."""
    playoff_spots = 6 if league_size == 12 else 4 if league_size == 10 else 1
    table_sim["playoff"] = np.where(table_sim["position"] <= playoff_spots, 1, 0)
    return table_sim


def _calculate_bye_weeks(table_sim, league_config):
    """Calculate bye week eligibility."""
    if league_config.selected_league == "Super Flex Keeper":
        table_sim["bye"] = np.where(table_sim["position"] == 1, 1, 0)
    else:
        table_sim["bye"] = np.where(table_sim["position"] <= 2, 1, 0)
    return table_sim


def merge_adjusted_data(df_summary_week, df_remaining_matchups_wide):
    """Merge original data with adjusted winner selections."""
    df_summary_week_new = df_summary_week.merge(
        df_remaining_matchups_wide[["matchup_id", "week", "adjusted_winner"]],
        on=["matchup_id", "week"],
        how="left",
    )
    df_summary_week_new["adjusted_win"] = np.where(
        df_summary_week_new["adjusted_winner"].isna(),
        df_summary_week_new["selected_win"],
        np.where(
            df_summary_week_new["adjusted_winner"] == df_summary_week_new["manager"],
            1,
            0,
        ),
    )
    return df_summary_week_new


# UI rendering functions
def render_header():
    """Render the page header and instructions."""
    st.markdown(
        """
        Select the outcomes of the remaining games this season to understand how your playoff chances change with
        the outcomes of each matchup.
        """
    )


def create_clear_all_callback(df_remaining_matchups_wide):
    """Create callback function to clear all radio selections."""
    def clear_all():
        # Use actual dataframe indices instead of range(0, len)
        for idx in df_remaining_matchups_wide.index:
            if f"radio_{idx}" in st.session_state:
                st.session_state[f"radio_{idx}"] = None
    return clear_all


def render_matchup_selectors(df_remaining_matchups, df_remaining_matchups_wide):
    """Render the matchup selection interface."""
    placeholders = {}
    
    for week in df_remaining_matchups["week"].unique():
        df_remaining_matchups_wide_week = df_remaining_matchups_wide[
            df_remaining_matchups_wide["week"] == week
        ]
        st.markdown(f"##### Week {week}")
        cols = st.columns(3)

        week_idx = 0
        for idx, row in df_remaining_matchups_wide_week.iterrows():
            with cols[week_idx % 3]:
                with st.container(height=85):
                    container_cols = st.columns([3, 1])
                    with container_cols[0]:
                        winner = st.radio(
                            "Select a Winner",
                            index=None,
                            options=[row[0], row[1]],
                            key=f"radio_{idx}",
                            horizontal=False,
                            label_visibility="collapsed",
                        )
                    with container_cols[1]:
                        placeholder1 = st.empty()
                        placeholder2 = st.empty()
                        placeholders[idx] = (placeholder1, placeholder2)
            df_remaining_matchups_wide.at[idx, "adjusted_winner"] = winner
            week_idx += 1

    clear_all_callback = create_clear_all_callback(df_remaining_matchups_wide)
    st.button("Clear all", on_click=clear_all_callback)
    
    return placeholders


def get_playoff_chances_config(league_size):
    """Get configuration for playoff chances display."""
    config_columns = {
        "manager": st.column_config.TextColumn(
            "Manager", help="Username of team manager"
        ),
        "playoff_original": st.column_config.NumberColumn(
            "Current Playoff %", help="% chance of team making the playoffs"
        ),
        "playoff_adjusted": st.column_config.NumberColumn(
            "Adjusted Playoff %",
            help="% chance of team making the playoffs after adjusting wins above",
        ),
        "bye_original": st.column_config.NumberColumn(
            "Current Bye %", help="% chance of team getting a bye"
        ),
        "bye_adjusted": st.column_config.NumberColumn(
            "Adjusted Bye %",
            help="% chance of team getting a bye after adjusting wins above",
        ),
    }
    
    if league_size == 10:
        del config_columns["bye_original"]
        del config_columns["bye_adjusted"]
    
    return config_columns


def render_playoff_table(adjusted_playoff_chances, league_size, standings_count, selected_user=None):
    """Render the adjusted playoff chances table."""
    st.markdown("##### Playoff Chances")
    
    # Calculate percentage changes and add arrow columns
    df_display = adjusted_playoff_chances.copy()
    
    # Calculate playoff percentage change
    df_display['playoff_change'] = df_display['playoff_adjusted'] - df_display['playoff_original']
    df_display['playoff_change_display'] = df_display['playoff_change'].apply(
        lambda x: f"{'↑' if x > 0 else '↓' if x < 0 else '→'} {x:.1%}" if x != 0 else "→ 0.0%"
    )
    
    config_columns = {
        "manager": st.column_config.TextColumn(
            "Manager", help="Username of team manager"
        ),
        "playoff_original": st.column_config.NumberColumn(
            "Current %", help="% chance of team making the playoffs"
        ),
        "playoff_adjusted": st.column_config.NumberColumn(
            "Adjusted %",
            help="% chance of team making the playoffs after adjusting wins above",
        ),
        "playoff_change_display": st.column_config.TextColumn(
            "Change", help="Change in playoff percentage", width="small"
        ),
    }
    
    sort_columns = ["playoff_original"]
    format_columns = ["playoff_original", "playoff_adjusted"]
    
    # Select columns to display
    display_columns = ["manager", "playoff_original", "playoff_adjusted", "playoff_change_display"]

    def highlight_selected_user(row):
        """Highlight the row for the selected user."""
        if selected_user and row['manager'] == selected_user:
            return ['background-color: #e6f3ff; font-weight: bold'] * len(row)
        return [''] * len(row)
    
    def color_change_arrows(row):
        """Color the change arrows based on positive/negative change."""
        styles = [''] * len(row)
        
        # Color playoff change
        if 'playoff_change_display' in row.index:
            playoff_idx = row.index.get_loc('playoff_change_display')
            if '↑' in str(row['playoff_change_display']):
                styles[playoff_idx] = 'color: green; font-weight: bold'
            elif '↓' in str(row['playoff_change_display']):
                styles[playoff_idx] = 'color: red; font-weight: bold'
            else:
                styles[playoff_idx] = 'color: gray'
        
        return styles

    styled_df = df_display[display_columns].sort_values(
        by=sort_columns, ascending=False
    ).style.apply(highlight_selected_user, axis=1).apply(
        color_change_arrows, axis=1
    ).format("{:.1%}", subset=format_columns)

    st.dataframe(
        styled_df,
        column_config=config_columns,
        height=35 * standings_count + 38,
        hide_index=True,
        use_container_width=False,
    )


def render_bye_table(adjusted_playoff_chances, standings_count, selected_user=None):
    """Render the adjusted bye chances table."""
    st.markdown("##### Bye Chances")
    
    # Calculate percentage changes and add arrow columns
    df_display = adjusted_playoff_chances.copy()
    
    # Calculate bye percentage change
    df_display['bye_change'] = df_display['bye_adjusted'] - df_display['bye_original']
    df_display['bye_change_display'] = df_display['bye_change'].apply(
        lambda x: f"{'↑' if x > 0 else '↓' if x < 0 else '→'} {x:.1%}" if x != 0 else "→ 0.0%"
    )
    
    config_columns = {
        "manager": st.column_config.TextColumn(
            "Manager", help="Username of team manager"
        ),
        "bye_original": st.column_config.NumberColumn(
            "Current %", help="% chance of team getting a bye"
        ),
        "bye_adjusted": st.column_config.NumberColumn(
            "Adjusted %",
            help="% chance of team getting a bye after adjusting wins above",
        ),
        "bye_change_display": st.column_config.TextColumn(
            "Change", help="Change in bye percentage", width="small"
        ),
    }
    
    sort_columns = ["bye_original"]
    format_columns = ["bye_original", "bye_adjusted"]
    
    # Select columns to display
    display_columns = ["manager", "bye_original", "bye_adjusted", "bye_change_display"]

    def highlight_selected_user(row):
        """Highlight the row for the selected user."""
        if selected_user and row['manager'] == selected_user:
            return ['background-color: #e6f3ff; font-weight: bold'] * len(row)
        return [''] * len(row)
    
    def color_change_arrows(row):
        """Color the change arrows based on positive/negative change."""
        styles = [''] * len(row)
        
        # Color bye change
        if 'bye_change_display' in row.index:
            bye_idx = row.index.get_loc('bye_change_display')
            if '↑' in str(row['bye_change_display']):
                styles[bye_idx] = 'color: green; font-weight: bold'
            elif '↓' in str(row['bye_change_display']):
                styles[bye_idx] = 'color: red; font-weight: bold'
            else:
                styles[bye_idx] = 'color: gray'
        
        return styles

    styled_df = df_display[display_columns].sort_values(
        by=sort_columns, ascending=False
    ).style.apply(highlight_selected_user, axis=1).apply(
        color_change_arrows, axis=1
    ).format("{:.1%}", subset=format_columns)

    st.dataframe(
        styled_df,
        column_config=config_columns,
        height=35 * standings_count + 38,
        hide_index=True,
        use_container_width=False,
    )


def render_playoff_chances_table(adjusted_playoff_chances, league_size, standings_count, selected_user=None):
    """Render the adjusted playoff chances table - for backward compatibility, only shows playoff table."""
    
    # Render playoff table only (bye table is now in separate tab)
    render_playoff_table(adjusted_playoff_chances, league_size, standings_count, selected_user)


# Percentage calculation functions
def calculate_playoff_percentage_change_by_fixture(df, week, winner, loser, league_config):
    """Calculate playoff percentage change for a specific fixture outcome."""
    df_adjusted = df.copy()
    df_adjusted.loc[
        (df["week"] == week) & (df_adjusted["manager"].isin([winner, loser])),
        "adjusted_winner",
    ] = winner
    df_adjusted["adjusted_win"] = np.where(
        df_adjusted["adjusted_winner"].isna(),
        df_adjusted["selected_win"],
        np.where(df_adjusted["adjusted_winner"] == df_adjusted["manager"], 1, 0),
    )
    df_summary = calculate_adjusted_playoff_chance(df_adjusted, league_config)
    return df_summary


def get_percentage_change_display(percentage_change):
    """Get arrow and color for percentage change display."""
    if percentage_change < 0:
        return "&#x2193;", "red"
    elif percentage_change > 0:
        return "&#x2191;", "green"
    else:
        return "&#x2192;", "grey"


def calculate_single_matchup_percentages(args):
    """Calculate percentages for a single matchup outcome - designed for parallel execution."""
    (week, team1, team2, df_adjusted_summary_week, league_config, selected_user) = args
    
    # Calculate both outcomes for this matchup
    try:
        percentage1 = calculate_playoff_percentage_change_by_fixture(
            df_adjusted_summary_week, week, team1, team2, league_config
        )
        percentage1 = percentage1[percentage1["manager"] == selected_user]["playoff"].iloc[0]
        
        percentage2 = calculate_playoff_percentage_change_by_fixture(
            df_adjusted_summary_week, week, team2, team1, league_config
        )
        percentage2 = percentage2[percentage2["manager"] == selected_user]["playoff"].iloc[0]
        
        return (week, team1, team2), percentage1, percentage2
    except Exception as e:
        # Fallback to 0 if calculation fails
        return (week, team1, team2), 0.0, 0.0


def update_percentage_placeholders_parallel(
    placeholders, df_remaining_matchups, df_remaining_matchups_wide, 
    df_adjusted_summary_week, adjusted_playoff_chances, selected_user, league_config,
    max_workers=None
):
    """Update the percentage change placeholders in real-time using parallel processing."""
    
    if max_workers is None:
        # Optimize worker count based on system and task size
        import os
        cpu_count = os.cpu_count()
        task_count = len(df_remaining_matchups_wide)
        # Use fewer workers for smaller tasks to reduce overhead
        max_workers = min(cpu_count, max(2, task_count // 2))
    
    # Get current percentage once
    current_percentage = adjusted_playoff_chances[
        adjusted_playoff_chances["manager"] == selected_user
    ]["playoff_adjusted"].iloc[0]
    
    # Prepare all calculation tasks
    calculation_tasks = []
    matchup_index_map = {}  # Map to track which index corresponds to which matchup
    
    for week in df_remaining_matchups["week"].unique():
        df_remaining_matchups_wide_week = df_remaining_matchups_wide[
            df_remaining_matchups_wide["week"] == week
        ]
        
        for idx, row in df_remaining_matchups_wide_week.iterrows():
            # Store the mapping between calculation key and dataframe index
            calculation_key = (week, row[0], row[1])
            matchup_index_map[calculation_key] = idx
            
            # Add task for parallel execution - pass full dataset
            calculation_tasks.append((
                week, row[0], row[1], df_adjusted_summary_week, league_config, selected_user
            ))
    
    # Show progress placeholder
    if calculation_tasks:
        progress_placeholder = st.empty()
        progress_placeholder.info(f"Calculating playoff impacts for {len(calculation_tasks)} matchups...")
        
        completed_count = 0
        total_tasks = len(calculation_tasks)
        
        # Execute calculations in parallel with real-time updates
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_key = {
                executor.submit(calculate_single_matchup_percentages, task): task[:3] 
                for task in calculation_tasks
            }
            
            # Process results as they complete and update UI immediately
            for future in as_completed(future_to_key):
                try:
                    calculation_key, percentage1, percentage2 = future.result()
                    completed_count += 1
                    
                    # Update progress
                    progress_placeholder.info(
                        f"Calculating playoff impacts... {completed_count}/{total_tasks} complete"
                    )
                    
                    # Update placeholder immediately when result is available
                    if calculation_key in matchup_index_map:
                        idx = matchup_index_map[calculation_key]
                        
                        percentage_change_1 = (percentage1 - current_percentage) * 100
                        percentage_change_2 = (percentage2 - current_percentage) * 100

                        arrow1, color1 = get_percentage_change_display(percentage_change_1)
                        arrow2, color2 = get_percentage_change_display(percentage_change_2)

                        if idx in placeholders:
                            placeholder1, placeholder2 = placeholders[idx]
                            placeholder1.markdown(
                                f"<div style='text-align: right; color:{color1};'>{percentage_change_1:.1f}% {arrow1}</div>",
                                unsafe_allow_html=True,
                            )
                            placeholder2.markdown(
                                f"<div style='text-align: right; color:{color2};'>{percentage_change_2:.1f}% {arrow2}</div>",
                                unsafe_allow_html=True,
                            )
                            
                except Exception as e:
                    # Handle errors gracefully - show neutral indicators
                    calculation_key = future_to_key[future]
                    completed_count += 1
                    
                    progress_placeholder.info(
                        f"Calculating playoff impacts... {completed_count}/{total_tasks} complete"
                    )
                    
                    if calculation_key in matchup_index_map:
                        idx = matchup_index_map[calculation_key]
                        if idx in placeholders:
                            placeholder1, placeholder2 = placeholders[idx]
                            placeholder1.markdown(
                                "<div style='text-align: right; color:gray;'>0.0% →</div>",
                                unsafe_allow_html=True,
                            )
                            placeholder2.markdown(
                                "<div style='text-align: right; color:gray;'>0.0% →</div>",
                                unsafe_allow_html=True,
                            )
        
        # Clear progress indicator when done
        progress_placeholder.success("✅ All playoff impacts calculated!")
        import time
        time.sleep(1)  # Brief pause to show completion
        progress_placeholder.empty()


def update_percentage_placeholders(
    placeholders, df_remaining_matchups, df_remaining_matchups_wide, 
    df_adjusted_summary_week, adjusted_playoff_chances, selected_user, league_config
):
    """Update the percentage change placeholders in real-time."""
    # Use the parallel version for better performance
    update_percentage_placeholders_parallel(
        placeholders, df_remaining_matchups, df_remaining_matchups_wide,
        df_adjusted_summary_week, adjusted_playoff_chances, selected_user, league_config
    )


def render_adjusted_playoff_sidebar(d):
    """Replace the original sidebar metrics with adjusted playoff metrics for playoff scenarios page."""
    if ("adjusted_playoff_chances" in st.session_state and 
        d.selected_user in st.session_state.adjusted_playoff_chances):
        
        adjusted_data = st.session_state.adjusted_playoff_chances[d.selected_user]
        playoff_adjusted = adjusted_data["playoff_adjusted"]
        bye_adjusted = adjusted_data.get("bye_adjusted")
        
        # Get original playoff probability for comparison
        user_standings = d.df_standings[d.df_standings["manager"] == d.selected_user]
        if not user_standings.empty:
            playoff_original = user_standings["playoff"].iloc[0]
            playoff_change = playoff_adjusted - playoff_original
            
            # Replace the existing playoff metric container if it exists
            if hasattr(st.session_state, 'playoff_metric_container') and st.session_state.playoff_metric_container:
                with st.session_state.playoff_metric_container.container():
                    st.markdown("<style>div[data-testid='stMetric'] div { font-size: 15px; }</style>", unsafe_allow_html=True)
                    # Create columns for side-by-side display
                    if d.league_size == 12 and bye_adjusted is not None and "bye" in user_standings.columns:
                        # Two columns for playoff and bye
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric(
                                label="Playoff %",
                                value=f"{playoff_adjusted:.1%}",
                                delta=f"{playoff_change:.1%}" if playoff_change != 0 else None,
                                help="Your playoff chance with selected matchup outcomes"
                            )
                        
                        bye_original = user_standings["bye"].iloc[0]
                        bye_change = bye_adjusted - bye_original
                        
                        with col2:
                            st.metric(
                                label="Bye %",
                                value=f"{bye_adjusted:.1%}",
                                delta=f"{bye_change:.1%}" if bye_change != 0 else None,
                                help="Your bye chance with selected matchup outcomes"
                            )
                    else:
                        # Single column for playoff only (10-team leagues)
                        st.metric(
                            label="Playoff Chance",
                            value=f"{playoff_adjusted:.1%}",
                            delta=f"{playoff_change:.1%}" if playoff_change != 0 else None,
                            help="Your playoff chance with selected matchup outcomes"
                        )


def playoff_scenarios(d):
    """Main function orchestrating the playoff scenarios page."""
    # Data preparation
    df_remaining_matchups, df_remaining_matchups_wide = prepare_remaining_matchups(
        d.df_summary_week, d.gameweek_end
    )
    
    # UI rendering
    render_header()
    
    # Create tabs
    if d.league_size == 12:
        tab1, tab2, tab3 = st.tabs(["📅 Weekly Matchups", "📊 Playoff Chances", "🎯 Bye Chances"])
    else:
        tab1, tab2 = st.tabs(["📅 Weekly Matchups", "📊 Playoff Chances"])
    
    with tab1:
        st.markdown("##### Select Winners for Remaining Matchups")
        st.markdown("Use the slider to select more gameweek options, but note that loading times will increase for each added week.")
        
        # Add gameweek filter slider
        available_weeks = sorted(df_remaining_matchups["week"].unique())
        if len(available_weeks) > 1:
            selected_weeks = st.slider(
                "Select gameweeks to display:",
                min_value=min(available_weeks),
                max_value=max(available_weeks),
                value=(min(available_weeks), min(min(available_weeks)+1, max(available_weeks))),
                help="Choose which weeks to show matchups for",
                width=400
            )
            
            # Filter matchups based on selected weeks
            df_remaining_matchups_filtered = df_remaining_matchups[
                (df_remaining_matchups["week"] >= selected_weeks[0]) & 
                (df_remaining_matchups["week"] <= selected_weeks[1])
            ]
            df_remaining_matchups_wide_filtered = df_remaining_matchups_wide[
                (df_remaining_matchups_wide["week"] >= selected_weeks[0]) & 
                (df_remaining_matchups_wide["week"] <= selected_weeks[1])
            ]
        else:
            # If only one week available, don't show slider
            df_remaining_matchups_filtered = df_remaining_matchups
            df_remaining_matchups_wide_filtered = df_remaining_matchups_wide
        
        placeholders = render_matchup_selectors(df_remaining_matchups_filtered, df_remaining_matchups_wide_filtered)
        
        # Important: Transfer selections from filtered data back to original data
        # This ensures that playoff calculations in tab2 reflect user selections
        for idx, row in df_remaining_matchups_wide_filtered.iterrows():
            if idx in df_remaining_matchups_wide.index:
                df_remaining_matchups_wide.at[idx, "adjusted_winner"] = row["adjusted_winner"]
    
    # Data processing (shared between tabs) - moved here so it updates with user selections
    df_summary_week_new = merge_adjusted_data(d.df_summary_week, df_remaining_matchups_wide)
    
    # Calculate adjusted playoff chances
    adjusted_playoff_chances = calculate_adjusted_playoff_chance(df_summary_week_new, d).merge(
        d.df_standings[
            ["manager", "playoff", "bye"] if "bye" in d.df_standings 
            else ["manager", "playoff"]
        ],
        on="manager",
        how="left",
        suffixes=("_adjusted", "_original"),
    )
    
    table_cols = ["manager", "playoff_original", "playoff_adjusted"] + (
        ["bye_original", "bye_adjusted"] if d.league_size == 12 else []
    )
    adjusted_playoff_chances = adjusted_playoff_chances[table_cols]

    # Store adjusted playoff chances in session state for sidebar display
    if "adjusted_playoff_chances" not in st.session_state:
        st.session_state.adjusted_playoff_chances = {}
    
    # Store the adjusted chances for the selected user
    user_adjusted_data = adjusted_playoff_chances[
        adjusted_playoff_chances["manager"] == d.selected_user
    ]
    if not user_adjusted_data.empty:
        st.session_state.adjusted_playoff_chances[d.selected_user] = {
            "playoff_adjusted": user_adjusted_data["playoff_adjusted"].iloc[0],
            "bye_adjusted": user_adjusted_data["bye_adjusted"].iloc[0] if "bye_adjusted" in user_adjusted_data.columns else None
        }

    with tab2:
        # Render playoff table only in second tab
        render_playoff_table(adjusted_playoff_chances, d.league_size, len(d.df_standings), d.selected_user)

    # Add bye chances tab for 12-team leagues
    if d.league_size == 12:
        with tab3:
            # Render bye table in third tab
            render_bye_table(adjusted_playoff_chances, len(d.df_standings), d.selected_user)

    # Update percentage indicators (this happens in the background for tab1)
    # Note: We use the original unfiltered data for calculations to ensure all matchups are considered
    df_adjusted_summary_week = df_summary_week_new[
        ["season", "division", "manager", "week", "adjusted_winner", "selected_win", "selected_pts"]
    ]
    
    # Only update placeholders for the filtered matchups that are currently displayed
    if 'df_remaining_matchups_filtered' in locals() and 'df_remaining_matchups_wide_filtered' in locals():
        update_percentage_placeholders(
            placeholders, df_remaining_matchups_filtered, df_remaining_matchups_wide_filtered,
            df_adjusted_summary_week, adjusted_playoff_chances, d.selected_user, d
        )
    else:
        # Fallback for when no filtering is applied
        update_percentage_placeholders(
            placeholders, df_remaining_matchups, df_remaining_matchups_wide,
            df_adjusted_summary_week, adjusted_playoff_chances, d.selected_user, d
        )
    
    # Render adjusted playoff metrics in sidebar
    render_adjusted_playoff_sidebar(d)