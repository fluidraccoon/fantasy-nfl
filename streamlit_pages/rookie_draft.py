import streamlit as st
import pandas as pd
import numpy as np
import os


def show_rookie_draft_page(d):
    """Show the Rookie Draft page."""
    st.title(f"{d.season + 1} Rookie Draft")
    st.markdown(f"""
    Welcome to the {d.season + 1} Rookie Draft page! This section will help you prepare for the upcoming rookie draft
    by providing insights into draft order, prospect rankings, and team needs analysis.
    """)
    
    # Create tabs for different draft-related content
    tab1, tab2 = st.tabs(["📋 Draft Order", "🎯 Pick Likelihood"])
    
    with tab1:
        render_draft_order(d)
    
    with tab2:
        render_pick_likelihood(d)


def load_draft_picks_data(league_name):
    """Load draft picks data from CSV files."""
    try:
        # Load from the combined draft_picks.csv file
        combined_file = "data/draft_picks.csv"
        if os.path.exists(combined_file):
            df = pd.read_csv(combined_file)
            # Filter by league_name (first column)
            if 'league_name' in df.columns:
                return df[df['league_name'] == league_name]
            else:
                st.warning("No league_name column found in draft_picks.csv")
                return pd.DataFrame()
        else:
            st.info("No draft_picks.csv file found. Run the data creation script to generate traded picks data.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading draft picks data: {e}")
        return pd.DataFrame()


def calculate_draft_order_with_traded_picks(standings_df, traded_picks_df, d=None):
    """Calculate draft order based on potential points with traded picks applied."""
    # Use potential points if available, otherwise regular points
    sort_column = 'pp_points' if 'pp_points' in standings_df.columns else 'points'
    
    # Sort by potential points (ascending - worst team gets pick 1)
    draft_order = standings_df.copy()
    draft_order = draft_order.sort_values(sort_column, ascending=True).reset_index(drop=True)
    
    # Get roster_id mapping from matchup schedule if available
    roster_mapping = None
    if d and hasattr(d, 'df_matchup_schedule') and not d.df_matchup_schedule.empty:
        # Create a mapping of roster_id to manager from matchup schedule
        roster_mapping = d.df_matchup_schedule[['roster_id', 'manager']].drop_duplicates()
    
    # Create draft picks DataFrame for multiple rounds
    all_picks = []
    num_teams = len(draft_order)
    
    for round_num in range(1, 5):  # Show first 4 rounds
        round_picks = draft_order.copy()
        round_picks['round'] = round_num
        round_picks['pick_number'] = range(1, num_teams + 1)
        round_picks['overall_pick'] = (round_num - 1) * num_teams + round_picks['pick_number']
        round_picks['owner'] = round_picks['manager']  # Current owner (default to original)
        round_picks['original_owner'] = ""  # Will be filled only for traded picks
        round_picks['traded'] = False
        
        # Apply traded picks for this round
        if not traded_picks_df.empty and roster_mapping is not None:
            # Use next season for draft picks
            next_season = d.season + 1 if d and hasattr(d, 'season') else 2026
            traded_this_round = traded_picks_df[
                (traded_picks_df['season'] == next_season) & 
                (traded_picks_df['round'] == round_num)
            ]
            
            for _, trade in traded_this_round.iterrows():
                # The roster_id in the trade tells us which team slot the pick originally belonged to
                original_roster_id = trade['roster_id']
                
                # The owner_id tells us which roster currently owns the pick
                current_owner_roster_id = trade['owner_id']
                
                # The previous_owner_id tells us which roster originally owned the pick
                previous_owner_roster_id = trade['previous_owner_id']
                
                # Get manager names from roster mapping
                original_manager_match = roster_mapping[roster_mapping['roster_id'] == original_roster_id]
                current_owner_match = roster_mapping[roster_mapping['roster_id'] == current_owner_roster_id]
                
                if not original_manager_match.empty and not current_owner_match.empty:
                    original_manager = original_manager_match['manager'].iloc[0]
                    current_owner_name = current_owner_match['manager'].iloc[0]
                    
                    # Find the pick in our draft order by matching the original manager
                    manager_mask = round_picks['manager'] == original_manager
                    if manager_mask.any():
                        # Update the pick ownership - show current owner and original owner
                        round_picks.loc[manager_mask, 'owner'] = current_owner_name
                        round_picks.loc[manager_mask, 'original_owner'] = original_manager
                        round_picks.loc[manager_mask, 'traded'] = True
        
        all_picks.append(round_picks)
    
    return pd.concat(all_picks, ignore_index=True)


def render_draft_order(d):
    """Render the draft order section based on potential points and traded picks."""
    next_season = d.season + 1 if hasattr(d, 'season') else 2026
    st.markdown(f"##### {next_season} Rookie Draft Order")
    
    st.info(f"""
    📝 **Note**: Draft order is determined by potential points (lowest gets first pick).
    Traded picks are reflected in the ownership column for the {next_season} draft.
    """)
    
    if hasattr(d, 'df_standings') and not d.df_standings.empty:
        # Load traded picks data
        traded_picks_df = load_draft_picks_data(d.selected_league)
        
        # Calculate draft order with traded picks
        draft_picks_df = calculate_draft_order_with_traded_picks(d.df_standings, traded_picks_df, d)
        
        # Get available rounds
        rounds = sorted(draft_picks_df['round'].unique())
        
        # Create tabs for each round
        if len(rounds) == 4:
            tab_round1, tab_round2, tab_round3, tab_round4 = st.tabs([
                "🥇 Round 1", "🥈 Round 2", "🥉 Round 3", "4️⃣ Round 4"
            ])
            tabs = [tab_round1, tab_round2, tab_round3, tab_round4]
        else:
            # Fallback for different number of rounds
            tab_labels = [f"Round {r}" for r in rounds]
            tabs = st.tabs(tab_labels)
        
        for i, round_num in enumerate(rounds):
            with tabs[i]:
                round_data = draft_picks_df[draft_picks_df['round'] == round_num]
                
                # Prepare display columns - only pick, owner, and original owner
                display_cols = ['pick_number', 'owner', 'original_owner']
                
                # Configure columns
                column_config = {
                    "pick_number": st.column_config.NumberColumn("Pick", help="Draft pick number in round"),
                    "owner": st.column_config.TextColumn("Owner", help="Team that currently owns this pick"),
                    "original_owner": st.column_config.TextColumn("Original Owner", help="Team that originally owned this pick (blank if not traded)")
                }
                
                # Style the dataframe to highlight traded picks
                def highlight_traded_picks(row):
                    if row.get('original_owner', '') != '':  # Highlight if original_owner is not empty
                        return ['background-color: #fff2cc'] * len(row)  # Light yellow for traded picks
                    return [''] * len(row)
                
                styled_df = round_data[display_cols].style.apply(highlight_traded_picks, axis=1)
                
                st.dataframe(
                    styled_df,
                    column_config=column_config,
                    hide_index=True,
                    use_container_width=False,
                    height=457  # Show all 12 picks without scrolling
                )
    else:
        st.warning("Draft order will be available once season data is loaded.")


def render_pick_likelihood(d):
    """Render the pick likelihood section."""
    st.markdown("##### Pick Likelihood Analysis")
    
    st.info(f"""
    🎯 **Draft Position Probabilities**: Based on {getattr(d, 'sims', 1000)} season simulations, 
    this shows the likelihood of each manager finishing in each draft position.
    """)
    
    if hasattr(d, 'df_standings') and not d.df_standings.empty:
        # Run simulations to get draft position probabilities
        from simulate_table import simulate_table
        
        # Get simulation results
        sim_results = simulate_table(d)
        
        if not sim_results.empty and 'draft_pos' in sim_results.columns:
            # Calculate percentages for each manager and draft position
            draft_position_probs = (
                sim_results.groupby(['manager', 'draft_pos'])
                .size()
                .unstack(fill_value=0)
            )
            
            # Convert to percentages (1 decimal place)
            total_sims = getattr(d, 'sims', len(sim_results.groupby('season')))
            draft_position_probs = (draft_position_probs / total_sims * 100).round(1)
            
            # Remove index header and capitalize manager names
            draft_position_probs.index.name = None
            
            # Group picks 7+ into "Playoffs" column
            display_probs = draft_position_probs.copy()
            
            # Keep individual columns for picks 1-6
            picks_to_keep = [col for col in display_probs.columns if col <= 6]
            picks_to_group = [col for col in display_probs.columns if col > 6]
            
            # Create the "Playoffs" column by summing picks 7+
            if picks_to_group:
                display_probs['Playoffs'] = display_probs[picks_to_group].sum(axis=1)
                # Drop the individual columns for picks 7+
                display_probs = display_probs.drop(columns=picks_to_group)
            
            # Sort by playoff probability (low to high - worst teams with best draft picks first)
            if 'Playoffs' in display_probs.columns:
                display_probs = display_probs.sort_values('Playoffs', ascending=True)
            
            # Reorder columns: 1, 2, 3, 4, 5, 6, Playoffs
            column_order = picks_to_keep + (['Playoffs'] if picks_to_group else [])
            display_probs = display_probs.reindex(columns=column_order)
            
            # Create multi-level column headers to group 1-6 under "Draft Pick"
            if picks_to_keep and picks_to_group:
                # Create new column names with grouping
                new_columns = []
                for col in display_probs.columns:
                    if col in picks_to_keep:
                        new_columns.append(('Draft Pick', f'{col}'))
                    else:  # Playoffs column
                        new_columns.append(('', 'Playoffs'))
                
                # Apply multi-level columns
                display_probs.columns = pd.MultiIndex.from_tuples(new_columns)
            
            # Format values to show % with 1 decimal place
            display_probs_formatted = display_probs.copy()
            for col in display_probs_formatted.columns:
                display_probs_formatted[col] = display_probs_formatted[col].apply(lambda x: f"{x:.1f}%")
            
            # Style the dataframe to highlight probabilities with red color scale
            def highlight_probability_scale(df):
                # Convert all percentage strings to numbers for global min/max calculation
                numeric_df = df.copy()
                for col in numeric_df.columns:
                    numeric_df[col] = numeric_df[col].apply(lambda x: float(x.rstrip('%')))
                
                # Get global min/max from columns 1-6 (draft pick columns) across ALL rows
                draft_pick_cols = [col for col in numeric_df.columns if 'Playoffs' not in str(col)]
                
                if draft_pick_cols:
                    # Get global min/max across all cells in draft pick columns
                    global_min = numeric_df[draft_pick_cols].min().min()
                    global_max = numeric_df[draft_pick_cols].max().max()
                else:
                    global_min = global_max = 0
                
                # Create a DataFrame to store the styling
                style_df = pd.DataFrame('', index=df.index, columns=df.columns)
                
                # Apply colors cell by cell using global scale
                for idx in df.index:
                    for col in df.columns:
                        v = df.loc[idx, col]
                        prob = float(v.rstrip('%'))
                        
                        # Keep playoffs column white (no color)
                        if 'Playoffs' in str(col):
                            style_df.loc[idx, col] = ''
                        elif prob == 0:
                            style_df.loc[idx, col] = ''  # No color for 0%
                        else:
                            # Scale from light red (low probability) to dark red (high probability)
                            # Normalize using GLOBAL min/max values from draft pick columns only
                            if global_max > global_min:
                                intensity = (prob - global_min) / (global_max - global_min)
                            else:
                                intensity = 1.0 if prob > 0 else 0.0
                            
                            # Apply a power function to make low values more distinct
                            # This spreads out the lower values more dramatically
                            intensity = intensity ** 0.5  # Square root makes lower values more visible
                            
                            # Create red color with more dramatic intensity range
                            # Very light red: rgb(255, 245, 245) to Dark red: rgb(200, 50, 50)
                            red_value = int(255 - (55 * intensity))  # 255 to 200
                            green_blue_value = int(245 - (195 * intensity))  # 245 to 50
                            style_df.loc[idx, col] = f'background-color: rgb({red_value}, {green_blue_value}, {green_blue_value})'
                
                return style_df
            
            # Apply styling using the new function
            styled_df = display_probs_formatted.style.apply(highlight_probability_scale, axis=None)
            
            st.dataframe(
                styled_df,
                use_container_width=False,
                height=492
            )
        else:
            st.warning("Simulation data not available. Please ensure simulations have been run.")
    else:
        st.warning("Draft position likelihood will be available once season data is loaded.")

