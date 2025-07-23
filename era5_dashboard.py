"""
ERA5 Data Dashboard - Main Entry Point

A modular Streamlit dashboard for ERA5 data exploration, visualization, and machine learning predictions.
This is the main entry point that orchestrates all the modular components.
"""

import streamlit as st
import os
from tabs.data_fetching import render_data_fetching_tab
from tabs.file_inspection import render_file_inspection_tab
from tabs.visualization import render_visualization_tab
from tabs.report import render_report_tab
from tabs.prediction import render_prediction_tab
from utils.constants import TIMEZONE_OPTIONS, DEFAULT_TIMEZONE_OFFSET, TEMPERATURE_UNITS, DEFAULT_TEMPERATURE_UNIT

# Configure Streamlit page
st.set_page_config(
    page_title="ERA5 Data Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Session State Initialization ---
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None
# Initialize area for bounding box selection
if 'area' not in st.session_state:
    st.session_state.area = {'north': 26.5, 'west': 50.5, 'south': 24.5, 'east': 52.0}
if 'latitude' not in st.session_state:
    st.session_state.latitude = 25.0
if 'longitude' not in st.session_state:
    st.session_state.longitude = 51.0
if 'selection_mode' not in st.session_state:
    st.session_state.selection_mode = "Single Point"
# Initialize timezone setting
if 'timezone_offset' not in st.session_state:
    st.session_state.timezone_offset = DEFAULT_TIMEZONE_OFFSET
# Initialize temperature unit setting
if 'temperature_unit' not in st.session_state:
    st.session_state.temperature_unit = DEFAULT_TEMPERATURE_UNIT

def main():
    """Main dashboard application."""
    
    # Dashboard header
    st.title("ERA5 Data Dashboard")
    st.markdown("**A comprehensive tool for ERA5 climate data exploration, visualization, and machine learning predictions.**")
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("### ⚙️ Global Settings")
        
        # Timezone Configuration
        timezone_options = list(TIMEZONE_OPTIONS.keys())
        current_tz = f"UTC{'+' if st.session_state.timezone_offset >= 0 else ''}{st.session_state.timezone_offset}"
        
        if current_tz in timezone_options:
            default_index = timezone_options.index(current_tz)
        else:
            default_index = timezone_options.index(f"UTC+{DEFAULT_TIMEZONE_OFFSET}")
        
        selected_timezone = st.selectbox(
            "🕒 Time Zone Display",
            options=timezone_options,
            index=default_index,
            help="Choose the timezone for displaying time data. All ERA5 data is originally in UTC.",
            key="timezone_selector"  # Use a stable key
        )
        
        # Update session state when timezone changes (reactive, no rerun)
        st.session_state.timezone_offset = TIMEZONE_OPTIONS[selected_timezone]
        
        # Temperature Unit Configuration
        temperature_options = list(TEMPERATURE_UNITS.keys())
        current_temp_unit = st.session_state.temperature_unit
        
        if current_temp_unit in temperature_options:
            default_temp_index = temperature_options.index(current_temp_unit)
        else:
            default_temp_index = temperature_options.index(DEFAULT_TEMPERATURE_UNIT)
        
        selected_temp_unit = st.selectbox(
            "🌡️ Temperature Unit",
            options=temperature_options,
            index=default_temp_index,
            help="Choose the temperature unit for displaying temperature data throughout the dashboard. Original .nc files remain unchanged.",
            key="temp_unit_selector"  # Use a stable key
        )
        
        # Update session state when temperature unit changes (reactive, no rerun)
        st.session_state.temperature_unit = selected_temp_unit
        
        st.markdown("---")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Data Fetching", 
        "File Inspection", 
        "Visualization", 
        "Report", 
        "Prediction"
    ])
    
    # Render each tab with its respective module
    with tab1:
        render_data_fetching_tab()
    
    with tab2:
        render_file_inspection_tab()
    
    with tab3:
        render_visualization_tab()
    
    with tab4:
        render_report_tab()
    
    with tab5:
        render_prediction_tab()

if __name__ == "__main__":
    main()
