"""
Helper functions and utilities for the ERA5 Dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
from .constants import DEFAULT_TIMEZONE_OFFSET

def convert_utc_to_local_timezone(datetime_obj, offset_hours=None):
    """
    Convert UTC datetime to local timezone.
    
    Args:
        datetime_obj: pandas datetime or numpy datetime64
        offset_hours: timezone offset in hours (if None, uses session state or default)
    
    Returns:
        pandas datetime with timezone conversion applied
    """
    if datetime_obj is None:
        return None
    
    # Get offset from session state if available, otherwise use default
    if offset_hours is None:
        try:
            import streamlit as st
            offset_hours = st.session_state.get('timezone_offset', DEFAULT_TIMEZONE_OFFSET)
        except:
            offset_hours = DEFAULT_TIMEZONE_OFFSET
    
    # Convert to pandas datetime if needed
    if isinstance(datetime_obj, (np.datetime64, np.ndarray)):
        dt = pd.to_datetime(datetime_obj)
    else:
        dt = pd.to_datetime(datetime_obj)
    
    # Add the timezone offset
    return dt + pd.Timedelta(hours=offset_hours)

def convert_time_array_to_local_timezone(time_array, offset_hours=None):
    """
    Convert an array of UTC datetimes to local timezone.
    
    Args:
        time_array: array-like of datetime objects
        offset_hours: timezone offset in hours (if None, uses session state or default)
    
    Returns:
        converted time array
    """
    if time_array is None or len(time_array) == 0:
        return time_array
    
    # Get offset from session state if available, otherwise use default
    if offset_hours is None:
        try:
            import streamlit as st
            offset_hours = st.session_state.get('timezone_offset', DEFAULT_TIMEZONE_OFFSET)
        except:
            offset_hours = DEFAULT_TIMEZONE_OFFSET
    
    # Convert each element
    return [convert_utc_to_local_timezone(t, offset_hours) for t in time_array]

def format_datetime_with_timezone(datetime_obj, offset_hours=None, format_str='%Y-%m-%d %H:%M'):
    """
    Format datetime with timezone conversion and display timezone info.
    
    Args:
        datetime_obj: datetime object
        offset_hours: timezone offset in hours (if None, uses session state or default)
        format_str: strftime format string
    
    Returns:
        formatted string with timezone info
    """
    if datetime_obj is None:
        return "N/A"
    
    # Get offset from session state if available, otherwise use default
    if offset_hours is None:
        try:
            import streamlit as st
            offset_hours = st.session_state.get('timezone_offset', DEFAULT_TIMEZONE_OFFSET)
        except:
            offset_hours = DEFAULT_TIMEZONE_OFFSET
    
    # Convert to local timezone
    local_dt = convert_utc_to_local_timezone(datetime_obj, offset_hours)
    
    # Format with timezone info
    tz_str = f"UTC{'+' if offset_hours >= 0 else ''}{offset_hours}"
    return f"{local_dt.strftime(format_str)} ({tz_str})"

def get_time_dim_name(dataset):
    """Finds the name of the time coordinate in a dataset."""
    if 'time' in dataset.coords:
        return 'time'
    if 'valid_time' in dataset.coords:
        return 'valid_time'
    # Check dims as a fallback
    if 'time' in dataset.dims:
        return 'time'
    if 'valid_time' in dataset.dims:
        return 'valid_time'
    st.error("Could not find a recognizable time coordinate ('time' or 'valid_time') in the dataset.")
    return None

def st_checkbox_grid(label, options, num_cols, key):
    """Helper function for creating a grid of checkboxes inside an expander"""
    # Key for storing selections in session state
    state_key = f"{key}_selections"

    # Initialize state if it's not there
    if state_key not in st.session_state:
        if key == 'years':
            st.session_state[state_key] = ['2023'] # Default for years
        else:
            st.session_state[state_key] = [] # Default to none for others

    with st.expander(label):
        # --- Buttons for Select/Deselect All ---
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Select All", key=f"{key}_select_all", use_container_width=True):
                st.session_state[state_key] = options
                st.rerun()
        with col2:
            if st.button("Deselect All", key=f"{key}_deselect_all", use_container_width=True):
                st.session_state[state_key] = []
                st.rerun()

        # --- Display Checkboxes in a Grid ---
        selected_options = []
        cols = st.columns(num_cols)
        for i, option in enumerate(options):
            with cols[i % num_cols]:
                is_checked = st.checkbox(str(option), value=(option in st.session_state.get(state_key, [])), key=f"{key}_{option}")
                if is_checked:
                    selected_options.append(option)
        
        st.session_state[state_key] = selected_options

    # Always return the value from session state
    return st.session_state.get(state_key, [])

def clear_prediction_results():
    """Clears prediction results and download data from the session state."""
    keys_to_delete = ['prediction_results', 'download_data']
    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]
