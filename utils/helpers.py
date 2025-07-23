"""
Helper functions and utilities for the ERA5 Dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
from .constants import DEFAULT_TIMEZONE_OFFSET, convert_temperature, get_temperature_unit_symbol, is_temperature_variable, DEFAULT_TEMPERATURE_UNIT

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

def apply_temperature_conversion(data, variable_short_name, target_unit=None):
    """
    Apply temperature conversion to data if it's a temperature variable.
    
    Args:
        data: The temperature data (numpy array, pandas series, xarray DataArray, or scalar)
        variable_short_name: Short name of the variable (e.g., 't2m', 'sst', 'skt')
        target_unit: Target temperature unit (if None, uses session state)
    
    Returns:
        Converted data if temperature variable, otherwise original data
    """
    # Check if this is a temperature variable
    if not is_temperature_variable(variable_short_name):
        return data
    
    # Get target unit from session state if not provided
    if target_unit is None:
        try:
            target_unit = st.session_state.get('temperature_unit', DEFAULT_TEMPERATURE_UNIT)
        except:
            target_unit = DEFAULT_TEMPERATURE_UNIT
    
    # ERA5 temperature data is originally in Kelvin
    source_unit = "Kelvin"
    
    # Convert the data
    return convert_temperature(data, source_unit, target_unit)

def get_temperature_display_unit(variable_short_name=None):
    """
    Get the display unit for temperature variables based on current session state.
    
    Args:
        variable_short_name: Short name of the variable (optional, for validation)
    
    Returns:
        Display unit string (e.g., '°C', '°F', 'K')
    """
    if variable_short_name and not is_temperature_variable(variable_short_name):
        return None  # Not a temperature variable
    
    try:
        current_unit = st.session_state.get('temperature_unit', DEFAULT_TEMPERATURE_UNIT)
    except:
        current_unit = DEFAULT_TEMPERATURE_UNIT
    
    return get_temperature_unit_symbol(current_unit)

def format_temperature_label(base_label, variable_short_name):
    """
    Format a label for temperature variables with the appropriate unit.
    
    Args:
        base_label: Base label (e.g., "Temperature", "2m Temperature")
        variable_short_name: Short name of the variable
    
    Returns:
        Formatted label with unit if temperature variable, otherwise original label
    """
    if not is_temperature_variable(variable_short_name):
        return base_label
    
    unit_symbol = get_temperature_display_unit(variable_short_name)
    if unit_symbol:
        return f"{base_label} ({unit_symbol})"
    
    return base_label

def update_dataset_temperature_units(dataset, variable_short_names=None):
    """
    Update temperature units in an xarray dataset based on current session state.
    
    Args:
        dataset: xarray Dataset
        variable_short_names: List of variable short names to check, if None checks all
    
    Returns:
        Dataset with converted temperature values (original dataset is not modified)
    """
    # Create a copy to avoid modifying the original
    ds_copy = dataset.copy()
    
    # Get variables to check
    if variable_short_names is None:
        variable_short_names = list(ds_copy.data_vars.keys())
    
    # Convert temperature variables
    for var_name in variable_short_names:
        if var_name in ds_copy.data_vars and is_temperature_variable(var_name):
            ds_copy[var_name] = apply_temperature_conversion(ds_copy[var_name], var_name)
            
            # Update units attribute if it exists
            unit_symbol = get_temperature_display_unit(var_name)
            if unit_symbol and 'units' in ds_copy[var_name].attrs:
                ds_copy[var_name].attrs['units'] = unit_symbol
    
    return ds_copy


def categorize_nc_files(file_list, database_path="database"):
    """
    Categorize NetCDF files into Point/Spatial and Actual/Prediction categories.
    
    Args:
        file_list: List of NetCDF filenames
        database_path: Path to the database folder
    
    Returns:
        dict: Categorized files in the structure:
        {
            'Point': {
                'Actual': [list of point actual files],
                'Prediction': [list of point prediction files]
            },
            'Spatial': {
                'Actual': [list of spatial actual files],
                'Prediction': [list of spatial prediction files]
            }
        }
    """
    import os
    import xarray as xr
    
    categorized = {
        'Point': {'Actual': [], 'Prediction': []},
        'Spatial': {'Actual': [], 'Prediction': []}
    }
    
    for file_name in file_list:
        file_path = os.path.join(database_path, file_name)
        
        try:
            # Determine if file is prediction or actual based on filename
            is_prediction = (file_name.startswith('prediction_') or 
                           file_name.startswith('spatial_prediction_'))
            
            # Determine if file is point or spatial by checking dimensions
            with xr.open_dataset(file_path) as ds:
                is_spatial = ('latitude' in ds.dims and 'longitude' in ds.dims and 
                            len(ds.latitude) > 1 and len(ds.longitude) > 1)
            
            # Categorize the file
            file_type = 'Spatial' if is_spatial else 'Point'
            data_type = 'Prediction' if is_prediction else 'Actual'
            
            categorized[file_type][data_type].append(file_name)
            
        except Exception as e:
            # If we can't read the file, try to categorize by filename patterns
            is_prediction = (file_name.startswith('prediction_') or 
                           file_name.startswith('spatial_prediction_'))
            
            # Guess spatial vs point from filename patterns
            is_spatial = ('area_' in file_name or 'spatial_' in file_name or 
                         '_area_' in file_name or file_name.startswith('spatial_'))
            
            file_type = 'Spatial' if is_spatial else 'Point'
            data_type = 'Prediction' if is_prediction else 'Actual'
            
            categorized[file_type][data_type].append(file_name)
    
    return categorized
