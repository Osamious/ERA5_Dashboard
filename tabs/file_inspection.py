"""
File Inspection tab for the ERA5 Dashboard.
"""

import streamlit as st
import xarray as xr
import os

def render_file_inspection_tab():
    """Renders the File Inspection tab content."""
    st.header("File Inspection")
    
    st.subheader("Raw Data Inspector")
    database_path = "database"
    if not os.path.exists(database_path):
        st.info("No database folder found. Please download some data first in the 'Data Fetching' tab.")
        return
        
    nc_files_for_inspection = [f for f in os.listdir(database_path) if f.endswith('.nc')]
    if not nc_files_for_inspection:
        st.info("No NetCDF files found in the database folder to inspect.")
    else:
        selected_file_for_inspection = st.selectbox(
            "Choose a NetCDF file to inspect",
            ["None"] + nc_files_for_inspection,
            key="raw_data_inspector_selector"
        )

        if selected_file_for_inspection and selected_file_for_inspection != "None":
            file_path = os.path.join(database_path, selected_file_for_inspection)
            try:
                with xr.open_dataset(file_path) as ds:
                    with st.expander(f"Inspect: `{selected_file_for_inspection}`", expanded=True):
                        st.markdown("#### Full Dataset Structure (via xarray)")
                        st.code(str(ds))

                        st.markdown("#### Data Values Table")
                        # Convert the entire dataset to a pandas DataFrame for display
                        df = ds.to_dataframe().reset_index()
                        
                        # Use st.dataframe for better scrolling with wide tables
                        st.dataframe(df)

            except Exception as e:
                st.error(f"Could not read or display data from {selected_file_for_inspection}: {e}")

    st.markdown("---")
    st.subheader("File Content Checker")

    checker_files = [f for f in os.listdir(database_path) if f.endswith('.nc')]
    if not checker_files:
        st.info("No NetCDF files found in the database folder to check.")
    else:
        selected_file_to_check = st.selectbox(            "Choose a NetCDF file to check its type",
            ["None"] + checker_files,
            key="file_checker_selector"
        )
        
        if selected_file_to_check and selected_file_to_check != "None":
            file_path = os.path.join(database_path, selected_file_to_check)
            try:
                with xr.open_dataset(file_path) as ds:
                    # Robust check for gridded data
                    is_area = 'latitude' in ds.dims and 'longitude' in ds.dims and len(ds.latitude) > 1 and len(ds.longitude) > 1
                    
                    if is_area:
                        st.success(f"**File:** `{selected_file_to_check}` is an **Area file**.")
                        st.markdown("It contains gridded data with multiple latitude and longitude points, suitable for spatial plots.")
                    else:
                        st.warning(f"**File:** `{selected_file_to_check}` is a **Single Point file**.")
                        st.markdown("It contains data for a single coordinate and cannot be used for spatial plots.")

                    # Check for number of variables
                    st.markdown("---")
                    num_vars = len(ds.data_vars)
                    var_names = list(ds.data_vars.keys())
                    if num_vars > 1:
                        st.info(f"**Variable Content:** This file contains **{num_vars} variables**: `{', '.join(var_names)}`")
                    elif num_vars == 1:
                        st.info(f"**Variable Content:** This file contains a **single variable**: `{var_names[0]}`")
                    else:
                        st.warning("**Variable Content:** This file contains no data variables.")
                    
                    # Optionally show the dimensions for clarity
                    st.code(f"Dimensions: {ds.dims}")

            except Exception as e:
                st.error(f"Could not read or check the file {selected_file_to_check}: {e}")
