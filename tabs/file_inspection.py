"""
File Inspection tab for th            try:
                with xr.open_dataset(file_path) as ds:
                    # Apply temperature conversion to the dataset for display
                    ds_display = update_dataset_temperature_units(ds)
                    
                    with st.expander(f"Inspect: `{selected_file_for_inspection}`", expanded=True):
                        # Show temperature unit info if applicable
                        temp_vars = [var for var in ds.data_vars if any(temp_name in var for temp_name in ['t2m', 'sst', 'skt', 'stl1'])]
                        if temp_vars:
                            current_temp_unit = st.session_state.get('temperature_unit', 'Celsius')
                            st.info(f"🌡️ Temperature data is displayed in **{current_temp_unit}**. Original files remain in Kelvin.")
                        
                        st.markdown("#### Full Dataset Structure (via xarray)")
                        st.code(str(ds_display))

                        st.markdown("#### Data Values Table")
                        # Convert the entire dataset to a pandas DataFrame for display
                        df = ds_display.to_dataframe().reset_index()
                        
                        # Use st.dataframe for better scrolling with wide tables
                        st.dataframe(df)oard.
"""

import streamlit as st
import xarray as xr
import os
from utils.helpers import apply_temperature_conversion, get_temperature_display_unit, update_dataset_temperature_units, categorize_nc_files

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
        # Categorize files for better organization
        categorized_files = categorize_nc_files(nc_files_for_inspection, database_path)
        
        # Create hierarchical categorized options
        file_options = ["None"]
        
        # Add Point Files section
        if categorized_files['Point']['Actual'] or categorized_files['Point']['Prediction']:
            file_options.append("📍 POINT FILES")
            file_options.append("─" * 50)  # Separator line
            
            if categorized_files['Point']['Actual']:
                file_options.append("    ✓ Actual:")
                for file in sorted(categorized_files['Point']['Actual']):
                    file_options.append(f"      └─ {file}")
            
            if categorized_files['Point']['Prediction']:
                file_options.append("    🔮 Predicted:")
                for file in sorted(categorized_files['Point']['Prediction']):
                    file_options.append(f"      └─ {file}")
            
            # Add separator if spatial files exist
            if categorized_files['Spatial']['Actual'] or categorized_files['Spatial']['Prediction']:
                file_options.append("─" * 50)  # Separator line
        
        # Add Spatial Files section
        if categorized_files['Spatial']['Actual'] or categorized_files['Spatial']['Prediction']:
            file_options.append("🗺️ SPATIAL FILES")
            file_options.append("─" * 50)  # Separator line
            
            if categorized_files['Spatial']['Actual']:
                file_options.append("    ✓ Actual:")
                for file in sorted(categorized_files['Spatial']['Actual']):
                    file_options.append(f"      └─ {file}")
            
            if categorized_files['Spatial']['Prediction']:
                file_options.append("    🔮 Predicted:")
                for file in sorted(categorized_files['Spatial']['Prediction']):
                    file_options.append(f"      └─ {file}")
        
        selected_file_for_inspection = st.selectbox(
            "Choose a NetCDF file to inspect",
            file_options,
            key="raw_data_inspector_selector"
        )

        # Extract actual filename if a categorized option was selected
        actual_filename = None
        if selected_file_for_inspection and selected_file_for_inspection != "None":
            # Skip separator lines, headers, and subcategory labels
            if (selected_file_for_inspection.startswith("─") or 
                selected_file_for_inspection.startswith("📍 POINT FILES") or 
                selected_file_for_inspection.startswith("🗺️ SPATIAL FILES") or
                selected_file_for_inspection.strip().endswith("Actual:") or
                selected_file_for_inspection.strip().endswith("Predicted:")):
                st.info("Please select a file from the dropdown menu.")
            elif selected_file_for_inspection.startswith("      └─ "):
                actual_filename = selected_file_for_inspection.replace("      └─ ", "")
            else:
                actual_filename = selected_file_for_inspection

        if actual_filename:
            file_path = os.path.join(database_path, actual_filename)
            try:
                with xr.open_dataset(file_path) as ds:
                    with st.expander(f"Inspect: `{actual_filename}`", expanded=True):
                        st.markdown("#### Full Dataset Structure (via xarray)")
                        st.code(str(ds))

                        st.markdown("#### Data Values Table")
                        # Convert the entire dataset to a pandas DataFrame for display
                        df = ds.to_dataframe().reset_index()
                        
                        # Use st.dataframe for better scrolling with wide tables
                        st.dataframe(df)

            except Exception as e:
                st.error(f"Could not read or display data from {actual_filename}: {e}")

    st.markdown("---")
    st.subheader("File Content Checker")

    checker_files = [f for f in os.listdir(database_path) if f.endswith('.nc')]
    if not checker_files:
        st.info("No NetCDF files found in the database folder to check.")
    else:
        # Categorize files for the checker as well
        categorized_checker_files = categorize_nc_files(checker_files, database_path)
        
        # Create hierarchical categorized options for checker
        checker_options = ["None"]
        
        # Add Point Files section
        if categorized_checker_files['Point']['Actual'] or categorized_checker_files['Point']['Prediction']:
            checker_options.append("📍 POINT FILES")
            checker_options.append("─" * 50)  # Separator line
            
            if categorized_checker_files['Point']['Actual']:
                checker_options.append("    ✓ Actual:")
                for file in sorted(categorized_checker_files['Point']['Actual']):
                    checker_options.append(f"      └─ {file}")
            
            if categorized_checker_files['Point']['Prediction']:
                checker_options.append("    🔮 Predicted:")
                for file in sorted(categorized_checker_files['Point']['Prediction']):
                    checker_options.append(f"      └─ {file}")
            
            # Add separator if spatial files exist
            if categorized_checker_files['Spatial']['Actual'] or categorized_checker_files['Spatial']['Prediction']:
                checker_options.append("─" * 50)  # Separator line
        
        # Add Spatial Files section
        if categorized_checker_files['Spatial']['Actual'] or categorized_checker_files['Spatial']['Prediction']:
            checker_options.append("🗺️ SPATIAL FILES")
            checker_options.append("─" * 50)  # Separator line
            
            if categorized_checker_files['Spatial']['Actual']:
                checker_options.append("    ✓ Actual:")
                for file in sorted(categorized_checker_files['Spatial']['Actual']):
                    checker_options.append(f"      └─ {file}")
            
            if categorized_checker_files['Spatial']['Prediction']:
                checker_options.append("    🔮 Predicted:")
                for file in sorted(categorized_checker_files['Spatial']['Prediction']):
                    checker_options.append(f"      └─ {file}")
        
        selected_file_to_check = st.selectbox(
            "Choose a NetCDF file to check its type",
            checker_options,
            key="file_checker_selector"
        )
        
        # Extract actual filename for checker
        actual_checker_filename = None
        if selected_file_to_check and selected_file_to_check != "None":
            # Skip separator lines, headers, and subcategory labels
            if (selected_file_to_check.startswith("─") or 
                selected_file_to_check.startswith("📍 POINT FILES") or 
                selected_file_to_check.startswith("🗺️ SPATIAL FILES") or
                selected_file_to_check.strip().endswith("Actual:") or
                selected_file_to_check.strip().endswith("Predicted:")):
                st.info("Please select a file from the dropdown menu.")
            elif selected_file_to_check.startswith("      └─ "):
                actual_checker_filename = selected_file_to_check.replace("      └─ ", "")
            else:
                actual_checker_filename = selected_file_to_check
        
        if actual_checker_filename:
            file_path = os.path.join(database_path, actual_checker_filename)
            try:
                with xr.open_dataset(file_path) as ds:
                    # Robust check for gridded data
                    is_area = 'latitude' in ds.dims and 'longitude' in ds.dims and len(ds.latitude) > 1 and len(ds.longitude) > 1
                    
                    if is_area:
                        st.success(f"**File:** `{actual_checker_filename}` is an **Area file**.")
                        st.markdown("It contains gridded data with multiple latitude and longitude points, suitable for spatial plots.")
                    else:
                        st.warning(f"**File:** `{actual_checker_filename}` is a **Single Point file**.")
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
                st.error(f"Could not read or check the file {actual_checker_filename}: {e}")
