"""
Report tab for the ERA5 Dashboard.
"""

import streamlit as st
import xarray as xr
import numpy as np
import pandas as pd
import os

from utils.helpers import get_time_dim_name, format_datetime_with_timezone
from utils.constants import SHORT_NAME_MAP

def render_report_tab():
    """Renders the Report tab content."""
    st.header("Weather Report Generator")
    st.info("Select any available data file (point or area) to generate a detailed statistical and analytical report.")

    # Let user select from ANY available .nc file in database
    database_path = "database"
    if not os.path.exists(database_path):
        st.warning("No database folder found. Please download some data first in the 'Data Fetching' tab.")
        return
        
    all_nc_files_list = sorted([f for f in os.listdir(database_path) if f.endswith('.nc')])

    if not all_nc_files_list:
        st.warning("No NetCDF files found in the database folder. Please download a data file from the 'Data Fetching' tab.")
        return

    all_nc_files_dict = { f"File: {os.path.basename(f)}": f for f in all_nc_files_list }
    options = ["Select a file"] + list(all_nc_files_dict.keys())

    selected_report_key = st.selectbox(
        "Select a data file for the report",
        options=options,
        index=0,
        key="report_file_selector"
    )

    if selected_report_key and selected_report_key != "Select a file":
        file_name = all_nc_files_dict[selected_report_key]
        file_path = os.path.join(database_path, file_name)
        try:
            with st.spinner(f"Analyzing {os.path.basename(file_path)} and generating report..."):
                ds = xr.open_dataset(file_path)
                time_dim = get_time_dim_name(ds)
                if not time_dim:
                    st.error("Could not generate report: A recognizable time coordinate ('time' or 'valid_time') was not found.")
                    return                # --- Report Header ---
                st.markdown(f"### Analysis Report for `{os.path.basename(file_path)}`")
                import pandas as pd
                time_start = format_datetime_with_timezone(ds[time_dim].values[0])
                time_end = format_datetime_with_timezone(ds[time_dim].values[-1])
                st.markdown(f"**Time Range:** `{time_start}` to `{time_end}`")
                
                # --- Determine File Type (Point vs. Area) ---
                is_area = 'latitude' in ds.dims and 'longitude' in ds.dims and len(ds.latitude) > 1 and len(ds.longitude) > 1

                if is_area:
                    st.success("**File Type: Area Data**")
                    lat_range = f"{ds.latitude.min().item():.2f}° to {ds.latitude.max().item():.2f}°"
                    lon_range = f"{ds.longitude.min().item():.2f}° to {ds.longitude.max().item():.2f}°"
                    st.markdown(f"**Latitude Range:** `{lat_range}`")
                    st.markdown(f"**Longitude Range:** `{lon_range}`")
                else:
                    st.success("**File Type: Single Point Data**")
                    lat = ds.latitude.item()
                    lon = ds.longitude.item()
                    st.markdown(f"**Location:** `{lat:.4f}°` Latitude, `{lon:.4f}°` Longitude")
                
                st.markdown("---")

                # --- Variable Processing ---
                vars_in_file = list(ds.data_vars.keys())
                processed_vars = {} # To store calculated data like wind speed

                # Handle Wind Components first
                if 'u10' in vars_in_file and 'v10' in vars_in_file:
                    st.markdown("#### Combined Wind Analysis")
                    wind_speed = np.sqrt(ds['u10']**2 + ds['v10']**2)
                    wind_speed.attrs['long_name'] = 'Wind Speed'
                    wind_speed.attrs['units'] = 'm/s'
                    processed_vars['wind_speed'] = wind_speed
                    
                    # Remove original components from list
                    vars_in_file.remove('u10')
                    vars_in_file.remove('v10')

                # --- Generate Report based on File Type ---
                if not is_area:
                    # ==================================
                    # --- SINGLE POINT REPORT LOGIC ---
                    # ==================================
                    st.header("Statistical Summary for Single Point")
                      # Add processed vars to the list to be analyzed
                    vars_to_analyze = vars_in_file + list(processed_vars.keys())
                    ds_report = ds.copy() # Create a copy to add processed vars
                    for var_name, var_data in processed_vars.items():
                        ds_report[var_name] = var_data

                    for var_name in vars_to_analyze:
                        var_data = ds_report[var_name].squeeze()
                        
                        # Use SHORT_NAME_MAP for consistent naming
                        var_info = SHORT_NAME_MAP.get(var_name, {})
                        long_name = var_info.get('name', var_data.attrs.get('long_name', var_name).title())
                        units = var_info.get('units', var_data.attrs.get('units', ''))

                        st.markdown(f"#### {long_name}")

                        # --- Key Statistics ---
                        mean_val = var_data.mean().item()
                        median_val = var_data.median().item()
                        std_val = var_data.std().item()
                        from scipy import stats
                        mode_res = stats.mode(var_data.values, nan_policy='omit')
                        # Ensure mode_res.mode is not an empty array
                        mode_val = np.atleast_1d(mode_res.mode)
                        mode_val_str = f"{mode_val[0]:.2f} {units}" if mode_val.size > 0 else "N/A"

                        min_val = var_data.min().item()
                        max_val = var_data.max().item()
                        time_of_min = var_data.idxmin(dim=time_dim).item()
                        time_of_max = var_data.idxmax(dim=time_dim).item()

                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric(label="Mean", value=f"{mean_val:.2f} {units}")
                        col2.metric(label="Median", value=f"{median_val:.2f} {units}")
                        col3.metric(label="Mode", value=mode_val_str)
                        col4.metric(label="Std. Dev.", value=f"{std_val:.2f} {units}")

                        st.markdown(f"**Minimum:** `{min_val:.2f} {units}` (at {pd.to_datetime(time_of_min).strftime('%Y-%m-%d %H:%M')})")
                        st.markdown(f"**Maximum:** `{max_val:.2f} {units}` (at {pd.to_datetime(time_of_max).strftime('%Y-%m-%d %H:%M')})")

                        # --- Histogram ---
                        with st.expander("Show Value Distribution (Histogram)"):
                            import matplotlib.pyplot as plt
                            fig_hist, ax_hist = plt.subplots()
                            ax_hist.hist(var_data.values, bins=20, density=True, alpha=0.7, label='Frequency')
                            ax_hist.set_title(f'Distribution of {long_name}')
                            ax_hist.set_xlabel(f'Value ({units})')
                            ax_hist.set_ylabel('Frequency')
                            ax_hist.grid(True)
                            st.pyplot(fig_hist)
                        
                        st.markdown("---")

                else:
                    # =============================
                    # --- AREA DATA REPORT LOGIC ---
                    # =============================
                    st.header("Spatial Analysis for Area Data")

                    vars_to_analyze = vars_in_file + list(processed_vars.keys())
                    ds_report = ds.copy()
                    for var_name, var_data in processed_vars.items():
                        ds_report[var_name] = var_data

                    for var_name in vars_to_analyze:
                        var_data_area = ds_report[var_name]
                        
                        # Use SHORT_NAME_MAP for consistent naming
                        var_info = SHORT_NAME_MAP.get(var_name, {})
                        long_name = var_info.get('name', var_data_area.attrs.get('long_name', var_name).title())
                        units = var_info.get('units', var_data_area.attrs.get('units', ''))

                        st.markdown(f"#### {long_name}")

                        # --- Spatial Mean Map ---
                        with st.expander("Show Map of Mean Values", expanded=True):
                            import matplotlib.pyplot as plt
                            import cartopy.crs as ccrs
                            
                            mean_spatial = var_data_area.mean(dim=time_dim)
                            fig_map = plt.figure(figsize=(10, 8))
                            ax_map = plt.axes(projection=ccrs.PlateCarree())
                            
                            mean_spatial.plot.contourf(ax=ax_map, transform=ccrs.PlateCarree(), cmap='viridis', cbar_kwargs={'label': f'Mean {long_name} ({units})'})
                            ax_map.coastlines()
                            gl = ax_map.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
                            gl.top_labels = False
                            gl.right_labels = False
                            ax_map.set_title(f'Spatial Distribution of Mean {long_name}')
                            st.pyplot(fig_map)

                        # --- Time Series of Spatial Average ---
                        with st.expander("Show Time Series of Spatially Averaged Values"):
                            import plotly.graph_objects as go
                            spatial_avg_ts = var_data_area.mean(dim=['latitude', 'longitude'])
                            fig_ts = go.Figure()
                            fig_ts.add_trace(go.Scatter(x=spatial_avg_ts[time_dim].values, y=spatial_avg_ts.values, mode='lines'))
                            fig_ts.update_layout(
                                title=f'Time Series of Spatially Averaged {long_name}',
                                xaxis_title='Time',
                                yaxis_title=f'{long_name} ({units})',
                                height=400
                            )
                            st.plotly_chart(fig_ts, use_container_width=True)

                        # --- Extreme Value Analysis ---
                        with st.expander("Show Extreme Value Analysis"):
                            st.markdown("##### Maps of Maximum and Minimum Values")
                            st.info("These maps show the highest and lowest values recorded at each geographic point over the entire time period.")
                            
                            col1, col2 = st.columns(2)

                            with col1:
                                # Map of Maximum
                                max_spatial = var_data_area.max(dim=time_dim)
                                fig_max = plt.figure(figsize=(10, 8))
                                ax_max = plt.axes(projection=ccrs.PlateCarree())
                                max_spatial.plot.contourf(ax=ax_max, transform=ccrs.PlateCarree(), cmap='Reds', cbar_kwargs={'label': f'Max {long_name} ({units})'})
                                ax_max.coastlines()
                                gl_max = ax_max.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
                                gl_max.top_labels = False
                                gl_max.right_labels = False
                                ax_max.set_title(f'Maximum Value Distribution')
                                st.pyplot(fig_max)

                            with col2:
                                # Map of Minimum
                                min_spatial = var_data_area.min(dim=time_dim)
                                fig_min = plt.figure(figsize=(10, 8))
                                ax_min = plt.axes(projection=ccrs.PlateCarree())
                                min_spatial.plot.contourf(ax=ax_min, transform=ccrs.PlateCarree(), cmap='Blues_r', cbar_kwargs={'label': f'Min {long_name} ({units})'})
                                ax_min.coastlines()
                                gl_min = ax_min.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
                                gl_min.top_labels = False
                                gl_min.right_labels = False
                                ax_min.set_title(f'Minimum Value Distribution')
                                st.pyplot(fig_min)
                        
                        st.markdown("---")

        except Exception as e:
            st.error(f"Failed to generate report for {os.path.basename(file_path)}: {e}")
            st.exception(e)
