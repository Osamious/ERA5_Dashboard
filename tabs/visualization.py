"""
Visualization tab for the ERA5 Dashboard.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import os
import pandas as pd
import xarray as xr

from utils.constants import SHORT_NAME_MAP
from utils.helpers import get_time_dim_name, convert_time_array_to_local_timezone, format_datetime_with_timezone

# --- Helper functions for spatial plotting ---
def plot_spatial_map(ds, time_dim, lon, lat, time_to_plot, contour_var=None, quiver=False, 
                     contour_type="filled", contour_levels=20, colormap="RdYlBu_r"):
    # Import matplotlib and cartopy only when needed for spatial plotting
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Set map extent
    lon_min, lon_max = lon.min().item(), lon.max().item()
    lat_min, lat_max = lat.min().item(), lat.max().item()
    buffer = 0.5
    ax.set_extent([lon_min - buffer, lon_max + buffer, lat_min - buffer, lat_max + buffer], crs=ccrs.PlateCarree())

    # Plot Contour with fixed scale
    if contour_var and contour_var != "None":
        data_to_plot = ds[contour_var].sel({time_dim: time_to_plot})
        
        # Ensure data is 2D for plotting by removing any singleton dimensions
        # and handling the case where we might have extra dimensions
        while data_to_plot.ndim > 2:
            # If there are more than 2 dimensions, squeeze out size-1 dimensions first
            data_to_plot = data_to_plot.squeeze()
            
            # If still more than 2D after squeezing, take the first slice of the first extra dimension
            if data_to_plot.ndim > 2:
                # Get the name of the first dimension that's not lat/lon
                dims_to_remove = []
                for dim in data_to_plot.dims:
                    if dim not in ['latitude', 'longitude', 'lat', 'lon'] and data_to_plot.sizes[dim] > 1:
                        dims_to_remove.append(dim)
                        break
                
                if dims_to_remove:
                    # Take the first slice of the extra dimension
                    data_to_plot = data_to_plot.isel({dims_to_remove[0]: 0})
                else:
                    # Last resort: just squeeze all size-1 dimensions
                    data_to_plot = data_to_plot.squeeze(drop=True)
                    break
        
        # Final check: ensure we have 2D data
        if data_to_plot.ndim != 2:
            st.error(f"Unable to create 2D plot: data has {data_to_plot.ndim} dimensions after processing. "
                    f"Dimensions: {data_to_plot.dims}")
            return None
            
        var_info = SHORT_NAME_MAP.get(contour_var, {})
        # Calculate fixed scale based on entire dataset min/max
        full_data = ds[contour_var]
        vmin, vmax = full_data.min().item(), full_data.max().item()
        
        # Create contour levels
        if isinstance(contour_levels, int):
            levels = np.linspace(vmin, vmax, contour_levels)
        else:
            levels = contour_levels
        
        if contour_type == "filled":
            contour = ax.contourf(lon, lat, data_to_plot, transform=ccrs.PlateCarree(), 
                                cmap=colormap, levels=levels, vmin=vmin, vmax=vmax, extend='both')
            plt.colorbar(contour, ax=ax, shrink=0.7, pad=0.02, 
                        label=f"{var_info.get('name', contour_var)} ({var_info.get('units', 'N/A')})")
        elif contour_type == "lines":
            contour = ax.contour(lon, lat, data_to_plot, transform=ccrs.PlateCarree(), 
                               levels=levels, colors='black', linewidths=1.0)
            ax.clabel(contour, inline=True, fontsize=8, fmt='%.1f')
            # Add a colorbar for reference even with line contours
            cbar_contour = ax.contourf(lon, lat, data_to_plot, transform=ccrs.PlateCarree(), 
                                     cmap=colormap, levels=levels, vmin=vmin, vmax=vmax, alpha=0.3)
            plt.colorbar(cbar_contour, ax=ax, shrink=0.7, pad=0.02, 
                        label=f"{var_info.get('name', contour_var)} ({var_info.get('units', 'N/A')})")
        elif contour_type == "both":
            # Filled contours with line overlays
            contour_filled = ax.contourf(lon, lat, data_to_plot, transform=ccrs.PlateCarree(), 
                                       cmap=colormap, levels=levels, vmin=vmin, vmax=vmax, extend='both')
            contour_lines = ax.contour(lon, lat, data_to_plot, transform=ccrs.PlateCarree(), 
                                     levels=levels, colors='black', linewidths=0.5, alpha=0.7)
            ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')
            plt.colorbar(contour_filled, ax=ax, shrink=0.7, pad=0.02, 
                        label=f"{var_info.get('name', contour_var)} ({var_info.get('units', 'N/A')})")

    # Plot Quiver with enhanced wind speed visualization
    if quiver and 'u10' in ds.data_vars and 'v10' in ds.data_vars:
        u_data = ds['u10'].sel({time_dim: time_to_plot}).squeeze()
        v_data = ds['v10'].sel({time_dim: time_to_plot}).squeeze()
        
        # Calculate wind speed for color coding
        wind_speed = np.sqrt(u_data**2 + v_data**2)
        
        # Subsample for readability
        step = max(1, len(lon) // 20)
        lon_sub = lon[::step]
        lat_sub = lat[::step]
        u_sub = u_data[::step, ::step]
        v_sub = v_data[::step, ::step]
        wind_speed_sub = wind_speed[::step, ::step]
        
        # Create color-coded quiver plot
        quiver_plot = ax.quiver(lon_sub, lat_sub, u_sub, v_sub, wind_speed_sub,
                              transform=ccrs.PlateCarree(), alpha=0.8, scale=200,
                              cmap='plasma', scale_units='width')
        
        # Add colorbar for wind speed
        cbar = plt.colorbar(quiver_plot, ax=ax, shrink=0.7, pad=0.08, aspect=20)
        cbar.set_label('Wind Speed (m/s)', rotation=270, labelpad=15)

    # Map features
    ax.coastlines()
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False

    # Title
    title_parts = []
    if contour_var and contour_var != "None":
        title_parts.append(f"Contour of {SHORT_NAME_MAP.get(contour_var, {}).get('name', contour_var)}")
    if quiver:
        title_parts.append("Wind Vectors (colored by speed)")
    title = " and ".join(title_parts) if title_parts else "Geographical Map"
    ax.set_title(title, pad=20)

    return fig

def plot_3d_surface(ds, time_dim, lon, lat, time_to_plot, contour_var):
    data_3d = ds[contour_var].sel({time_dim: time_to_plot})
    
    # Ensure data is 2D for plotting by removing any singleton dimensions
    # and handling the case where we might have extra dimensions  
    while data_3d.ndim > 2:
        # If there are more than 2 dimensions, squeeze out size-1 dimensions first
        data_3d = data_3d.squeeze()
        
        # If still more than 2D after squeezing, take the first slice of the first extra dimension
        if data_3d.ndim > 2:
            # Get the name of the first dimension that's not lat/lon
            dims_to_remove = []
            for dim in data_3d.dims:
                if dim not in ['latitude', 'longitude', 'lat', 'lon'] and data_3d.sizes[dim] > 1:
                    dims_to_remove.append(dim)
                    break
            
            if dims_to_remove:
                # Take the first slice of the extra dimension
                data_3d = data_3d.isel({dims_to_remove[0]: 0})
            else:
                # Last resort: just squeeze all size-1 dimensions
                data_3d = data_3d.squeeze(drop=True)
                break
    
    # Final check: ensure we have 2D data
    if data_3d.ndim != 2:
        st.error(f"Unable to create 3D surface plot: data has {data_3d.ndim} dimensions after processing. "
                f"Dimensions: {data_3d.dims}")
        return None
        
    var_info = SHORT_NAME_MAP.get(contour_var, {})
    # Calculate fixed scale based on entire dataset min/max
    full_data = ds[contour_var]
    cmin, cmax = full_data.min().item(), full_data.max().item()
    
    fig = go.Figure(data=[go.Surface(z=data_3d.values, x=lon, y=lat, colorscale='jet',
                                   cmin=cmin, cmax=cmax)])
    fig.update_layout(
        title=f"3D Surface Plot of {var_info.get('name', contour_var)}",
        scene={
            'xaxis_title': 'Longitude',
            'yaxis_title': 'Latitude',
            'zaxis_title': f"{var_info.get('name', contour_var)} ({var_info.get('units', 'N/A')})",
            'zaxis': {'range': [cmin, cmax]}  # Fixed Z-axis range
        },
        height=700
    )
    return fig

def plot_interactive_contour(ds, time_dim, lon, lat, time_to_plot, contour_var, 
                           contour_levels=20, colormap="RdYlBu"):
    """Create an interactive Plotly contour plot."""
    data_to_plot = ds[contour_var].sel({time_dim: time_to_plot})
    
    # Ensure data is 2D for plotting by removing any singleton dimensions
    # and handling the case where we might have extra dimensions
    while data_to_plot.ndim > 2:
        # If there are more than 2 dimensions, squeeze out size-1 dimensions first
        data_to_plot = data_to_plot.squeeze()
        
        # If still more than 2D after squeezing, take the first slice of the first extra dimension
        if data_to_plot.ndim > 2:
            # Get the name of the first dimension that's not lat/lon
            dims_to_remove = []
            for dim in data_to_plot.dims:
                if dim not in ['latitude', 'longitude', 'lat', 'lon'] and data_to_plot.sizes[dim] > 1:
                    dims_to_remove.append(dim)
                    break
            
            if dims_to_remove:
                # Take the first slice of the extra dimension
                data_to_plot = data_to_plot.isel({dims_to_remove[0]: 0})
            else:
                # Last resort: just squeeze all size-1 dimensions
                data_to_plot = data_to_plot.squeeze(drop=True)
                break
    
    # Final check: ensure we have 2D data
    if data_to_plot.ndim != 2:
        st.error(f"Unable to create 2D plot: data has {data_to_plot.ndim} dimensions after processing. "
                f"Dimensions: {data_to_plot.dims}")
        return None
        
    var_info = SHORT_NAME_MAP.get(contour_var, {})
    
    # Calculate fixed scale based on entire dataset min/max
    full_data = ds[contour_var]
    zmin, zmax = full_data.min().item(), full_data.max().item()
    
    # Create mesh grids for plotting
    lon_mesh, lat_mesh = np.meshgrid(lon, lat)
    
    fig = go.Figure()
    
    # Add contour plot
    fig.add_trace(go.Contour(
        z=data_to_plot.values,
        x=lon.values,
        y=lat.values,
        colorscale=colormap,
        contours=dict(
            start=zmin,
            end=zmax,
            size=(zmax - zmin) / contour_levels,
            showlabels=True,
            labelfont=dict(size=10, color="black")
        ),
        colorbar=dict(
            title=f"{var_info.get('name', contour_var)}<br>({var_info.get('units', 'N/A')}"
        ),
        hovertemplate='<b>Longitude</b>: %{x:.2f}°<br>' +
                      '<b>Latitude</b>: %{y:.2f}°<br>' +
                      '<b>Value</b>: %{z:.2f} ' + var_info.get('units', '') +
                      '<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=f"Interactive Contour Plot: {var_info.get('name', contour_var)}",
        xaxis_title="Longitude (°)",
        yaxis_title="Latitude (°)",
        height=600,
        showlegend=False
    )
    
    # Set aspect ratio to be geographic
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    
    return fig

def plot_interactive_combined(ds, time_dim, lon, lat, time_to_plot, contour_var, 
                           contour_levels=20, colormap="RdYlBu"):
    """Create an interactive Plotly plot combining contours and wind vectors."""
    fig = go.Figure()
    
    # Add contour plot first
    if contour_var and contour_var != "None":
        data_to_plot = ds[contour_var].sel({time_dim: time_to_plot})
        
        # Ensure data is 2D for plotting by removing any singleton dimensions
        # and handling the case where we might have extra dimensions
        while data_to_plot.ndim > 2:
            # If there are more than 2 dimensions, squeeze out size-1 dimensions first
            data_to_plot = data_to_plot.squeeze()
            
            # If still more than 2D after squeezing, take the first slice of the first extra dimension
            if data_to_plot.ndim > 2:
                # Get the name of the first dimension that's not lat/lon
                dims_to_remove = []
                for dim in data_to_plot.dims:
                    if dim not in ['latitude', 'longitude', 'lat', 'lon'] and data_to_plot.sizes[dim] > 1:
                        dims_to_remove.append(dim)
                        break
                
                if dims_to_remove:
                    # Take the first slice of the extra dimension
                    data_to_plot = data_to_plot.isel({dims_to_remove[0]: 0})
                else:
                    # Last resort: just squeeze all size-1 dimensions
                    data_to_plot = data_to_plot.squeeze(drop=True)
                    break
        
        # Final check: ensure we have 2D data
        if data_to_plot.ndim != 2:
            st.error(f"Unable to create combined plot: data has {data_to_plot.ndim} dimensions after processing. "
                    f"Dimensions: {data_to_plot.dims}")
            return None
            
        var_info = SHORT_NAME_MAP.get(contour_var, {})
        
        # Calculate fixed scale based on entire dataset min/max
        full_data = ds[contour_var]
        zmin, zmax = full_data.min().item(), full_data.max().item()
        
        # Add contour plot
        fig.add_trace(go.Contour(
            z=data_to_plot.values,
            x=lon.values,
            y=lat.values,
            colorscale=colormap,
            contours=dict(
                start=zmin,
                end=zmax,
                size=(zmax - zmin) / contour_levels,
                showlabels=True,
                labelfont=dict(size=8, color="white")
            ),
            colorbar=dict(
                title=f"{var_info.get('name', contour_var)}<br>({var_info.get('units', 'N/A')})",
                x=1.0
            ),
            hovertemplate='<b>Longitude</b>: %{x:.2f}°<br>' +
                          '<b>Latitude</b>: %{y:.2f}°<br>' +
                          '<b>' + var_info.get('name', contour_var) + '</b>: %{z:.2f} ' + var_info.get('units', '') +
                          '<extra></extra>',
            name="Contour"
        ))
    
    # Add wind vectors if available
    if 'u10' in ds.data_vars and 'v10' in ds.data_vars:
        u_data = ds['u10'].sel({time_dim: time_to_plot}).squeeze()
        v_data = ds['v10'].sel({time_dim: time_to_plot}).squeeze()
        wind_speed = np.sqrt(u_data**2 + v_data**2)
        
        # Subsample for readability
        step = max(1, len(lon) // 15)
        lon_sub = lon[::step].values
        lat_sub = lat[::step].values
        u_sub = u_data[::step, ::step].values
        v_sub = v_data[::step, ::step].values
        wind_speed_sub = wind_speed[::step, ::step].values
        
        # Create mesh grids
        lon_mesh, lat_mesh = np.meshgrid(lon_sub, lat_sub)
        
        # Add wind vectors using a more sophisticated approach
        scale_factor = 0.3
        
        # Create line traces for arrows (batched for performance)
        for i in range(0, len(lon_mesh.flatten()), 3):  # Show every 3rd arrow
            lon_start = lon_mesh.flatten()[i]
            lat_start = lat_mesh.flatten()[i]
            lon_end = lon_start + u_sub.flatten()[i] * scale_factor
            lat_end = lat_start + v_sub.flatten()[i] * scale_factor
            speed = wind_speed_sub.flatten()[i]
            
            # Arrow shaft
            fig.add_trace(go.Scatter(
                x=[lon_start, lon_end],
                y=[lat_start, lat_end],
                mode='lines',
                line=dict(color='black', width=1.5),
                showlegend=False,
                hovertemplate=f'<b>Wind Speed</b>: {speed:.1f} m/s<extra></extra>',
                name="Wind"
            ))
    
    # Update layout
    title_parts = []
    if contour_var and contour_var != "None":
        var_info = SHORT_NAME_MAP.get(contour_var, {})
        title_parts.append(f"Contour: {var_info.get('name', contour_var)}")
    if 'u10' in ds.data_vars and 'v10' in ds.data_vars:
        title_parts.append("Wind Vectors")
    
    title = " + ".join(title_parts) if title_parts else "Interactive Plot"
    
    fig.update_layout(
        title=title,
        xaxis_title="Longitude (°)",
        yaxis_title="Latitude (°)",
        height=650,
        showlegend=False
    )
    
    # Set aspect ratio to be geographic
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    
    return fig

def plot_interactive_wind_vectors(ds, time_dim, lon, lat, time_to_plot):
    """Create an interactive Plotly wind vector plot."""
    u_data = ds['u10'].sel({time_dim: time_to_plot}).squeeze()
    v_data = ds['v10'].sel({time_dim: time_to_plot}).squeeze()
    
    # Calculate wind speed for color coding
    wind_speed = np.sqrt(u_data**2 + v_data**2)
    
    # Subsample for readability (similar to matplotlib version)
    step = max(1, len(lon) // 20)
    lon_sub = lon[::step].values
    lat_sub = lat[::step].values
    u_sub = u_data[::step, ::step].values
    v_sub = v_data[::step, ::step].values
    wind_speed_sub = wind_speed[::step, ::step].values
    
    # Create mesh grids
    lon_mesh, lat_mesh = np.meshgrid(lon_sub, lat_sub)
    
    # Create figure
    fig = go.Figure()
    
    # Add wind speed as background (optional filled contour)
    fig.add_trace(go.Contour(
        z=wind_speed.values,
        x=lon.values,
        y=lat.values,
        colorscale="plasma",
        contours=dict(showlabels=False),
        colorbar=dict(
            title="Wind Speed<br>(m/s)"
        ),
        hovertemplate='<b>Longitude</b>: %{x:.2f}°<br>' +
                      '<b>Latitude</b>: %{y:.2f}°<br>' +
                      '<b>Wind Speed</b>: %{z:.2f} m/s' +
                      '<extra></extra>',
        showscale=True
    ))
    
    # Add wind vectors using scatter plot with custom symbols
    # Calculate arrow endpoints
    scale_factor = 0.5  # Adjust this to make arrows longer/shorter
    lon_end = lon_mesh.flatten() + u_sub.flatten() * scale_factor
    lat_end = lat_mesh.flatten() + v_sub.flatten() * scale_factor
    
    # Create arrow traces (simplified approach using lines)
    for i in range(0, len(lon_mesh.flatten()), 5):  # Show every 5th arrow to avoid clutter
        fig.add_trace(go.Scatter(
            x=[lon_mesh.flatten()[i], lon_end[i]],
            y=[lat_mesh.flatten()[i], lat_end[i]], 
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add arrowhead (simplified)
        fig.add_trace(go.Scatter(
            x=[lon_end[i]],
            y=[lat_end[i]],
            mode='markers',
            marker=dict(symbol='triangle-up', size=8, color='black'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Update layout
    fig.update_layout(
        title="Interactive Wind Vectors (colored by speed)",
        xaxis_title="Longitude (°)",
        yaxis_title="Latitude (°)",
        height=600,
        showlegend=False
    )
    
    # Set aspect ratio to be geographic
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    
    return fig

def render_visualization_tab():
    """Renders the Visualization tab content."""
    st.header("Visualize Data")

    # --- File Selection from Database ---
    database_path = "database"
    if not os.path.exists(database_path):
        st.warning("No database folder found. Please download some data first in the 'Data Fetching' tab.")
        return
    
    nc_files = [f for f in os.listdir(database_path) if f.endswith('.nc')]
    if not nc_files:
        st.warning("No NetCDF files found in the database folder. Please download some data first in the 'Data Fetching' tab.")
        return
    
    st.info("Select one or multiple files from your database to visualize.")
    selected_files = st.multiselect(
        "Choose NetCDF file(s) to visualize",
        nc_files,
        key="visualization_file_selector",
        help="You can select multiple files to compare them. Point data files will be shown as time series, area data as maps."
    )

    if not selected_files:
        st.info("Please select at least one file to visualize.")
        return    # Process selected files to determine their types and variables
    point_files = []
    area_files = []
    
    for file in selected_files:
        file_path = os.path.join(database_path, file)
        try:
            import xarray as xr
            with xr.open_dataset(file_path) as ds:
                is_area = 'latitude' in ds.dims and 'longitude' in ds.dims and len(ds.latitude) > 1 and len(ds.longitude) > 1
                if is_area:
                    area_files.append(file_path)
                else:
                    point_files.append(file_path)
        except Exception as e:
            st.error(f"Error reading {file}: {e}")
            continue    # --- 1. Plotting Point Files (Time Series Comparison) ---
    if point_files:
        st.subheader("Time Series Comparison Plot")
        st.info("All selected point-data files are plotted on the graph below. Click legend items to toggle visibility.")
        
        # Check if any point files have wind data
        wind_files = []
        multi_var_files = []
        for file_path in point_files:
            try:
                import xarray as xr
                with xr.open_dataset(file_path) as ds:
                    vars_in_file = list(ds.data_vars)
                    if 'u10' in vars_in_file and 'v10' in vars_in_file:
                        wind_files.append(os.path.basename(file_path))
                        # Remove wind components and add wind_speed for counting
                        vars_in_file = [v for v in vars_in_file if v not in ['u10', 'v10']] + ['wind_speed']
                    if len(vars_in_file) > 1:
                        multi_var_files.append(file_path)
            except:
                pass
        
        if wind_files:
            st.info(f"🌬️ **Wind data detected** in: {', '.join(wind_files)}. Wind speed will be calculated and plotted automatically.")
        
        # Multi-variable plotting option
        plot_option = "Combined Plot"
        if multi_var_files:
            st.markdown("#### Multi-Variable Plot Options")
            plot_option = st.radio(
                "How would you like to display files with multiple variables?",
                ["Combined Plot", "Separate Plots"],
                index=0,
                help="Combined Plot: All variables on one chart with dual y-axes. Separate Plots: Each variable gets its own subplot.",
                horizontal=True
            )
        
        if plot_option == "Combined Plot":
            # Single combined plot with potentially dual y-axes
            fig_ts = go.Figure()
            
            # Track different units for secondary y-axis
            primary_unit = None
            secondary_traces = []
            
            for file_path in point_files:
                try:
                    import xarray as xr
                    with xr.open_dataset(file_path) as ds:
                        time_dim = get_time_dim_name(ds)
                        if not time_dim:
                            st.warning(f"Could not find time dimension in `{os.path.basename(file_path)}`. Skipping.")
                            continue

                        vars_in_file = list(ds.data_vars)
                        if 'u10' in vars_in_file and 'v10' in vars_in_file:
                            ds['wind_speed'] = np.sqrt(ds['u10']**2 + ds['v10']**2)
                            ds['wind_speed'].attrs['units'] = 'm/s'
                            ds['wind_speed'].attrs['long_name'] = '10m Wind Speed'
                            vars_in_file = [v for v in vars_in_file if v not in ['u10', 'v10']] + ['wind_speed']

                        for var_name in vars_in_file:
                            var_data = ds[var_name].squeeze()
                            if var_data.isnull().all():
                                continue

                            var_info = SHORT_NAME_MAP.get(var_name, {})
                            units = var_info.get('units', ds[var_name].attrs.get('units', ''))
                            long_name = var_info.get('name', var_name)
                            trace_name = f"{os.path.basename(file_path)}: {long_name}"
                            
                            # Determine if this should go on secondary y-axis
                            use_secondary = False
                            if primary_unit is None:
                                primary_unit = units
                            elif units != primary_unit and len(vars_in_file) > 1:
                                use_secondary = True
                                secondary_traces.append(trace_name)                            # Convert time to local timezone for display
                            local_times = convert_time_array_to_local_timezone(ds[time_dim].values)
                            current_tz = st.session_state.get('timezone_offset', 3)
                            tz_str = f"UTC{'+' if current_tz >= 0 else ''}{current_tz}"
                            
                            fig_ts.add_trace(go.Scatter(
                                x=local_times,
                                y=var_data.values,
                                mode='lines',
                                name=trace_name,
                                yaxis='y2' if use_secondary else 'y',
                                customdata=[units]*len(ds[time_dim]),
                                hovertemplate=f'<b>Time ({tz_str})</b>: %{{x}}<br><b>Value</b>: %{{y:.2f}} %{{customdata[0]}}<extra></extra>'
                            ))
                except Exception as e:
                    st.error(f"Could not process or plot time series for `{os.path.basename(file_path)}`: {e}")            # Set up layout with potential dual y-axes
            current_tz = st.session_state.get('timezone_offset', 3)
            tz_str = f"UTC{'+' if current_tz >= 0 else ''}{current_tz}"
            layout_config = {
                'title': f"Time Series Comparison (Local Time {tz_str})",
                'xaxis_title': f"Time ({tz_str})",
                'height': 600,
                'legend_title': "File: Variable"
            }
            
            if secondary_traces:
                # Dual y-axis setup
                layout_config.update({
                    'yaxis': dict(title=f"Primary Variables ({primary_unit})", side='left'),
                    'yaxis2': dict(title="Secondary Variables (mixed units)", side='right', overlaying='y')
                })
                if secondary_traces:
                    st.info(f"📊 **Dual Y-axis**: Secondary axis used for: {', '.join(secondary_traces)}")
            else:
                # Single y-axis
                if primary_unit:
                    # Get the variable name for single variable files
                    var_names = []
                    for file_path in point_files:
                        try:
                            with xr.open_dataset(file_path) as ds:
                                vars_in_file = list(ds.data_vars)
                                if 'u10' in vars_in_file and 'v10' in vars_in_file:
                                    vars_in_file = [v for v in vars_in_file if v not in ['u10', 'v10']] + ['wind_speed']
                                for var_name in vars_in_file:
                                    var_info = SHORT_NAME_MAP.get(var_name, {})
                                    long_name = var_info.get('name', var_name)
                                    if long_name not in var_names:
                                        var_names.append(long_name)
                        except:
                            pass
                    
                    if len(var_names) == 1:
                        layout_config['yaxis_title'] = f"{var_names[0]} ({primary_unit})"
                    else:
                        layout_config['yaxis_title'] = f"Value ({primary_unit})"
                else:
                    layout_config['yaxis_title'] = "Value (units may vary, see hover)"
            
            fig_ts.update_layout(**layout_config)
            st.plotly_chart(fig_ts, use_container_width=True)
            
        else:  # Separate Plots
            from plotly.subplots import make_subplots
            
            # Create separate plots for each variable
            for file_path in point_files:
                try:
                    import xarray as xr
                    with xr.open_dataset(file_path) as ds:
                        time_dim = get_time_dim_name(ds)
                        if not time_dim:
                            st.warning(f"Could not find time dimension in `{os.path.basename(file_path)}`. Skipping.")
                            continue

                        vars_in_file = list(ds.data_vars)
                        if 'u10' in vars_in_file and 'v10' in vars_in_file:
                            ds['wind_speed'] = np.sqrt(ds['u10']**2 + ds['v10']**2)
                            ds['wind_speed'].attrs['units'] = 'm/s'
                            ds['wind_speed'].attrs['long_name'] = '10m Wind Speed'
                            vars_in_file = [v for v in vars_in_file if v not in ['u10', 'v10']] + ['wind_speed']
                        
                        if len(vars_in_file) > 1:
                            st.markdown(f"#### {os.path.basename(file_path)}")
                            
                            # Create subplots for each variable
                            subplot_titles = []
                            for var_name in vars_in_file:
                                var_info = SHORT_NAME_MAP.get(var_name, {})
                                long_name = var_info.get('name', var_name)
                                units = var_info.get('units', ds[var_name].attrs.get('units', ''))
                                subplot_titles.append(f"{long_name} ({units})")
                            
                            fig_sub = make_subplots(
                                rows=len(vars_in_file), cols=1,
                                subplot_titles=subplot_titles,
                                vertical_spacing=0.15,  # Increased spacing between subplots
                                specs=[[{"secondary_y": False}] for _ in range(len(vars_in_file))]
                            )
                            
                            for idx, var_name in enumerate(vars_in_file):
                                var_data = ds[var_name].squeeze()
                                if var_data.isnull().all():
                                    continue
                                
                                var_info = SHORT_NAME_MAP.get(var_name, {})
                                units = var_info.get('units', ds[var_name].attrs.get('units', ''))
                                
                                fig_sub.add_trace(
                                    go.Scatter(
                                        x=ds[time_dim].values,
                                        y=var_data.values,
                                        mode='lines',
                                        name=var_name,
                                        showlegend=False,
                                        hovertemplate='<b>Time</b>: %{x}<br><b>Value</b>: %{y:.2f} ' + units + '<extra></extra>'
                                    ),
                                    row=idx+1, col=1                                )
                            
                            fig_sub.update_layout(
                                height=max(400, 250 * len(vars_in_file) + 150),  # Increased height per subplot
                                title_text=f"Time Series for {os.path.basename(file_path)}",
                                showlegend=False,
                                margin=dict(t=100, b=50, l=50, r=50)  # Add margins for better spacing
                            )
                            fig_sub.update_xaxes(title_text="Time", row=len(vars_in_file), col=1)
                            
                            st.plotly_chart(fig_sub, use_container_width=True)
                            
                            # Add spacing between files
                            st.markdown("<br>", unsafe_allow_html=True)
                        else:
                            # Single variable - use simple plot
                            var_name = vars_in_file[0]
                            var_data = ds[var_name].squeeze()
                            var_info = SHORT_NAME_MAP.get(var_name, {})
                            units = var_info.get('units', ds[var_name].attrs.get('units', ''))
                            long_name = var_info.get('name', var_name)
                            
                            fig_single = go.Figure()
                            fig_single.add_trace(go.Scatter(
                                x=ds[time_dim].values,
                                y=var_data.values,
                                mode='lines',
                                name=long_name,
                                hovertemplate='<b>Time</b>: %{x}<br><b>Value</b>: %{y:.2f} ' + units + '<extra></extra>'
                            ))
                            
                            fig_single.update_layout(
                                title=f"{os.path.basename(file_path)}: {long_name}",
                                xaxis_title="Time",
                                yaxis_title=f"{long_name} ({units})",
                                height=400
                            )
                            st.plotly_chart(fig_single, use_container_width=True)
                            
                            # Add spacing between files
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                except Exception as e:
                    st.error(f"Could not process or plot time series for `{os.path.basename(file_path)}`: {e}")

    # --- 2. Plotting Area Files (Sequentially) ---
    if area_files:
        st.subheader("Spatial Plots")
        if point_files: # If we also plotted point files, add a separator
            st.markdown("---")
        
        # Add information about visualization types
        with st.expander("ℹ️ About Spatial Visualization Types", expanded=False):
            st.markdown("""
            **Contour Plot Types:**
            - **Filled Contours**: Colored areas showing value ranges - great for seeing overall patterns
            - **Line Contours**: Labeled contour lines - precise for reading exact values  
            - **Both**: Combination of filled areas with labeled lines - best of both worlds
            
            **Plot Types:**
            - **2D Map**: Traditional geographic map with coastlines and gridlines (matplotlib)
            - **Interactive Contour**: Zoom and pan-enabled contour plots (plotly)
            - **3D Surface**: Three-dimensional visualization showing data as a landscape
            
            **Wind Visualization**: When wind data (u10, v10) is available, you can overlay wind vectors showing direction and speed.
            """)
        
        st.info("Each selected area-data file is plotted in its own section below.")

        for i, file_path in enumerate(area_files):
            with st.container():
                st.markdown(f"### Visualizing: `{os.path.basename(file_path)}`")
                try:
                    # Import xarray only when needed
                    import xarray as xr
                    
                    with xr.open_dataset(file_path) as ds:
                        time_dim = get_time_dim_name(ds)
                        if not time_dim:
                            st.error(f"Could not create spatial plot for `{os.path.basename(file_path)}` because a time coordinate was not found.")
                            continue                        # --- Plot Configuration ---
                        st.markdown("#### Plot Configuration")
                        widget_key_suffix = f"_{os.path.basename(file_path).replace('.', '_')}_{i}"                        # Process variables and handle wind components
                        vars_to_plot = list(ds.data_vars)
                        has_wind = 'u10' in vars_to_plot and 'v10' in vars_to_plot
                        
                        # Remove wind components from base layer options if present
                        if has_wind:
                            # Show notification about wind data detection
                            st.info("🌬️ **Wind data detected**: This file contains u10 and v10 wind components. Use 'Show Wind Vectors' to visualize wind direction and speed.")
                            # Remove u10 and v10 from contour options (wind vectors will handle wind visualization)
                            vars_to_plot = [v for v in vars_to_plot if v not in ['u10', 'v10']]

                        # Base Layer (Contour) Configuration
                        st.markdown("**Base Layer Configuration**")
                        
                        # Create two columns for contour controls
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            contour_var = st.selectbox(
                                "Contour Variable",
                                ["None"] + vars_to_plot,
                                index=1 if vars_to_plot else 0,
                                key="contour_var" + widget_key_suffix,
                                help="Choose a variable to plot as contours."
                            )
                        
                        # Show contour options only if a variable is selected
                        if contour_var != "None":
                            with col2:
                                contour_type = st.selectbox(
                                    "Contour Type",
                                    ["filled", "lines", "both"],
                                    index=0,
                                    key="contour_type" + widget_key_suffix,
                                    help="Choose contour style: filled (colored areas), lines (labeled contour lines), or both."
                                )
                            
                            # Additional contour controls in a new row
                            col3, col4 = st.columns(2)
                            
                            with col3:
                                contour_levels = st.slider(
                                    "Number of Contour Levels",
                                    min_value=5,
                                    max_value=50,
                                    value=20,
                                    step=5,
                                    key="contour_levels" + widget_key_suffix,
                                    help="Number of contour intervals to display."
                                )
                            
                            with col4:
                                colormap_options = [
                                    "RdYlBu_r", "viridis", "plasma", "coolwarm", "RdBu_r", 
                                    "seismic", "jet", "rainbow", "terrain", "ocean"
                                ]
                                colormap = st.selectbox(
                                    "Color Map",
                                    colormap_options,
                                    index=0,
                                    key="colormap" + widget_key_suffix,
                                    help="Choose the color scheme for contours."
                                )
                        else:
                            contour_type = "filled"
                            contour_levels = 20
                            colormap = "RdYlBu_r"

                        # Plot Type (must be defined before wind overlay options)
                        plot_type = st.radio(
                            "Plot Type",
                            ["2D Map", "Interactive Contour", "3D Surface"],
                            index=0,
                            key="plot_type" + widget_key_suffix,
                            help="Choose plot style: 2D Map (matplotlib with geographic features), Interactive Contour (plotly with zoom/pan), or 3D Surface (3D visualization).",
                            horizontal=True
                        )

                        # Wind Quiver Overlay (only show if wind components are available)
                        quiver = False
                        wind_overlay_option = "None"  # Initialize default value
                        if has_wind:
                            if plot_type == "Interactive Contour":
                                wind_overlay_option = st.radio(
                                    "Wind Vector Display",
                                    ["None", "Separate Plot", "Combined Overlay"],
                                    index=0,
                                    key="wind_overlay" + widget_key_suffix,
                                    help="Choose how to display wind vectors with interactive contours.",
                                    horizontal=True
                                )
                                quiver = wind_overlay_option != "None"
                            else:
                                quiver = st.checkbox(
                                    "Show Wind Vectors",
                                    value=False,
                                    key="quiver" + widget_key_suffix,
                                    help="Overlay wind direction and speed as arrows. Arrows are colored by wind speed with a colorbar scale."
                                )                        # Time Selection with timezone conversion
                        times_available = ds[time_dim].values
                        if len(times_available) > 1:
                            # Convert to local timezone and format for display
                            time_options = [format_datetime_with_timezone(t) for t in times_available]
                            current_tz = st.session_state.get('timezone_offset', 3)
                            tz_str = f"UTC{'+' if current_tz >= 0 else ''}{current_tz}"
                            time_index = st.select_slider(
                                f"Time Index (Local Time {tz_str})",
                                options=list(range(len(time_options))),
                                value=0,
                                format_func=lambda x: time_options[x],
                                key="time_slider" + widget_key_suffix,
                                help=f"Choose a time slice from {len(times_available)} available time steps. Times shown in {tz_str}."
                            )
                            time_to_plot = times_available[time_index]
                        else:
                            time_to_plot = times_available[0]
                            st.write(f"**Available Time:** {format_datetime_with_timezone(time_to_plot)}")

                        # Generate Plot
                        if contour_var != "None" or quiver:
                            st.markdown("#### Generated Plot")
                            
                            # Determine lon/lat coordinates
                            lon = ds.longitude if 'longitude' in ds.coords else ds.coords['longitude']
                            lat = ds.latitude if 'latitude' in ds.coords else ds.coords['latitude']

                            try:
                                if plot_type == "2D Map":
                                    fig = plot_spatial_map(ds, time_dim, lon, lat, time_to_plot, contour_var, quiver,
                                                         contour_type, contour_levels, colormap)
                                    st.pyplot(fig)
                                elif plot_type == "Interactive Contour" and contour_var != "None":
                                    # Convert matplotlib colormap name to plotly compatible
                                    plotly_colormap = colormap
                                    if colormap == "RdYlBu_r":
                                        plotly_colormap = "RdYlBu"
                                    elif colormap == "RdBu_r":
                                        plotly_colormap = "RdBu"
                                    
                                    # Check wind overlay option
                                    if has_wind and quiver and wind_overlay_option == "Combined Overlay":
                                        fig = plot_interactive_combined(ds, time_dim, lon, lat, time_to_plot, contour_var,
                                                                      contour_levels, plotly_colormap)
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        fig = plot_interactive_contour(ds, time_dim, lon, lat, time_to_plot, contour_var,
                                                                     contour_levels, plotly_colormap)
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Show wind vectors as separate plot if requested
                                        if has_wind and quiver and wind_overlay_option == "Separate Plot":
                                            st.markdown("**Wind Vectors (Separate Plot)**")
                                            fig_wind = plot_interactive_wind_vectors(ds, time_dim, lon, lat, time_to_plot)
                                            st.plotly_chart(fig_wind, use_container_width=True)
                                            
                                elif plot_type == "3D Surface" and contour_var != "None":
                                    fig = plot_3d_surface(ds, time_dim, lon, lat, time_to_plot, contour_var)
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    if plot_type in ["Interactive Contour", "3D Surface"]:
                                        st.warning(f"{plot_type} plots require a contour variable to be selected.")
                                    else:
                                        st.warning("Please select a contour variable or enable wind vectors to generate a plot.")
                            except Exception as e:
                                st.error(f"Could not generate plot for `{os.path.basename(file_path)}`: {e}")
                        else:
                            st.warning("Please select a contour variable or enable wind vectors to generate a plot.")

                        if i < len(area_files) - 1:  # Add separator between files
                            st.markdown("---")

                except Exception as e:
                    st.error(f"Could not process or visualize `{os.path.basename(file_path)}`: {e}")
