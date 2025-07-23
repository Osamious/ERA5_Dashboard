"""
Report tab for the ERA5 Dashboard.
"""

import streamlit as st
import xarray as xr
import numpy as np
import pandas as pd
import os
import warnings
import gc
from scipy import stats
from scipy.stats import zscore
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set environment variables to help with HDF5 issues
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ['NETCDF4_DECOMPRESSION_WARNING'] = '0'

# Suppress HDF5 and NetCDF warnings
warnings.filterwarnings('ignore', category=UserWarning, module='h5py')
warnings.filterwarnings('ignore', message='.*HDF5.*')
warnings.filterwarnings('ignore', message='.*NetCDF.*')

# Import cartopy with error handling
try:
    import cartopy.crs as ccrs
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False

# Import folium and streamlit-folium with error handling
try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

from utils.helpers import get_time_dim_name, format_datetime_with_timezone, apply_temperature_conversion, get_temperature_display_unit, update_dataset_temperature_units, categorize_nc_files
from utils.constants import SHORT_NAME_MAP

def analyze_trends_and_changes(data_array, time_dim, variable_name, units):
    """
    Analyze significant trends and changes in the data.
    
    Returns:
        dict: Contains trend analysis results
    """
    results = {}
    
    # Convert to pandas for easier time series analysis
    if len(data_array.dims) > 1:
        # For spatial data, take spatial average
        ts_data = data_array.mean(dim=[d for d in data_array.dims if d != time_dim])
    else:
        ts_data = data_array
    
    time_values = pd.to_datetime(ts_data[time_dim].values)
    values = ts_data.values
    
    # Remove NaN values
    valid_mask = ~np.isnan(values)
    if valid_mask.sum() < 5:  # Need at least 5 points for analysis
        return {"error": "Insufficient valid data points for trend analysis"}
    
    time_clean = time_values[valid_mask]
    values_clean = values[valid_mask]
    
    # Convert time to numeric for trend analysis
    time_numeric = (time_clean - time_clean[0]).total_seconds()
    
    # Linear trend analysis using scipy stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(time_numeric, values_clean)
    
    # Convert slope to units per year
    seconds_per_year = 365.25 * 24 * 3600
    trend_per_year = slope * seconds_per_year
    
    # Statistical significance
    is_significant = p_value < 0.05
    
    # Calculate relative change
    start_value = values_clean[0]
    end_value = values_clean[-1]
    total_change = end_value - start_value
    relative_change = (total_change / abs(start_value)) * 100 if start_value != 0 else 0
    
    # Detect change points using rolling statistics
    window_size = max(5, len(values_clean) // 10)
    rolling_mean = pd.Series(values_clean).rolling(window=window_size, center=True).mean()
    rolling_std = pd.Series(values_clean).rolling(window=window_size, center=True).std()
    
    # Find significant deviations
    z_scores = zscore(values_clean)
    extreme_indices = np.where(np.abs(z_scores) > 2.5)[0]  # 2.5 sigma threshold
    
    results = {
        "trend_per_year": trend_per_year,
        "r_squared": r_value**2,
        "p_value": p_value,
        "is_significant": is_significant,
        "total_change": total_change,
        "relative_change": relative_change,
        "start_value": start_value,
        "end_value": end_value,
        "extreme_events": len(extreme_indices),
        "extreme_indices": extreme_indices,
        "time_clean": time_clean,
        "values_clean": values_clean,
        "rolling_mean": rolling_mean,
        "z_scores": z_scores
    }
    
    return results

def calculate_climate_metrics(data_array, time_dim, variable_name):
    """
    Calculate climate research metrics for patterns and anomalies.
    
    Returns:
        dict: Contains various climate metrics
    """
    results = {}
    
    # Convert to pandas for easier analysis
    if len(data_array.dims) > 1:
        # For spatial data, take spatial average for time series metrics
        ts_data = data_array.mean(dim=[d for d in data_array.dims if d != time_dim])
    else:
        ts_data = data_array
    
    time_values = pd.to_datetime(ts_data[time_dim].values)
    values = ts_data.values
    
    # Remove NaN values
    valid_mask = ~np.isnan(values)
    if valid_mask.sum() < 10:  # Need at least 10 points
        return {"error": "Insufficient data for climate metrics"}
    
    values_clean = values[valid_mask]
    time_clean = time_values[valid_mask]
    
    # Basic statistics
    results["mean"] = np.mean(values_clean)
    results["std"] = np.std(values_clean)
    results["min"] = np.min(values_clean)
    results["max"] = np.max(values_clean)
    results["range"] = results["max"] - results["min"]
    
    # Percentiles for extreme analysis
    results["p5"] = np.percentile(values_clean, 5)
    results["p95"] = np.percentile(values_clean, 95)
    results["p99"] = np.percentile(values_clean, 99)
    results["p1"] = np.percentile(values_clean, 1)
    
    # Variability metrics
    results["coefficient_of_variation"] = (results["std"] / results["mean"]) * 100 if results["mean"] != 0 else 0
    results["interquartile_range"] = np.percentile(values_clean, 75) - np.percentile(values_clean, 25)
    
    # Anomaly detection
    baseline_mean = results["mean"]
    anomalies = values_clean - baseline_mean
    results["anomaly_count"] = np.sum(np.abs(anomalies) > 2 * results["std"])
    results["anomaly_percentage"] = (results["anomaly_count"] / len(values_clean)) * 100
    
    # Seasonal analysis if we have enough data
    if len(values_clean) >= 12:
        df = pd.DataFrame({"time": time_clean, "values": values_clean})
        df["month"] = df["time"].dt.month
        monthly_stats = df.groupby("month")["values"].agg(["mean", "std"]).reset_index()
        
        results["seasonal_amplitude"] = monthly_stats["mean"].max() - monthly_stats["mean"].min()
        results["most_variable_month"] = monthly_stats.loc[monthly_stats["std"].idxmax(), "month"]
        results["peak_month"] = monthly_stats.loc[monthly_stats["mean"].idxmax(), "month"]
        results["lowest_month"] = monthly_stats.loc[monthly_stats["mean"].idxmin(), "month"]
    
    # Persistence analysis (autocorrelation)
    if len(values_clean) > 20:
        lag1_corr = np.corrcoef(values_clean[:-1], values_clean[1:])[0, 1]
        results["lag1_autocorrelation"] = lag1_corr
        results["persistence_level"] = "High" if lag1_corr > 0.7 else "Medium" if lag1_corr > 0.3 else "Low"
    
    return results

def generate_climate_summary(data_info, trend_results, climate_metrics, variable_name, units, is_prediction=False):
    """
    Generate an LLM-style summary of climate trends and observations.
    
    Returns:
        str: Formatted climate analysis summary
    """
    summary_parts = []
    
    # Header
    summary_parts.append(f"## Climate Analysis Summary: {variable_name}")
    summary_parts.append("---")
    
    # Variable context
    var_info = SHORT_NAME_MAP.get(variable_name, {})
    long_name = var_info.get('name', variable_name)
    summary_parts.append(f"**Variable:** {long_name} ({units})")
    
    if is_prediction:
        summary_parts.append("*Note: This analysis includes predicted/forecasted values.*")
    
    summary_parts.append("")
    
    # Trend Analysis
    if "error" not in trend_results:
        summary_parts.append("### 📈 Trend Analysis")
        
        trend = trend_results["trend_per_year"]
        r_squared = trend_results["r_squared"]
        is_sig = trend_results["is_significant"]
        
        # Trend direction and magnitude
        if abs(trend) < 0.01:
            trend_desc = "stable with minimal change"
        elif trend > 0:
            trend_desc = f"increasing at {abs(trend):.3f} {units}/year"
        else:
            trend_desc = f"decreasing at {abs(trend):.3f} {units}/year"
        
        confidence = "high" if r_squared > 0.7 else "moderate" if r_squared > 0.3 else "low"
        significance = "statistically significant" if is_sig else "not statistically significant"
        
        summary_parts.append(f"The data shows a **{trend_desc}** over the analysis period. This trend has {confidence} confidence (R² = {r_squared:.3f}) and is {significance}.")
        
        # Total change
        total_change = trend_results["total_change"]
        relative_change = trend_results["relative_change"]
        summary_parts.append(f"Total change: {total_change:+.3f} {units} ({relative_change:+.1f}%)")
        
        # Extreme events
        extreme_count = trend_results["extreme_events"]
        if extreme_count > 0:
            summary_parts.append(f"⚠️ **{extreme_count} extreme events** detected (values beyond 2.5 standard deviations).")
        
        summary_parts.append("")
    
    # Climate Metrics
    if "error" not in climate_metrics:
        summary_parts.append("### 🌡️ Climate Characteristics")
        
        mean_val = climate_metrics["mean"]
        std_val = climate_metrics["std"]
        cv = climate_metrics["coefficient_of_variation"]
        
        summary_parts.append(f"**Central Tendency:** Average value of {mean_val:.3f} {units} with standard deviation of {std_val:.3f} {units}")
        
        # Variability assessment
        if cv < 10:
            variability = "low variability"
        elif cv < 30:
            variability = "moderate variability"
        else:
            variability = "high variability"
        
        summary_parts.append(f"**Variability:** The data shows {variability} (CV = {cv:.1f}%)")
        
        # Extreme values
        extreme_range = climate_metrics["max"] - climate_metrics["min"]
        summary_parts.append(f"**Range:** {climate_metrics['min']:.3f} to {climate_metrics['max']:.3f} {units} (range: {extreme_range:.3f} {units})")
        
        # Anomaly analysis
        anomaly_pct = climate_metrics["anomaly_percentage"]
        if anomaly_pct > 10:
            summary_parts.append(f"⚠️ **High anomaly activity:** {anomaly_pct:.1f}% of values are anomalous (>2σ from mean)")
        elif anomaly_pct > 5:
            summary_parts.append(f"📊 **Moderate anomaly activity:** {anomaly_pct:.1f}% of values are anomalous")
        else:
            summary_parts.append(f"✅ **Low anomaly activity:** {anomaly_pct:.1f}% of values are anomalous")
        
        # Seasonal patterns (if available)
        if "seasonal_amplitude" in climate_metrics:
            seasonal_amp = climate_metrics["seasonal_amplitude"]
            peak_month = climate_metrics["peak_month"]
            low_month = climate_metrics["lowest_month"]
            
            month_names = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            
            summary_parts.append(f"**Seasonal Pattern:** Peak in {month_names[peak_month]}, lowest in {month_names[low_month]} (amplitude: {seasonal_amp:.3f} {units})")
        
        # Persistence
        if "persistence_level" in climate_metrics:
            persistence = climate_metrics["persistence_level"]
            summary_parts.append(f"**Temporal Persistence:** {persistence} ({climate_metrics['lag1_autocorrelation']:.3f} lag-1 correlation)")
        
        summary_parts.append("")
    
    # Climate implications
    summary_parts.append("### 🌍 Climate Implications")
    
    # Variable-specific interpretations
    if variable_name in ["t2m", "skt", "sst"]:  # Temperature variables
        if "error" not in trend_results:
            trend = trend_results["trend_per_year"]
            if abs(trend) > 0.1:  # Significant temperature trend
                if trend > 0:
                    summary_parts.append("🔥 **Warming Trend:** The observed temperature increase is consistent with regional climate change patterns and could indicate local warming effects.")
                else:
                    summary_parts.append("❄️ **Cooling Trend:** The observed temperature decrease may indicate local cooling effects or natural climate variability.")
            else:
                summary_parts.append("🌡️ **Stable Temperature:** Temperature remains relatively stable, suggesting balanced climate conditions.")
    
    elif variable_name in ["sp", "msl"]:  # Pressure variables
        if "error" not in trend_results:
            trend = trend_results["trend_per_year"]
            if abs(trend) > 10:  # Significant pressure trend (Pa/year)
                summary_parts.append("🌀 **Pressure Changes:** Significant pressure trends may indicate shifts in regional weather patterns or storm tracks.")
    
    elif "wind" in variable_name or variable_name in ["u10", "v10"]:  # Wind variables
        if "error" not in climate_metrics:
            cv = climate_metrics["coefficient_of_variation"]
            if cv > 50:
                summary_parts.append("💨 **High Wind Variability:** Large wind variations suggest a dynamic atmospheric environment with frequent weather changes.")
    
    # Prediction-specific notes
    if is_prediction:
        summary_parts.append("")
        summary_parts.append("### 🔮 Forecasting Assessment")
        summary_parts.append("The analysis includes predicted values, which should be interpreted with caution:")
        summary_parts.append("- Forecast accuracy typically decreases with time")
        summary_parts.append("- Extreme events in predictions may be underestimated")
        summary_parts.append("- Long-term trends in forecasts reflect model assumptions")
    
    # Recommendations
    summary_parts.append("")
    summary_parts.append("### 💡 Recommendations")
    
    if "error" not in climate_metrics:
        anomaly_pct = climate_metrics["anomaly_percentage"]
        if anomaly_pct > 10:
            summary_parts.append("- **Monitor for extreme events:** High anomaly frequency suggests increased climate variability")
        
        if "error" not in trend_results and trend_results["is_significant"]:
            summary_parts.append("- **Consider long-term planning:** Statistically significant trends should inform future planning")
        
        if "persistence_level" in climate_metrics and climate_metrics["persistence_level"] == "High":
            summary_parts.append("- **Expect persistence:** High autocorrelation suggests current conditions may persist")
    
    summary_parts.append("- **Continuous monitoring:** Regular analysis helps track climate system evolution")
    
    return "\n".join(summary_parts)

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

    # Categorize files for better organization
    categorized_files = categorize_nc_files(all_nc_files_list, database_path)
    
    # Create hierarchical categorized options
    file_options = ["Select a file"]
    
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

    selected_report_key = st.selectbox(
        "Select a data file for the report",
        options=file_options,
        index=0,
        key="report_file_selector"
    )

    # Extract actual filename from hierarchical selection
    actual_filename = None
    if selected_report_key and selected_report_key != "Select a file":
        # Skip separator lines, headers, and subcategory labels
        if (selected_report_key.startswith("─") or 
            selected_report_key.startswith("📍 POINT FILES") or 
            selected_report_key.startswith("🗺️ SPATIAL FILES") or
            selected_report_key.strip().endswith("Actual:") or
            selected_report_key.strip().endswith("Predicted:")):
            st.info("Please select a file from the dropdown menu.")
        elif selected_report_key.startswith("      └─ "):
            actual_filename = selected_report_key.replace("      └─ ", "")

    if actual_filename:
        file_path = os.path.join(database_path, actual_filename)
        try:
            with st.spinner(f"Analyzing {os.path.basename(file_path)} and generating report..."):
                # Use context manager and load data into memory to avoid file handle issues
                try:
                    # Try h5netcdf engine first (sometimes more stable)
                    with xr.open_dataset(file_path, engine='h5netcdf') as ds_temp:
                        ds = ds_temp.load()  # Load into memory
                except:
                    # Fallback to default netcdf4 engine
                    with xr.open_dataset(file_path, engine='netcdf4') as ds_temp:
                        ds = ds_temp.load()  # Load into memory
                
                # Do NOT apply temperature conversion here - we'll do it per variable as needed
                
                time_dim = get_time_dim_name(ds)
                if not time_dim:
                    st.error("Could not generate report: A recognizable time coordinate ('time' or 'valid_time') was not found.")
                    return
                
                # --- Report Header ---
                st.markdown(f"### Analysis Report for `{os.path.basename(file_path)}`")
                
                # Show temperature unit info if applicable
                temp_vars = [var for var in ds.data_vars if any(temp_name in var for temp_name in ['t2m', 'sst', 'skt', 'stl1'])]
                if temp_vars:
                    current_temp_unit = st.session_state.get('temperature_unit', 'Celsius')
                    st.info(f"🌡️ Temperature data is displayed in **{current_temp_unit}**. Original files remain in Kelvin.")
                
                import pandas as pd
                time_start = format_datetime_with_timezone(ds[time_dim].values[0])
                time_end = format_datetime_with_timezone(ds[time_dim].values[-1])
                st.markdown(f"**Full Time Range:** `{time_start}` to `{time_end}`")
                
                # --- Time Range Selector ---
                st.markdown("#### Time Range Selection")
                st.info("📅 Select a subset of the time range for analysis. All statistics and plots will be calculated using only the selected period.")
                
                # Convert time values to datetime for the slider
                time_values = pd.to_datetime(ds[time_dim].values)
                min_time = time_values.min()
                max_time = time_values.max()
                
                # Create columns for better layout
                col1, col2 = st.columns(2)
                
                with col1:
                    selected_start = st.date_input(
                        "Start Date",
                        value=min_time.date(),
                        min_value=min_time.date(),
                        max_value=max_time.date(),
                        key=f"report_start_date_{actual_filename}"
                    )
                
                with col2:
                    selected_end = st.date_input(
                        "End Date", 
                        value=max_time.date(),
                        min_value=min_time.date(),
                        max_value=max_time.date(),
                        key=f"report_end_date_{actual_filename}"
                    )
                
                # Validate date selection
                if selected_start > selected_end:
                    st.error("⚠️ Start date must be before or equal to end date.")
                    return
                
                # Convert selected dates to pandas datetime for filtering
                selected_start_dt = pd.to_datetime(selected_start)
                selected_end_dt = pd.to_datetime(selected_end) + pd.Timedelta(hours=23, minutes=59, seconds=59)  # Include the entire end day
                
                # Filter dataset to selected time range
                time_mask = (time_values >= selected_start_dt) & (time_values <= selected_end_dt)
                
                if not time_mask.any():
                    st.error("⚠️ No data points found in the selected time range.")
                    return
                
                # Apply time filter to dataset
                ds_filtered = ds.isel({time_dim: time_mask})
                
                # Update displayed time range
                filtered_time_start = format_datetime_with_timezone(ds_filtered[time_dim].values[0])
                filtered_time_end = format_datetime_with_timezone(ds_filtered[time_dim].values[-1])
                num_timesteps = len(ds_filtered[time_dim])
                
                st.success(f"✅ **Selected Time Range:** `{filtered_time_start}` to `{filtered_time_end}` ({num_timesteps} time steps)")
                
                # Replace the original dataset with the filtered one for all subsequent analysis
                ds = ds_filtered
                
                # --- Determine File Type (Point vs. Area) ---
                is_area = 'latitude' in ds.dims and 'longitude' in ds.dims and len(ds.latitude) > 1 and len(ds.longitude) > 1

                if is_area:
                    st.success("**File Type: Area Data**")
                    lat_range = f"{ds.latitude.min().item():.2f}° to {ds.latitude.max().item():.2f}°"
                    lon_range = f"{ds.longitude.min().item():.2f}° to {ds.longitude.max().item():.2f}°"
                    st.markdown(f"**Latitude Range:** `{lat_range}`")
                    st.markdown(f"**Longitude Range:** `{lon_range}`")
                    
                    # Display grid information
                    grid_points = len(ds.latitude) * len(ds.longitude)
                    st.success(f"🌍 **Analysis Region:** Full spatial domain ({ds.latitude.min().item():.2f}°-{ds.latitude.max().item():.2f}°N, {ds.longitude.min().item():.2f}°-{ds.longitude.max().item():.2f}°E)")
                    st.info(f"📈 **Grid Resolution:** {len(ds.latitude)} × {len(ds.longitude)} points ({grid_points} total grid cells)")
                    
                    # --- Sub-Region Selection for Spatial Data ---
                    st.markdown("#### 🗺️ Geographic Sub-Region Selection")
                    st.info("Focus the analysis on a specific geographic region within the data bounds.")
                    
                    # Get full spatial bounds
                    full_lat_min = float(ds.latitude.min().item())
                    full_lat_max = float(ds.latitude.max().item())
                    full_lon_min = float(ds.longitude.min().item())
                    full_lon_max = float(ds.longitude.max().item())
                    
                    # Create a safe filename key for session state
                    safe_filename = "".join(c for c in actual_filename if c.isalnum() or c in ['_', '-'])
                    
                    # Simple checkbox to enable sub-region selection
                    use_subregion = st.checkbox(
                        "🎯 Enable Sub-Region Selection", 
                        value=False,
                        key=f"subregion_{safe_filename}",
                        help="Select a custom geographic region for analysis"
                    )
                    
                    if use_subregion:
                        st.markdown("##### Define Analysis Region")
                        
                        # Initialize session state for region bounds
                        region_key = f"region_{safe_filename}"
                        if region_key not in st.session_state:
                            st.session_state[region_key] = {
                                'north': full_lat_max,
                                'south': full_lat_min,
                                'west': full_lon_min,
                                'east': full_lon_max
                            }
                        
                        # Method selection
                        st.markdown("**🗺️ Choose Region Selection Method:**")
                        method = st.radio(
                            "Selection Method:",
                            ["📝 Manual Coordinates", "🗺️ Interactive Map"],
                            key=f"method_{safe_filename}",
                            help="Choose how to define your analysis region"
                        )
                        
                        if method == "🗺️ Interactive Map" and FOLIUM_AVAILABLE:
                            # Interactive map method
                            st.markdown("**🗺️ Interactive Map Selection:**")
                            st.info("📍 **Instructions:** Draw a rectangle on the map to select your analysis region. The region will automatically update when you finish drawing.")
                            
                            try:
                                # Create a folium map centered on the data bounds
                                center_lat = (full_lat_min + full_lat_max) / 2
                                center_lon = (full_lon_min + full_lon_max) / 2
                                
                                m = folium.Map(
                                    location=[center_lat, center_lon],
                                    zoom_start=6,
                                    tiles="OpenStreetMap"
                                )
                                
                                # Add data bounds rectangle
                                folium.Rectangle(
                                    bounds=[[full_lat_min, full_lon_min], [full_lat_max, full_lon_max]],
                                    popup="Data Coverage Area",
                                    tooltip="Available Data Region",
                                    color="blue",
                                    weight=2,
                                    fill=False
                                ).add_to(m)
                                
                                # Add current selection rectangle if exists
                                current_bounds = [
                                    [st.session_state[region_key]['south'], st.session_state[region_key]['west']],
                                    [st.session_state[region_key]['north'], st.session_state[region_key]['east']]
                                ]
                                
                                folium.Rectangle(
                                    bounds=current_bounds,
                                    popup="Current Selection",
                                    tooltip="Analysis Region",
                                    color="red",
                                    weight=3,
                                    fill=True,
                                    fillColor="red",
                                    fillOpacity=0.2
                                ).add_to(m)
                                
                                # Enable drawing tools
                                from folium import plugins
                                draw = plugins.Draw(
                                    export=True,
                                    filename='data.geojson',
                                    position='topleft',
                                    draw_options={
                                        'rectangle': {'repeatMode': False},
                                        'polygon': False,
                                        'circle': False,
                                        'marker': False,
                                        'circlemarker': False,
                                        'polyline': False
                                    },
                                    edit_options={'edit': False}
                                )
                                m.add_child(draw)
                                
                                # Display the map
                                map_data = st_folium(
                                    m, 
                                    width=700, 
                                    height=500,
                                    key=f"map_{safe_filename}",
                                    returned_objects=["last_object_clicked_tooltip", "all_drawings"]
                                )
                                
                                # Process map selections
                                if map_data['all_drawings'] and len(map_data['all_drawings']) > 0:
                                    # Get the most recent rectangle
                                    latest_drawing = map_data['all_drawings'][-1]
                                    
                                    if latest_drawing['geometry']['type'] == 'Polygon':
                                        # Extract bounds from polygon coordinates
                                        coords = latest_drawing['geometry']['coordinates'][0]
                                        lats = [coord[1] for coord in coords]
                                        lons = [coord[0] for coord in coords]
                                        
                                        selected_south = min(lats)
                                        selected_north = max(lats)
                                        selected_west = min(lons)
                                        selected_east = max(lons)
                                        
                                        # Constrain to data bounds
                                        selected_south = max(selected_south, full_lat_min)
                                        selected_north = min(selected_north, full_lat_max)
                                        selected_west = max(selected_west, full_lon_min)
                                        selected_east = min(selected_east, full_lon_max)
                                        
                                        # Update session state
                                        st.session_state[region_key] = {
                                            'north': selected_north,
                                            'south': selected_south,
                                            'west': selected_west,
                                            'east': selected_east
                                        }
                                        
                                        st.success(f"✅ **Region Selected:** {selected_south:.2f}°-{selected_north:.2f}°N, {selected_west:.2f}°-{selected_east:.2f}°E")
                                
                            except Exception as e:
                                st.error(f"⚠️ Map error: {str(e)}")
                                st.info("Please use manual coordinate input below.")
                                method = "📝 Manual Coordinates"  # Fallback to manual method
                        
                        elif method == "🗺️ Interactive Map" and not FOLIUM_AVAILABLE:
                            st.warning("⚠️ Interactive map not available. Please install folium and streamlit-folium, or use manual coordinates.")
                            method = "📝 Manual Coordinates"  # Fallback to manual method
                        
                        if method == "📝 Manual Coordinates":
                            # Simple coordinate input method (always available)
                            st.markdown("**📝 Set Region Coordinates:**")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Latitude Bounds**")
                                lat_min = st.number_input(
                                    "South Latitude (°)", 
                                    value=st.session_state[region_key]['south'],
                                    min_value=full_lat_min, 
                                    max_value=full_lat_max,
                                    format="%.4f",
                                    key=f"lat_min_{safe_filename}"
                                )
                                lat_max = st.number_input(
                                    "North Latitude (°)", 
                                    value=st.session_state[region_key]['north'],
                                    min_value=full_lat_min, 
                                    max_value=full_lat_max,
                                    format="%.4f",
                                    key=f"lat_max_{safe_filename}"
                                )
                            
                            with col2:
                                st.markdown("**Longitude Bounds**")
                                lon_min = st.number_input(
                                    "West Longitude (°)", 
                                    value=st.session_state[region_key]['west'],
                                    min_value=full_lon_min, 
                                    max_value=full_lon_max,
                                    format="%.4f",
                                    key=f"lon_min_{safe_filename}"
                                )
                                lon_max = st.number_input(
                                    "East Longitude (°)", 
                                    value=st.session_state[region_key]['east'],
                                    min_value=full_lon_min, 
                                    max_value=full_lon_max,
                                    format="%.4f",
                                    key=f"lon_max_{safe_filename}"
                                )
                            
                            # Update session state with manual inputs
                            st.session_state[region_key] = {
                                'north': lat_max,
                                'south': lat_min,
                                'west': lon_min,
                                'east': lon_max
                            }
                        
                        # Use the region bounds from session state for validation and filtering
                        lat_min = st.session_state[region_key]['south']
                        lat_max = st.session_state[region_key]['north']
                        lon_min = st.session_state[region_key]['west']
                        lon_max = st.session_state[region_key]['east']
                        
                        # Validate region selection
                        if lat_min >= lat_max or lon_min >= lon_max:
                            st.error("⚠️ Invalid region: North must be greater than South, East must be greater than West")
                        else:
                            # Apply spatial filtering (no map needed)
                            try:
                                # Handle coordinate ordering for proper slicing
                                lat_ascending = ds.latitude.values[0] < ds.latitude.values[-1]
                                lon_ascending = ds.longitude.values[0] < ds.longitude.values[-1]
                                
                                # Create appropriate slices based on coordinate ordering
                                if lat_ascending:
                                    lat_slice = slice(lat_min, lat_max)
                                else:
                                    lat_slice = slice(lat_max, lat_min)
                                
                                if lon_ascending:
                                    lon_slice = slice(lon_min, lon_max)
                                else:
                                    lon_slice = slice(lon_max, lon_min)
                                
                                # Apply spatial selection
                                ds_region = ds.sel(latitude=lat_slice, longitude=lon_slice)
                                
                                if len(ds_region.latitude) == 0 or len(ds_region.longitude) == 0:
                                    st.error("⚠️ No data points in selected region. Please adjust coordinates.")
                                else:
                                    # Update dataset for analysis
                                    ds = ds_region
                                    
                                    # Show region info
                                    actual_lat_min = float(ds.latitude.min().item())
                                    actual_lat_max = float(ds.latitude.max().item())
                                    actual_lon_min = float(ds.longitude.min().item())
                                    actual_lon_max = float(ds.longitude.max().item())
                                    
                                    st.success(f"✅ **Sub-Region Selected:** {actual_lat_min:.2f}°-{actual_lat_max:.2f}°N, {actual_lon_min:.2f}°-{actual_lon_max:.2f}°E")
                                    region_points = len(ds.latitude) * len(ds.longitude)
                                    st.info(f"📊 **Selected Grid:** {len(ds.latitude)} × {len(ds.longitude)} points ({region_points} total)")
                                    
                            except Exception as e:
                                st.error(f"⚠️ Error applying region filter: {str(e)}")
                                st.info("Using full dataset for analysis.")
                    
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
                    wind_speed = np.sqrt(ds['u10']**2 + ds['v10']**2)
                    wind_speed.attrs['long_name'] = 'Wind Speed'
                    wind_speed.attrs['units'] = 'm/s'
                    processed_vars['wind_speed'] = wind_speed
                    
                    # Remove original components from list
                    vars_in_file.remove('u10')
                    vars_in_file.remove('v10')

                # --- Variable Selection Interface ---
                vars_to_analyze = vars_in_file + list(processed_vars.keys())
                
                if len(vars_to_analyze) > 1:
                    st.markdown("#### Variable Selection")
                    st.info("📊 This file contains multiple variables. Select which variable(s) you want to analyze in the report.")
                    
                    # Create user-friendly variable options with their display names
                    var_options = []
                    var_mapping = {}
                    
                    for var_name in vars_to_analyze:
                        # Get display name for this variable
                        if var_name in processed_vars:
                            # For processed variables like wind_speed
                            display_name = processed_vars[var_name].attrs.get('long_name', var_name.title())
                        else:
                            # For original variables
                            var_info = SHORT_NAME_MAP.get(var_name, {})
                            display_name = var_info.get('name', ds[var_name].attrs.get('long_name', var_name).title())
                        
                        # Create option string
                        option_str = f"{display_name} ({var_name})"
                        var_options.append(option_str)
                        var_mapping[option_str] = var_name
                    
                    # Add "All Variables" option
                    var_options.insert(0, "🔄 All Variables (analyze everything)")
                    
                    # Multi-select for variables
                    selected_var_options = st.multiselect(
                        "Choose variables to analyze:",
                        options=var_options,
                        default=["🔄 All Variables (analyze everything)"],
                        key=f"var_selector_{actual_filename}",
                        help="Select specific variables or choose 'All Variables' to analyze everything. "
                             "For large datasets, selecting fewer variables will speed up the analysis."
                    )
                    
                    if not selected_var_options:
                        st.warning("⚠️ Please select at least one variable to analyze.")
                        return
                    
                    # Determine which variables to actually analyze
                    if "🔄 All Variables (analyze everything)" in selected_var_options:
                        final_vars_to_analyze = vars_to_analyze
                        st.success(f"✅ **Analyzing all {len(vars_to_analyze)} variables:** {', '.join(vars_to_analyze)}")
                    else:
                        final_vars_to_analyze = [var_mapping[option] for option in selected_var_options]
                        st.success(f"✅ **Analyzing {len(final_vars_to_analyze)} selected variable(s):** {', '.join(final_vars_to_analyze)}")
                else:
                    # Only one variable available, no need for selection
                    final_vars_to_analyze = vars_to_analyze
                    st.info(f"📊 **Single variable detected:** {vars_to_analyze[0]}")
                
                st.markdown("---")

                # --- Generate Report based on File Type ---
                if not is_area:
                    # ==================================
                    # --- SINGLE POINT REPORT LOGIC ---
                    # ==================================
                    st.header("Statistical Summary for Single Point")
                    
                    # Add processed vars to the dataset
                    ds_report = ds.copy() # Create a copy to add processed vars
                    for var_name, var_data in processed_vars.items():
                        ds_report[var_name] = var_data

                    for var_name in final_vars_to_analyze:
                        var_data = ds_report[var_name].squeeze()
                        
                        # Apply temperature conversion if this is a temperature variable
                        var_data = apply_temperature_conversion(var_data, var_name)
                        
                        # Use SHORT_NAME_MAP for consistent naming
                        var_info = SHORT_NAME_MAP.get(var_name, {})
                        long_name = var_info.get('name', var_data.attrs.get('long_name', var_name).title())
                        
                        # Get appropriate unit for display (temperature converted units)
                        display_unit = get_temperature_display_unit(var_name)
                        if display_unit is None:
                            units = var_info.get('units', var_data.attrs.get('units', ''))
                        else:
                            units = display_unit

                        st.markdown(f"#### {long_name}")

                        # --- Key Statistics ---
                        mean_val = var_data.mean().item()
                        median_val = var_data.median().item()
                        std_val = var_data.std().item()
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
                            fig_hist, ax_hist = plt.subplots()
                            ax_hist.hist(var_data.values, bins=20, density=True, alpha=0.7, label='Frequency')
                            ax_hist.set_title(f'Distribution of {long_name}')
                            ax_hist.set_xlabel(f'Value ({units})')
                            ax_hist.set_ylabel('Frequency')
                            ax_hist.grid(True)
                            st.pyplot(fig_hist)

                        # --- NEW: Trend and Change Analysis ---
                        with st.expander("📈 Trend and Change Analysis", expanded=True):
                            st.markdown("##### Significant Trends and Changes")
                            st.info("ℹ️ **Note:** This analysis uses linear regression to identify trends and statistical methods to assess their significance.")
                            
                            trend_results = analyze_trends_and_changes(var_data, time_dim, var_name, units)
                            
                            if "error" not in trend_results:
                                # Display trend metrics with help tooltips
                                col1, col2, col3 = st.columns(3)
                                
                                trend_per_year = trend_results["trend_per_year"]
                                with col1:
                                    trend_direction = "↗️ Increasing" if trend_per_year > 0 else "↘️ Decreasing" if trend_per_year < 0 else "➡️ Stable"
                                    st.metric(
                                        label="Trend Direction", 
                                        value=trend_direction,
                                        help="**Trend Direction:**\n\n"
                                             "• Based on linear regression slope\n"
                                             "• ↗️ Increasing: Positive slope (values rising over time)\n"
                                             "• ↘️ Decreasing: Negative slope (values falling over time)\n"
                                             "• ➡️ Stable: Near-zero slope (minimal change)"
                                    )
                                
                                with col2:
                                    st.metric(
                                        label="Trend Rate", 
                                        value=f"{trend_per_year:.4f} {units}/year",
                                        help="**Trend Rate Calculation:**\n\n"
                                             "• Linear regression slope converted to per-year rate\n"
                                             "• Formula: slope × (365.25 / days_per_timestep)\n"
                                             "• Positive = increasing trend, Negative = decreasing trend\n"
                                             "• Units: Original variable units per year (e.g., °C/year, hPa/year)"
                                    )
                                
                                with col3:
                                    significance = "✅ Significant" if trend_results["is_significant"] else "❌ Not Significant"
                                    st.metric(
                                        label="Statistical Significance", 
                                        value=significance,
                                        help="**Statistical Significance:**\n\n"
                                             "• Based on p-value from linear regression\n"
                                             "• ✅ Significant: p < 0.05 (trend is statistically reliable)\n"
                                             "• ❌ Not Significant: p ≥ 0.05 (trend may be due to chance)\n"
                                             "• Lower p-value = higher confidence in trend"
                                    )
                                
                                # Trend visualization
                                fig_trend = go.Figure()
                                
                                # Original data
                                fig_trend.add_trace(go.Scatter(
                                    x=trend_results["time_clean"],
                                    y=trend_results["values_clean"],
                                    mode='lines+markers',
                                    name='Actual Data',
                                    line=dict(color='blue', width=1),
                                    marker=dict(size=3)
                                ))
                                
                                # Trend line
                                x_numeric = (trend_results["time_clean"] - trend_results["time_clean"][0]).total_seconds()
                                trend_line = trend_results["values_clean"][0] + (x_numeric * trend_results["trend_per_year"] / (365.25 * 24 * 3600))
                                fig_trend.add_trace(go.Scatter(
                                    x=trend_results["time_clean"],
                                    y=trend_line,
                                    mode='lines',
                                    name='Trend Line',
                                    line=dict(color='red', width=2, dash='dash')
                                ))
                                
                                # Rolling average
                                if not np.isnan(trend_results["rolling_mean"]).all():
                                    fig_trend.add_trace(go.Scatter(
                                        x=trend_results["time_clean"],
                                        y=trend_results["rolling_mean"],
                                        mode='lines',
                                        name='Rolling Average',
                                        line=dict(color='orange', width=2)
                                    ))
                                
                                # Highlight extreme events
                                if len(trend_results["extreme_indices"]) > 0:
                                    extreme_times = trend_results["time_clean"][trend_results["extreme_indices"]]
                                    extreme_values = trend_results["values_clean"][trend_results["extreme_indices"]]
                                    fig_trend.add_trace(go.Scatter(
                                        x=extreme_times,
                                        y=extreme_values,
                                        mode='markers',
                                        name='Extreme Events',
                                        marker=dict(color='red', size=8, symbol='x')
                                    ))
                                
                                fig_trend.update_layout(
                                    title=f'Trend Analysis: {long_name}',
                                    xaxis_title='Time',
                                    yaxis_title=f'{long_name} ({units})',
                                    height=500,
                                    showlegend=True
                                )
                                
                                st.plotly_chart(fig_trend, use_container_width=True)
                                
                                # Summary statistics with help tooltips
                                st.markdown("**Change Summary:**")
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric(
                                        label="Total Change", 
                                        value=f"{trend_results['total_change']:+.3f} {units} ({trend_results['relative_change']:+.1f}%)",
                                        help="**Total Change Calculation:**\n\n"
                                             "• Formula: Final Value - Initial Value\n"
                                             "• Shows absolute change over entire time period\n"
                                             "• Positive = overall increase, Negative = overall decrease\n"
                                             "• Independent of trend linearity - just endpoint comparison\n"
                                             "• Percentage shows relative change from starting value"
                                    )
                                
                                with col2:
                                    st.metric(
                                        label="R² Correlation", 
                                        value=f"{trend_results['r_squared']:.3f}",
                                        help="**R² Correlation Calculation:**\n\n"
                                             "• Formula: Square of Pearson correlation coefficient\n"
                                             "• Range: 0.0 to 1.0 (0% to 100%)\n"
                                             "• Interpretation:\n"
                                             "  - R² = 0.8 means 80% of variance explained by linear trend\n"
                                             "  - Higher R² = stronger linear relationship with time\n"
                                             "  - R² < 0.5 indicates weak linear trend"
                                    )
                                
                                with col3:
                                    st.metric(
                                        label="Extreme Events", 
                                        value=f"{trend_results['extreme_events']}",
                                        help="**Extreme Events Calculation:**\n\n"
                                             "• Method: Z-score analysis with 2.5σ threshold\n"
                                             "• Formula: Count(|z-score| > 2.5)\n"
                                             "• Z-score = (value - mean) / standard_deviation\n"
                                             "• Purpose: Identifies most unusual observations\n"
                                             "• Climate Context: Heat waves, cold snaps, extreme weather\n"
                                             "• Threshold: Beyond 2.5 standard deviations from mean"
                                    )
                            else:
                                st.warning(trend_results["error"])

                        # --- NEW: Climate Research Metrics ---
                        with st.expander("🌡️ Climate Research Metrics", expanded=True):
                            st.markdown("##### Patterns, Variability, and Anomalies")
                            st.info("ℹ️ **Note:** These metrics follow WMO and IPCC standards for climate analysis and statistical robustness.")
                            
                            climate_metrics = calculate_climate_metrics(var_data, time_dim, var_name)
                            
                            if "error" not in climate_metrics:
                                # Key climate metrics with help tooltips
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric(
                                        label="Coefficient of Variation", 
                                        value=f"{climate_metrics['coefficient_of_variation']:.1f}%",
                                        help="**Coefficient of Variation (CV):**\n\n"
                                             "• Formula: (Standard Deviation / Mean) × 100\n"
                                             "• Purpose: Measures relative variability independent of units\n"
                                             "• Interpretation:\n"
                                             "  - CV < 10%: Low variability\n"
                                             "  - CV 10-30%: Moderate variability\n"
                                             "  - CV > 30%: High variability\n"
                                             "• Climate Use: Compare variability between different variables/regions"
                                    )
                                
                                with col2:
                                    st.metric(
                                        label="Anomaly Frequency", 
                                        value=f"{climate_metrics['anomaly_percentage']:.1f}%",
                                        help="**Anomaly Frequency Calculation:**\n\n"
                                             "• Method: Z-score with 2σ threshold\n"
                                             "• Formula: |(value - mean) / std| > 2\n"
                                             "• Expected: ~5% for normal distribution\n"
                                             "• Interpretation:\n"
                                             "  - > 10%: High anomaly activity\n"
                                             "  - 5-10%: Moderate anomaly activity\n"
                                             "  - < 5%: Low anomaly activity\n"
                                             "• Climate Use: Identify unusual climate events"
                                    )
                                
                                with col3:
                                    if "persistence_level" in climate_metrics:
                                        st.metric(
                                            label="Persistence Level", 
                                            value=climate_metrics["persistence_level"],
                                            help="**Persistence (Lag-1 Autocorrelation):**\n\n"
                                                 "• Formula: Correlation between consecutive time steps\n"
                                                 "• Range: -1 to +1\n"
                                                 "• Interpretation:\n"
                                                 "  - High (>0.7): Strong persistence (today predicts tomorrow)\n"
                                                 "  - Medium (0.3-0.7): Moderate persistence\n"
                                                 "  - Low (<0.3): Weak persistence (more random)\n"
                                                 "• Climate Significance: Memory in climate system"
                                        )
                                    else:
                                        st.metric(
                                            label="Data Range", 
                                            value=f"{climate_metrics['range']:.3f} {units}",
                                            help="**Data Range:**\n\n"
                                                 "• Formula: Maximum - Minimum value\n"
                                                 "• Purpose: Shows total variability span\n"
                                                 "• Climate Context: Total variation experienced\n"
                                                 "• Large range indicates high variability"
                                        )
                                
                                with col4:
                                    if "seasonal_amplitude" in climate_metrics:
                                        st.metric(
                                            label="Seasonal Amplitude", 
                                            value=f"{climate_metrics['seasonal_amplitude']:.3f} {units}",
                                            help="**Seasonal Amplitude Calculation:**\n\n"
                                                 "• Formula: Maximum monthly mean - Minimum monthly mean\n"
                                                 "• Purpose: Measures strength of seasonal cycle\n"
                                                 "• Interpretation:\n"
                                                 "  - Large amplitude: Strong seasonal variation\n"
                                                 "  - Small amplitude: Weak seasonal pattern\n"
                                                 "• Climate Context: Seasonal climate variability strength"
                                        )
                                    else:
                                        st.metric(
                                            label="IQR", 
                                            value=f"{climate_metrics['interquartile_range']:.3f} {units}",
                                            help="**Interquartile Range (IQR):**\n\n"
                                                 "• Formula: 75th percentile - 25th percentile\n"
                                                 "• Purpose: Measures middle 50% spread\n"
                                                 "• Robust to outliers unlike total range\n"
                                                 "• Climate Use: Central tendency variability"
                                        )
                                
                                # Percentile analysis
                                st.markdown("**Extreme Value Thresholds:**")
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.write(f"1st percentile: {climate_metrics['p1']:.3f} {units}")
                                with col2:
                                    st.write(f"5th percentile: {climate_metrics['p5']:.3f} {units}")
                                with col3:
                                    st.write(f"95th percentile: {climate_metrics['p95']:.3f} {units}")
                                with col4:
                                    st.write(f"99th percentile: {climate_metrics['p99']:.3f} {units}")
                                
                                # Seasonal analysis (if available)
                                if "peak_month" in climate_metrics:
                                    month_names = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                                                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                                    st.markdown("**Seasonal Patterns:**")
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.metric(
                                            label="Peak Season", 
                                            value=f"{month_names[climate_metrics['peak_month']]}",
                                            help="**Peak Season Identification:**\n\n"
                                                 "• Method: Month with highest average values\n"
                                                 "• Calculation: Monthly means across all years\n"
                                                 "• Purpose: Identify seasonal maximum timing\n"
                                                 "• Climate Applications:\n"
                                                 "  - Temperature: Warmest month\n"
                                                 "  - Precipitation: Wettest month\n"
                                                 "  - Wind: Strongest wind month"
                                        )
                                    
                                    with col2:
                                        st.metric(
                                            label="Lowest Season", 
                                            value=f"{month_names[climate_metrics['lowest_month']]}",
                                            help="**Lowest Season Identification:**\n\n"
                                                 "• Method: Month with lowest average values\n"
                                                 "• Calculation: Monthly means across all years\n"
                                                 "• Purpose: Identify seasonal minimum timing\n"
                                                 "• Climate Applications:\n"
                                                 "  - Temperature: Coldest month\n"
                                                 "  - Precipitation: Driest month\n"
                                                 "  - Wind: Calmest month"
                                        )
                                    
                                    with col3:
                                        st.metric(
                                            label="Most Variable Month", 
                                            value=f"{month_names[climate_metrics['most_variable_month']]}",
                                            help="**Most Variable Month:**\n\n"
                                                 "• Method: Month with highest standard deviation\n"
                                                 "• Calculation: Monthly variability across all years\n"
                                                 "• Purpose: Identify least predictable season\n"
                                                 "• Climate Significance:\n"
                                                 "  - Transition seasons often most variable\n"
                                                 "  - Important for climate risk assessment\n"
                                                 "  - Indicates seasonal uncertainty"
                                        )
                            else:
                                st.warning(climate_metrics["error"])
                        
                        st.markdown("---")

                else:
                    # =============================
                    # --- AREA DATA REPORT LOGIC ---
                    # =============================
                    st.header("Spatial Analysis for Area Data")

                    ds_report = ds.copy()
                    for var_name, var_data in processed_vars.items():
                        ds_report[var_name] = var_data

                    for var_name in final_vars_to_analyze:
                        var_data_area = ds_report[var_name]
                        
                        # Apply temperature conversion if this is a temperature variable
                        var_data_area = apply_temperature_conversion(var_data_area, var_name)
                        
                        # Use SHORT_NAME_MAP for consistent naming
                        var_info = SHORT_NAME_MAP.get(var_name, {})
                        long_name = var_info.get('name', var_data_area.attrs.get('long_name', var_name).title())
                        
                        # Get appropriate unit for display (temperature converted units)
                        display_unit = get_temperature_display_unit(var_name)
                        if display_unit is None:
                            units = var_info.get('units', var_data_area.attrs.get('units', ''))
                        else:
                            units = display_unit

                        st.markdown(f"#### {long_name}")

                        # --- Spatial Mean Map ---
                        with st.expander("Show Map of Mean Values", expanded=True):
                            mean_spatial = var_data_area.mean(dim=time_dim)
                            
                            # Check if we have enough spatial points for contouring
                            lat_size = len(var_data_area.latitude)
                            lon_size = len(var_data_area.longitude)
                            
                            if lat_size < 2 or lon_size < 2:
                                # Single point or very small region - show simple plot
                                st.info(f"📍 **Small Region:** {lat_size}×{lon_size} grid points. Showing point/line plot instead of contour map.")
                                
                                if lat_size == 1 and lon_size == 1:
                                    # Single point
                                    st.metric(
                                        label=f"Mean {long_name} at Point",
                                        value=f"{mean_spatial.item():.3f} {units}",
                                        help=f"Location: {var_data_area.latitude.item():.4f}°N, {var_data_area.longitude.item():.4f}°E"
                                    )
                                else:
                                    # Small region - use simple plot
                                    fig_simple, ax_simple = plt.subplots(figsize=(10, 6))
                                    if lat_size == 1:
                                        # Single latitude line
                                        mean_spatial.plot(ax=ax_simple, x='longitude', marker='o')
                                        ax_simple.set_title(f'Mean {long_name} along Longitude')
                                        ax_simple.set_xlabel('Longitude (°)')
                                    elif lon_size == 1:
                                        # Single longitude line  
                                        mean_spatial.plot(ax=ax_simple, x='latitude', marker='o')
                                        ax_simple.set_title(f'Mean {long_name} along Latitude')
                                        ax_simple.set_xlabel('Latitude (°)')
                                    else:
                                        # Small 2D region
                                        mean_spatial.plot(ax=ax_simple, cmap='viridis', cbar_kwargs={'label': f'Mean {long_name} ({units})'})
                                        ax_simple.set_title(f'Mean {long_name} Distribution')
                                    
                                    ax_simple.set_ylabel(f'{long_name} ({units})')
                                    ax_simple.grid(True, alpha=0.3)
                                    st.pyplot(fig_simple)
                            else:
                                # Large enough region for contour plotting
                                if CARTOPY_AVAILABLE:
                                    try:
                                        fig_map = plt.figure(figsize=(10, 8))
                                        ax_map = plt.axes(projection=ccrs.PlateCarree())
                                        
                                        mean_spatial.plot.contourf(ax=ax_map, transform=ccrs.PlateCarree(), cmap='viridis', cbar_kwargs={'label': f'Mean {long_name} ({units})'})
                                        ax_map.coastlines()
                                        gl = ax_map.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
                                        gl.top_labels = False
                                        gl.right_labels = False
                                        ax_map.set_title(f'Spatial Distribution of Mean {long_name}')
                                        st.pyplot(fig_map)
                                    except Exception as e:
                                        st.warning(f"⚠️ Contour plotting failed: {str(e)}. Showing simple plot instead.")
                                        fig_simple, ax_simple = plt.subplots(figsize=(10, 8))
                                        mean_spatial.plot(ax=ax_simple, cmap='viridis', cbar_kwargs={'label': f'Mean {long_name} ({units})'})
                                        ax_simple.set_title(f'Spatial Distribution of Mean {long_name}')
                                        st.pyplot(fig_simple)
                                else:
                                    st.warning("⚠️ Cartopy not available. Showing simple plot instead.")
                                    fig_simple, ax_simple = plt.subplots(figsize=(10, 8))
                                    mean_spatial.plot(ax=ax_simple, cmap='viridis', cbar_kwargs={'label': f'Mean {long_name} ({units})'})
                                    ax_simple.set_title(f'Spatial Distribution of Mean {long_name}')
                                    st.pyplot(fig_simple)

                        # --- Time Series of Spatial Average ---
                        with st.expander("Show Time Series of Spatially Averaged Values"):
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
                            
                            # Check grid size for appropriate plotting method
                            if lat_size < 2 or lon_size < 2:
                                # Small region - show statistics instead of maps
                                max_spatial = var_data_area.max(dim=time_dim)
                                min_spatial = var_data_area.min(dim=time_dim)
                                
                                if lat_size == 1 and lon_size == 1:
                                    # Single point
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric(
                                            label=f"Maximum {long_name}",
                                            value=f"{max_spatial.item():.3f} {units}"
                                        )
                                    with col2:
                                        st.metric(
                                            label=f"Minimum {long_name}",
                                            value=f"{min_spatial.item():.3f} {units}"
                                        )
                                else:
                                    # Small region - show line plots
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        fig_max, ax_max = plt.subplots(figsize=(8, 6))
                                        if lat_size == 1:
                                            max_spatial.plot(ax=ax_max, x='longitude', marker='o', color='red')
                                            ax_max.set_xlabel('Longitude (°)')
                                        else:
                                            max_spatial.plot(ax=ax_max, x='latitude', marker='o', color='red')
                                            ax_max.set_xlabel('Latitude (°)')
                                        ax_max.set_title(f'Maximum {long_name}')
                                        ax_max.set_ylabel(f'{long_name} ({units})')
                                        ax_max.grid(True, alpha=0.3)
                                        st.pyplot(fig_max)
                                    
                                    with col2:
                                        fig_min, ax_min = plt.subplots(figsize=(8, 6))
                                        if lat_size == 1:
                                            min_spatial.plot(ax=ax_min, x='longitude', marker='o', color='blue')
                                            ax_min.set_xlabel('Longitude (°)')
                                        else:
                                            min_spatial.plot(ax=ax_min, x='latitude', marker='o', color='blue')
                                            ax_min.set_xlabel('Latitude (°)')
                                        ax_min.set_title(f'Minimum {long_name}')
                                        ax_min.set_ylabel(f'{long_name} ({units})')
                                        ax_min.grid(True, alpha=0.3)
                                        st.pyplot(fig_min)
                            
                            elif CARTOPY_AVAILABLE:
                                col1, col2 = st.columns(2)

                                with col1:
                                    # Map of Maximum
                                    max_spatial = var_data_area.max(dim=time_dim)
                                    try:
                                        fig_max = plt.figure(figsize=(10, 8))
                                        ax_max = plt.axes(projection=ccrs.PlateCarree())
                                        max_spatial.plot.contourf(ax=ax_max, transform=ccrs.PlateCarree(), cmap='Reds', cbar_kwargs={'label': f'Max {long_name} ({units})'})
                                        ax_max.coastlines()
                                        gl_max = ax_max.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
                                        gl_max.top_labels = False
                                        gl_max.right_labels = False
                                        ax_max.set_title(f'Maximum Value Distribution')
                                        st.pyplot(fig_max)
                                    except Exception as e:
                                        st.warning(f"⚠️ Contour plotting failed for maximum values. Showing simple plot.")
                                        fig_max, ax_max = plt.subplots(figsize=(8, 6))
                                        max_spatial.plot(ax=ax_max, cmap='Reds', cbar_kwargs={'label': f'Max {long_name} ({units})'})
                                        ax_max.set_title(f'Maximum Value Distribution')
                                        st.pyplot(fig_max)

                                with col2:
                                    # Map of Minimum
                                    min_spatial = var_data_area.min(dim=time_dim)
                                    try:
                                        fig_min = plt.figure(figsize=(10, 8))
                                        ax_min = plt.axes(projection=ccrs.PlateCarree())
                                        min_spatial.plot.contourf(ax=ax_min, transform=ccrs.PlateCarree(), cmap='Blues_r', cbar_kwargs={'label': f'Min {long_name} ({units})'})
                                        ax_min.coastlines()
                                        gl_min = ax_min.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
                                        gl_min.top_labels = False
                                        gl_min.right_labels = False
                                        ax_min.set_title(f'Minimum Value Distribution')
                                        st.pyplot(fig_min)
                                    except Exception as e:
                                        st.warning(f"⚠️ Contour plotting failed for minimum values. Showing simple plot.")
                                        fig_min, ax_min = plt.subplots(figsize=(8, 6))
                                        min_spatial.plot(ax=ax_min, cmap='Blues_r', cbar_kwargs={'label': f'Min {long_name} ({units})'})
                                        ax_min.set_title(f'Minimum Value Distribution')
                                        st.pyplot(fig_min)
                            else:
                                # Fallback for no cartopy
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    max_spatial = var_data_area.max(dim=time_dim)
                                    fig_max, ax_max = plt.subplots(figsize=(8, 6))
                                    max_spatial.plot(ax=ax_max, cmap='Reds', cbar_kwargs={'label': f'Max {long_name} ({units})'})
                                    ax_max.set_title(f'Maximum Value Distribution')
                                    st.pyplot(fig_max)
                                
                                with col2:
                                    min_spatial = var_data_area.min(dim=time_dim)
                                    fig_min, ax_min = plt.subplots(figsize=(8, 6))
                                    min_spatial.plot(ax=ax_min, cmap='Blues_r', cbar_kwargs={'label': f'Min {long_name} ({units})'})
                                    ax_min.set_title(f'Minimum Value Distribution')
                                    st.pyplot(fig_min)

                        # --- NEW: Trend and Change Analysis (for spatial data) ---
                        with st.expander("📈 Spatial Trend and Change Analysis", expanded=True):
                            st.markdown("##### Regional Trends and Changes")
                            st.info("ℹ️ **Note:** This analysis uses spatially averaged values to reduce 3D data to a representative 1D time series for regional trend detection.")
                            
                            # Use spatially averaged time series for trend analysis
                            spatial_avg_ts = var_data_area.mean(dim=['latitude', 'longitude'])
                            trend_results = analyze_trends_and_changes(spatial_avg_ts, time_dim, var_name, units)
                            
                            if "error" not in trend_results:
                                # Display trend metrics with help tooltips
                                col1, col2, col3 = st.columns(3)
                                
                                trend_per_year = trend_results["trend_per_year"]
                                with col1:
                                    trend_direction = "↗️ Increasing" if trend_per_year > 0 else "↘️ Decreasing" if trend_per_year < 0 else "➡️ Stable"
                                    st.metric(
                                        label="Regional Trend", 
                                        value=trend_direction,
                                        help="**Regional Trend:**\n\n"
                                             "• Based on spatially averaged time series\n"
                                             "• ↗️ Increasing: Regional average rising over time\n"
                                             "• ↘️ Decreasing: Regional average falling over time\n"
                                             "• ➡️ Stable: Minimal regional change\n"
                                             "• Represents overall regional climate signal"
                                    )
                                
                                with col2:
                                    st.metric(
                                        label="Trend Rate", 
                                        value=f"{trend_per_year:.4f} {units}/year",
                                        help="**Regional Trend Rate:**\n\n"
                                             "• Same calculation as point data but on regional average\n"
                                             "• Formula: Linear regression slope converted to per-year\n"
                                             "• Represents region-wide change rate\n"
                                             "• Smooths out local variations to show regional signal"
                                    )
                                
                                with col3:
                                    significance = "✅ Significant" if trend_results["is_significant"] else "❌ Not Significant"
                                    st.metric(
                                        label="Statistical Significance", 
                                        value=significance,
                                        help="**Regional Trend Significance:**\n\n"
                                             "• P-value from regional time series regression\n"
                                             "• ✅ Significant: Regional trend is statistically reliable\n"
                                             "• ❌ Not Significant: Regional change may be random\n"
                                             "• Spatial averaging often increases trend significance"
                                    )
                                
                                # Spatial trend visualization
                                fig_spatial_trend = go.Figure()
                                
                                # Spatially averaged time series with trend
                                fig_spatial_trend.add_trace(go.Scatter(
                                    x=trend_results["time_clean"],
                                    y=trend_results["values_clean"],
                                    mode='lines+markers',
                                    name='Regional Average',
                                    line=dict(color='blue', width=2),
                                    marker=dict(size=4)
                                ))
                                
                                # Trend line
                                x_numeric = (trend_results["time_clean"] - trend_results["time_clean"][0]).total_seconds()
                                trend_line = trend_results["values_clean"][0] + (x_numeric * trend_results["trend_per_year"] / (365.25 * 24 * 3600))
                                fig_spatial_trend.add_trace(go.Scatter(
                                    x=trend_results["time_clean"],
                                    y=trend_line,
                                    mode='lines',
                                    name='Trend Line',
                                    line=dict(color='red', width=3, dash='dash')
                                ))
                                
                                fig_spatial_trend.update_layout(
                                    title=f'Regional Trend Analysis: {long_name}',
                                    xaxis_title='Time',
                                    yaxis_title=f'Regional Average {long_name} ({units})',
                                    height=500,
                                    showlegend=True
                                )
                                
                                st.plotly_chart(fig_spatial_trend, use_container_width=True)
                                
                                # Summary statistics with help tooltips
                                st.markdown("**Regional Change Summary:**")
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric(
                                        label="Total Regional Change", 
                                        value=f"{trend_results['total_change']:+.3f} {units} ({trend_results['relative_change']:+.1f}%)",
                                        help="**Total Regional Change:**\n\n"
                                             "• Same calculation as point data but for regional average\n"
                                             "• Shows overall change across the entire region\n"
                                             "• Spatial averaging reduces noise, shows clearer signal"
                                    )
                                
                                with col2:
                                    st.metric(
                                        label="Regional R² Correlation", 
                                        value=f"{trend_results['r_squared']:.3f}",
                                        help="**Regional R² Correlation:**\n\n"
                                             "• R² for spatially averaged time series\n"
                                             "• Often higher than local R² due to noise reduction\n"
                                             "• Shows how well linear trend explains regional variation"
                                    )
                                
                                with col3:
                                    st.metric(
                                        label="Regional Extreme Events", 
                                        value=f"{trend_results['extreme_events']}",
                                        help="**Regional Extreme Events:**\n\n"
                                             "• Extreme events in spatially averaged time series\n"
                                             "• Often fewer than local extremes due to averaging\n"
                                             "• Represents region-wide extreme conditions\n"
                                             "• More significant when they occur across large areas"
                                    )
                            else:
                                st.warning(trend_results["error"])

                        # --- NEW: Climate Research Metrics (for spatial data) ---
                        with st.expander("🌡️ Spatial Climate Research Metrics", expanded=True):
                            st.markdown("##### Regional Patterns, Variability, and Anomalies")
                            
                            # Use spatially averaged time series for climate metrics
                            spatial_avg_ts = var_data_area.mean(dim=['latitude', 'longitude'])
                            climate_metrics = calculate_climate_metrics(spatial_avg_ts, time_dim, var_name)
                            
                            if "error" not in climate_metrics:
                                # Key climate metrics with help tooltips
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric(
                                        label="Regional CV", 
                                        value=f"{climate_metrics['coefficient_of_variation']:.1f}%",
                                        help="**Regional Coefficient of Variation:**\n\n"
                                             "• Same as CV but for spatially averaged data\n"
                                             "• Formula: (Regional Std / Regional Mean) × 100\n"
                                             "• Purpose: Regional-scale climate variability\n"
                                             "• Interpretation: How variable is the region as a whole\n"
                                             "• Lower than local CV due to spatial averaging"
                                    )
                                
                                with col2:
                                    st.metric(
                                        label="Anomaly Frequency", 
                                        value=f"{climate_metrics['anomaly_percentage']:.1f}%",
                                        help="**Regional Anomaly Frequency:**\n\n"
                                             "• Based on spatially averaged time series\n"
                                             "• Same threshold (2σ) applied to regional data\n"
                                             "• Identifies region-wide unusual events\n"
                                             "• Often lower than local anomalies due to averaging"
                                    )
                                
                                with col3:
                                    if "persistence_level" in climate_metrics:
                                        st.metric(
                                            label="Temporal Persistence", 
                                            value=climate_metrics["persistence_level"],
                                            help="**Regional Temporal Persistence:**\n\n"
                                                 "• Autocorrelation of spatially averaged time series\n"
                                                 "• Measures regional climate memory\n"
                                                 "• Often higher than local persistence\n"
                                                 "• Important for regional climate forecasting"
                                        )
                                    else:
                                        st.metric(
                                            label="Regional Range", 
                                            value=f"{climate_metrics['range']:.3f} {units}",
                                            help="**Regional Range:**\n\n"
                                                 "• Range of spatially averaged values\n"
                                                 "• Smaller than local ranges due to averaging\n"
                                                 "• Represents region-wide variation"
                                        )
                                
                                with col4:
                                    if "seasonal_amplitude" in climate_metrics:
                                        st.metric(
                                            label="Seasonal Amplitude", 
                                            value=f"{climate_metrics['seasonal_amplitude']:.3f} {units}",
                                            help="**Regional Seasonal Amplitude:**\n\n"
                                                 "• Seasonal cycle strength for the region\n"
                                                 "• Based on spatially averaged monthly means\n"
                                                 "• Representative of regional seasonal signal"
                                        )
                                    else:
                                        st.metric(
                                            label="IQR", 
                                            value=f"{climate_metrics['interquartile_range']:.3f} {units}",
                                            help="**Regional Interquartile Range:**\n\n"
                                                 "• IQR of spatially averaged values\n"
                                                 "• Robust measure of regional variability"
                                        )
                                
                                # Spatial variability analysis with help tooltips
                                st.markdown("**Spatial Variability:**")
                                spatial_std = var_data_area.std(dim=['latitude', 'longitude']).mean().item()
                                spatial_range = (var_data_area.max(dim=['latitude', 'longitude']) - 
                                               var_data_area.min(dim=['latitude', 'longitude'])).mean().item()
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric(
                                        label="Average Spatial Std", 
                                        value=f"{spatial_std:.3f} {units}",
                                        help="**Average Spatial Standard Deviation:**\n\n"
                                             "• Standard deviation across grid points for each time step\n"
                                             "• Then averaged over all time steps\n"
                                             "• Measures typical spatial heterogeneity\n"
                                             "• Higher values = more spatially variable"
                                    )
                                with col2:
                                    st.metric(
                                        label="Average Spatial Range", 
                                        value=f"{spatial_range:.3f} {units}",
                                        help="**Average Spatial Range Calculation:**\n\n"
                                             "• Method: (Max - Min) across grid points for each time step\n"
                                             "• Formula: Mean of [Max(lat,lon) - Min(lat,lon)] over time\n"
                                             "• Purpose: Measures spatial heterogeneity\n"
                                             "• Interpretation:\n"
                                             "  - Large range: High spatial variability\n"
                                             "  - Small range: Spatially uniform conditions"
                                    )
                                
                                # Seasonal analysis (if available)
                                if "peak_month" in climate_metrics:
                                    month_names = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                                                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                                    st.markdown("**Regional Seasonal Patterns:**")
                                    st.write(f"• Peak month: {month_names[climate_metrics['peak_month']]}")
                                    st.write(f"• Lowest month: {month_names[climate_metrics['lowest_month']]}")
                                    st.write(f"• Most variable month: {month_names[climate_metrics['most_variable_month']]}")
                            else:
                                st.warning(climate_metrics["error"])
                        
                        st.markdown("---")

                # --- NEW: LLM Climate Summary Section ---
                st.markdown("---")
                st.markdown("## 🤖 Climate Analysis Summary")
                st.info("AI-generated summary of climate trends, patterns, and implications based on the analysis above.")
                
                # Check if this is a prediction file
                is_prediction_file = "prediction" in actual_filename.lower() or "predicted" in actual_filename.lower()
                
                # Generate summary for each variable analyzed
                ds_summary = ds.copy()
                for var_name, var_data in processed_vars.items():
                    ds_summary[var_name] = var_data
                
                for var_name in final_vars_to_analyze:
                    var_data_summary = ds_summary[var_name]
                    
                    # Apply temperature conversion if needed
                    var_data_summary = apply_temperature_conversion(var_data_summary, var_name)
                    
                    # Get variable info and units
                    var_info = SHORT_NAME_MAP.get(var_name, {})
                    long_name = var_info.get('name', var_data_summary.attrs.get('long_name', var_name).title())
                    display_unit = get_temperature_display_unit(var_name)
                    if display_unit is None:
                        units = var_info.get('units', var_data_summary.attrs.get('units', ''))
                    else:
                        units = display_unit
                    
                    # For spatial data, use spatially averaged time series
                    if is_area:
                        analysis_data = var_data_summary.mean(dim=['latitude', 'longitude'])
                    else:
                        analysis_data = var_data_summary.squeeze()
                    
                    # Get analysis results
                    trend_results = analyze_trends_and_changes(analysis_data, time_dim, var_name, units)
                    climate_metrics = calculate_climate_metrics(analysis_data, time_dim, var_name)
                    
                    # Store for summary generation
                    data_info = {
                        "file_type": "spatial" if is_area else "point",
                        "time_range": f"{format_datetime_with_timezone(ds[time_dim].values[0])} to {format_datetime_with_timezone(ds[time_dim].values[-1])}",
                        "n_timesteps": len(ds[time_dim])
                    }
                    
                    # Generate and display summary
                    summary_text = generate_climate_summary(
                        data_info, trend_results, climate_metrics, 
                        var_name, units, is_prediction_file
                    )
                    
                    st.markdown(summary_text)
                    st.markdown("---")

        except Exception as e:
            st.error(f"Failed to generate report for {os.path.basename(file_path)}: {e}")
            st.exception(e)
        finally:
            # Force garbage collection to clean up any remaining file handles
            gc.collect()
