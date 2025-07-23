"""
Data Fetching tab for the ERA5 Dashboard.
"""

import streamlit as st
import xarray as xr
import os
import pprint

# Import cdsapi with error handling
try:
    import cdsapi
    CDSAPI_AVAILABLE = True
except ImportError:
    CDSAPI_AVAILABLE = False

from utils.constants import VARIABLE_MAP, SHORT_NAME_MAP
from utils.helpers import st_checkbox_grid

def render_data_fetching_tab():
    """Renders the Data Fetching tab content."""
    st.header("Download ERA5 Data")
    
    if not CDSAPI_AVAILABLE:
        st.error("⚠️ **CDS API not available**")
        st.info("The `cdsapi` package is required for downloading data. Please install it using: `pip install cdsapi`")
        st.info("You can still use other tabs to analyze existing data files in the database folder.")
        return
    
    st.info("Use this tab to download new ERA5 data. Downloaded files will be saved to the 'database' folder and can be visualized in the 'Visualization' tab.")
    
    # --- Download new data from ERA5 ---
    st.subheader("1. Variable Selection")
    variable_selections = st.multiselect(
        "Select variable(s)",
        list(VARIABLE_MAP.keys()),
        default=None,
        help="You can select multiple variables to download and plot."
    )
    
    # Display descriptions for all selected variables
    if variable_selections:
        with st.expander("Show selected variable descriptions"):
            for var in variable_selections:
                st.info(f"**{VARIABLE_MAP[var]['name']}**: {VARIABLE_MAP[var]['description']}")
    else:
        st.info("Select one or more variables from the dropdown above.")

    # --- Download Options for Multiple Variables ---
    download_option = "Separate files for each variable"
    if len(variable_selections) > 1:
        download_option = st.radio(
            "Download Option",
            ("Separate files for each variable", "Single file for all variables"),
            key='download_option',
            help="Choose whether to download each selected variable into its own .nc file or combine them into a single file."
        )

    # --- Date and Time Selection using Checkbox Grids ---
    # Year selection
    all_years = [str(y) for y in range(1940, 2026)] # Full ERA5 range
    years = st_checkbox_grid("Year", all_years, num_cols=8, key='years')

    # Month selection
    all_months = [f"{i:02d}" for i in range(1, 13)]
    months = st_checkbox_grid("Month", all_months, num_cols=6, key='months')

    # Day selection
    all_days = [f"{i:02d}" for i in range(1, 32)]
    days = st_checkbox_grid("Day", all_days, num_cols=7, key='days')

    # Time selection
    all_times = [f"{h:02d}:00" for h in range(24)]
    time = st_checkbox_grid("Time (UTC)", all_times, num_cols=6, key='time')    # --- Area Selection ---
    st.subheader("2. Location")

    # Let user choose between single point or area
    st.session_state.selection_mode = st.radio(
        "Choose selection type for new download",
        ("Single Point", "Area"),
        horizontal=True,
        key='selection_mode_radio'
    )    # --- Single Point Selection UI ---
    if st.session_state.selection_mode == "Single Point":
        # Import folium only when needed for map functionality
        try:
            import folium
            from streamlit_folium import st_folium
            FOLIUM_AVAILABLE = True
        except ImportError:
            FOLIUM_AVAILABLE = False
        
        if FOLIUM_AVAILABLE:
            st.markdown("**Select a single point by clicking the map or entering coordinates:**")
            
            # Create a map centered on the current coordinates
            m_point = folium.Map(location=[st.session_state.latitude, st.session_state.longitude], zoom_start=4)
            folium.Marker([st.session_state.latitude, st.session_state.longitude], popup="Selected Location").add_to(m_point)
            map_data_point = st_folium(m_point, width=700, height=300)

            # Update coordinates based on map click
            if map_data_point and map_data_point['last_clicked']:
                st.session_state.latitude = map_data_point['last_clicked']['lat']
                st.session_state.longitude = map_data_point['last_clicked']['lng']
                st.rerun()
        else:
            st.warning("⚠️ Interactive map not available. Folium package not installed.")
            st.info("You can still enter coordinates manually below.")

        # Display and allow manual editing of coordinates
        st.markdown("**Manual Coordinate Entry:**")
        col1, col2 = st.columns(2)
        with col1:
            lat = st.number_input("Latitude", value=st.session_state.latitude, format="%.4f")
        with col2:
            lon = st.number_input("Longitude", value=st.session_state.longitude, format="%.4f")

        # Update session state if manual input changes
        if lat != st.session_state.latitude or lon != st.session_state.longitude:
            st.session_state.latitude = lat
            st.session_state.longitude = lon
            st.rerun()    # --- Area Selection UI ---
    else:
        # Import folium only when needed for area selection
        import folium
        from streamlit_folium import st_folium
        from folium.plugins import Draw
        
        st.markdown("**Define a bounding box by drawing on the map or entering coordinates:**")
        st.info(
            "**Note:** To retrieve gridded data, the bounding box must be larger than the data resolution (0.25° x 0.25° for ERA5). "
            "Selections smaller than this may result in a single point file."
        )

        # Create a map to visualize the selected area
        map_center_lat = (st.session_state.area['north'] + st.session_state.area['south']) / 2
        map_center_lon = (st.session_state.area['west'] + st.session_state.area['east']) / 2
        m_area = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=6)

        Draw(export=False, draw_options={'rectangle': True, 'polyline': False, 'polygon': False, 'circle': False, 'marker': False, 'circlemarker': False}).add_to(m_area)
        folium.Rectangle(bounds=[(st.session_state.area['north'], st.session_state.area['west']), (st.session_state.area['south'], st.session_state.area['east'])], tooltip="Current Bounding Box").add_to(m_area)
        
        st.info("Use the rectangle tool on the map to draw a new bounding box.")
        map_data_area = st_folium(m_area, width=700, height=350)

        # Check if a new area was drawn
        if map_data_area and map_data_area.get("last_active_drawing"):
            coords = map_data_area["last_active_drawing"]["geometry"]["coordinates"][0]
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            new_area = {'north': max(lats), 'west': min(lons), 'south': min(lats), 'east': max(lons)}
            if new_area != st.session_state.area:
                st.session_state.area = new_area
                st.rerun()

        # Display and allow manual editing of coordinates
        st.markdown("**Manual Coordinate Entry:**")
        col1, col2 = st.columns(2)
        with col1:
            north = st.number_input("North Latitude", value=st.session_state.area['north'], format="%.4f")
            west = st.number_input("West Longitude", value=st.session_state.area['west'], format="%.4f")
        with col2:
            south = st.number_input("South Latitude", value=st.session_state.area['south'], format="%.4f")
            east = st.number_input("East Longitude", value=st.session_state.area['east'], format="%.4f")        # Update session state if manual input changes
        if (north, west, south, east) != (st.session_state.area['north'], st.session_state.area['west'], st.session_state.area['south'], st.session_state.area['east']):
            st.session_state.area = {'north': north, 'west': west, 'south': south, 'east': east}
            st.rerun()

    st.markdown("---")
    st.subheader("3. Data Resolution")

    # --- Grid Resolution Selection ---
    st.markdown("**Grid Resolution (in degrees)**")
    col1, col2 = st.columns(2)
    with col1:
        grid_res_lat = st.number_input("Latitude Resolution", min_value=0.01, max_value=2.0, value=0.25, step=0.01, format="%.2f", help="Specify the latitudinal grid spacing. ERA5 native is 0.25.")
    with col2:
        grid_res_lon = st.number_input("Longitude Resolution", min_value=0.01, max_value=2.0, value=0.25, step=0.01, format="%.2f", help="Specify the longitudinal grid spacing. ERA5 native is 0.25.")

    st.info("""
    **About Grid Resolution:**
    - **ERA5 Native Resolution:** The original ERA5 dataset has a resolution of **0.25° x 0.25°**.
    - **Requesting Higher Resolution:** You can request data on a finer grid (e.g., 0.1° x 0.1°). The CDS will **interpolate** the native data to your requested grid. This does not add new measured information but can be useful for creating smoother plots.
    - **For true higher-resolution data over land**, consider using the **ERA5-Land** dataset, which has a native resolution of 0.1° x 0.1°. This dashboard currently uses the standard ERA5 dataset.
    """)

    def download_data(filename, api_area, api_variable, grid_spec):
        c = cdsapi.Client()
        
        request_params = {
            'product_type': 'reanalysis',
            'variable': api_variable,
            'year': years,
            'month': months,
            'day': days,
            'time': time,
            'format': 'netcdf',
            'area': api_area,
        }

        # Add grid specification only for area requests
        if st.session_state.selection_mode == "Area" and grid_spec:
            request_params['grid'] = grid_spec

        pprint.pprint(request_params)  # Print the request parameters for debugging

        c.retrieve(
            'reanalysis-era5-single-levels',
            request_params,
            filename
        )

    # --- UI for Adding and Clearing Data ---
    if st.button("Download and Add to Plot"):
        if not variable_selections:
            st.error("Please select at least one variable to download.")
            st.stop()

        # Validation for date/time
        if not years or not months or not days or not time:
            st.error("At least one selection must be made for year, month, day, and time before downloading.")
            st.stop()        # Define area for API call based on mode
        if st.session_state.selection_mode == "Single Point":
            lat, lon = st.session_state.latitude, st.session_state.longitude
            api_area = [lat, lon, lat, lon]
            grid_spec = None # No grid for single point
        else:
            api_area = [
                st.session_state.area['north'], st.session_state.area['west'],
                st.session_state.area['south'], st.session_state.area['east']
            ]
            grid_spec = f"{grid_res_lat}/{grid_res_lon}"

        # --- Combined file download logic ---
        if len(variable_selections) > 1 and download_option == 'Single file for all variables':
            # 1. Collect all api_variables
            all_api_variables = []
            for var in variable_selections:
                if var == "wind_components":
                    all_api_variables.extend(['10m_u_component_of_wind', '10m_v_component_of_wind'])
                else:
                    all_api_variables.append(var)
            all_api_variables = sorted(list(set(all_api_variables))) # Use set for uniqueness, then sort for consistent filenames            # 2. Generate filename
            var_short_names_str = "_".join(sorted([VARIABLE_MAP[v]['short_name'] for v in variable_selections]))
            
            if st.session_state.selection_mode == "Single Point":
                lat, lon = st.session_state.latitude, st.session_state.longitude
                filename = f"database/era5_multi_{var_short_names_str}_{years[0]}_{months[0]}_{days[0]}_point_{lat:.2f}_{lon:.2f}.nc"
            else: # Area mode
                area = st.session_state.area
                filename = f"database/era5_multi_{var_short_names_str}_{years[0]}_{months[0]}_{days[0]}_area_{area['north']:.2f}N_{area['west']:.2f}W_{area['south']:.2f}S_{area['east']:.2f}E_{grid_res_lat}x{grid_res_lon}.nc"            # 3. Download
            if not os.path.exists(filename):
                with st.spinner(f"Downloading {len(variable_selections)} variables into single file: {filename}..."):
                    download_data(filename, api_area, all_api_variables, grid_spec)
                st.success(f"Downloaded {filename} to database folder.")
            else:
                st.info(f"File {filename} already exists in database folder.")

        # --- Separate files download logic ---
        else:
            # Loop through each selected variable and download it
            for selected_var in variable_selections:
                var_short_name = VARIABLE_MAP[selected_var]['short_name']
                  # Generate a filename based on selections
                if st.session_state.selection_mode == "Single Point":
                    lat, lon = st.session_state.latitude, st.session_state.longitude
                    filename = f"database/era5_{var_short_name}_{years[0]}_{months[0]}_{days[0]}_point_{lat:.2f}_{lon:.2f}.nc"
                else: # Area mode
                    area = st.session_state.area
                    filename = f"database/era5_{var_short_name}_{years[0]}_{months[0]}_{days[0]}_area_{area['north']:.2f}N_{area['west']:.2f}W_{area['south']:.2f}S_{area['east']:.2f}E_{grid_res_lat}x{grid_res_lon}.nc"                # Determine the API variable(s) for this specific download
                if selected_var == "wind_components":
                    api_variable = ['10m_u_component_of_wind', '10m_v_component_of_wind']
                else:
                    api_variable = selected_var

                if not os.path.exists(filename):
                    with st.spinner(f"Downloading {VARIABLE_MAP[selected_var]['name']}..."):
                        download_data(filename, api_area, api_variable, grid_spec)
                    st.success(f"Downloaded {filename} to database folder.")
                else:
                    st.info(f"File {filename} already exists in database folder.")
        st.success("Download completed! Visit the 'Visualization' tab to explore your data.")

    # --- Show API Request Code ---
    st.markdown("---")
    st.subheader("4. API Request Code")
    st.info("This is the Python code that the dashboard runs when you click 'Download'. You can adapt this for your own scripts.")

    if not variable_selections:
        st.warning("Select one or more variables to see the API request code.")
    else:
        # Determine area for API call based on mode
        if st.session_state.selection_mode == "Single Point":
            lat, lon = st.session_state.latitude, st.session_state.longitude
            api_area_display = [lat, lon, lat, lon]
            grid_spec_display = None
        else:
            api_area_display = [
                st.session_state.area['north'], st.session_state.area['west'],
                st.session_state.area['south'], st.session_state.area['east']
            ]
            grid_spec_display = f"{grid_res_lat}/{grid_res_lon}"

        # --- Logic for single file download display ---
        if len(variable_selections) > 1 and st.session_state.get('download_option') == 'Single file for all variables':
            # 1. Collect all api_variables
            all_api_variables_display = []
            for var in variable_selections:
                if var == "wind_components":
                    all_api_variables_display.extend(['10m_u_component_of_wind', '10m_v_component_of_wind'])
                else:
                    all_api_variables_display.append(var)
            all_api_variables_display = sorted(list(set(all_api_variables_display)))

            # 2. Generate filename
            var_short_names_str = "_".join(sorted([VARIABLE_MAP[v]['short_name'] for v in variable_selections]))
            year_str = years[0] if years else "YYYY"
            month_str = months[0] if months else "MM"
            day_str = days[0] if days else "DD"

            if st.session_state.selection_mode == "Single Point":
                lat, lon = st.session_state.latitude, st.session_state.longitude
                filename = f"database/era5_multi_{var_short_names_str}_{year_str}_{month_str}_{day_str}_point_{lat:.2f}_{lon:.2f}.nc"
            else: # Area mode
                area = st.session_state.area
                filename = f"database/era5_multi_{var_short_names_str}_{year_str}_{month_str}_{day_str}_area_{area['north']:.2f}N_{area['west']:.2f}W_{area['south']:.2f}S_{area['east']:.2f}E_{grid_res_lat}x{grid_res_lon}.nc"

            request_params_display = {
                'product_type': 'reanalysis',
                'variable': all_api_variables_display,
                'year': years, 'month': months, 'day': days, 'time': time,
                'format': 'netcdf', 'area': api_area_display,
            }
            if grid_spec_display:
                request_params_display['grid'] = grid_spec_display

            full_code_to_display = f"""
import cdsapi
import os
import pprint

c = cdsapi.Client()

# --- Download Parameters ---
request_params = {pprint.pformat(request_params_display)}
filename = "{filename}"
# -------------------------

if not os.path.exists(filename):
    print(f"Downloading to {{filename}}...")
    c.retrieve(
        'reanalysis-era5-single-levels',
        request_params,
        filename
    )
    print(f"Downloaded {{filename}}")
else:
    print(f"File {{filename}} already exists.")
"""
        else:
            # --- Original logic for separate file download display ---
            full_code_to_display = f"""
import cdsapi
import os
import pprint

c = cdsapi.Client()

# --- Download Parameters ---
years = {years}
months = {months}
days = {days}
time = {time}
api_area = {api_area_display}
grid_spec = {repr(grid_spec_display)} # Use repr to handle None correctly
# -------------------------
"""
            for var in variable_selections:
                var_short_name = VARIABLE_MAP[var]['short_name']
                
                # Use placeholders for filename if date parts are not selected
                year_str = years[0] if years else "YYYY"
                month_str = months[0] if months else "MM"
                day_str = days[0] if days else "DD"                # Generate a filename based on selections
                if st.session_state.selection_mode == "Single Point":
                    lat, lon = st.session_state.latitude, st.session_state.longitude
                    filename = f"database/era5_{var_short_name}_{year_str}_{month_str}_{day_str}_point_{lat:.2f}_{lon:.2f}.nc"
                else: # Area mode
                    area = st.session_state.area
                    filename = f"database/era5_{var_short_name}_{year_str}_{month_str}_{day_str}_area_{area['north']:.2f}N_{area['west']:.2f}W_{area['south']:.2f}S_{area['east']:.2f}E_{grid_res_lat}x{grid_res_lon}.nc"

                if var == "wind_components":
                    api_variable_display = ['10m_u_component_of_wind', '10m_v_component_of_wind']
                else:
                    api_variable_display = var

                request_params_display = {
                    'product_type': 'reanalysis',
                    'variable': api_variable_display,
                    'year': years,
                    'month': months,
                    'day': days,
                    'time': time,
                    'format': 'netcdf',
                    'area': api_area_display,
                }
                if grid_spec_display:
                    request_params_display['grid'] = grid_spec_display

                full_code_to_display += f"""
# --- Request for {var} ---
print(f"Requesting: {var}")
request_params_{var_short_name} = {pprint.pformat(request_params_display)}
filename_{var_short_name} = "{filename}"

if not os.path.exists(filename_{var_short_name}):
    print(f"Downloading to {{filename_{var_short_name}}}...")
    c.retrieve(
        'reanalysis-era5-single-levels',
        request_params_{var_short_name},
        filename_{var_short_name}
    )
    print(f"Downloaded {{filename_{var_short_name}}}")
else:
    print(f"File {{filename_{var_short_name}}} already exists.")
"""
        
        st.code(full_code_to_display, language='python')
