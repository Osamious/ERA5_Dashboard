"""
Constants and variable mappings for the ERA5 Dashboard.
"""

# Timezone Configuration
DEFAULT_TIMEZONE_OFFSET = 3  # UTC+3 (can be modified as needed)
TIMEZONE_OPTIONS = {
    "UTC": 0,
    "UTC+1": 1,
    "UTC+2": 2,
    "UTC+3": 3,
    "UTC+4": 4,
    "UTC+5": 5,
    "UTC+6": 6,
    "UTC-1": -1,
    "UTC-2": -2,
    "UTC-3": -3,
    "UTC-4": -4,
    "UTC-5": -5,
    "UTC-6": -6,
}

VARIABLE_MAP = {
    "wind_components": {"name": "Wind Components (U & V)", "units": "m s**-1", "short_name": "wind", "description": "Downloads both U and V components of wind at 10m. Required for Quiver plots. Note: In time series plots, the U and V components are used to calculate and display a single wind speed value."},
    "2m_temperature": {"name": "2-meter Temperature", "units": "K", "short_name": "t2m", "description": "Temperature of air at 2m above the surface of land, sea or in-land waters."},
    "total_precipitation": {"name": "Total Precipitation", "units": "m", "short_name": "tp", "description": "The accumulated liquid and frozen water, including rain and snow, that falls to the Earth's surface."},
    "surface_pressure": {"name": "Surface Pressure", "units": "Pa", "short_name": "sp", "description": "The pressure (force per unit area) on the Earth's surface."},
    "10m_u_component_of_wind": {"name": "10m U-Component of Wind", "units": "m s**-1", "short_name": "u10", "description": "The eastward component of the 10m wind."},
    "10m_v_component_of_wind": {"name": "10m V-Component of Wind", "units": "m s**-1", "short_name": "v10", "description": "The northward component of the 10m wind."},
    "sea_surface_temperature": {"name": "Sea Surface Temperature", "units": "K", "short_name": "sst", "description": "The temperature of the sea surface."},
    "mean_sea_level_pressure": {"name": "Mean Sea Level Pressure", "units": "Pa", "short_name": "msl", "description": "The surface pressure reduced to a common reference level."},
    "skin_temperature": {"name": "Skin Temperature", "units": "K", "short_name": "skt", "description": "The temperature of the top of the surface."},
    "soil_temperature_level_1": {"name": "Soil Temperature Level 1", "units": "K", "short_name": "stl1", "description": "The temperature of the soil at level 1 (0-7cm)."},
    "volumetric_soil_water_layer_1": {"name": "Volumetric Soil Water Layer 1", "units": "m**3 m**-3", "short_name": "swvl1", "description": "The volume of water in soil layer 1 (0-7cm)."},
    "snow_depth": {"name": "Snow Depth", "units": "m of water equivalent", "short_name": "sd", "description": "The depth of snow on the ground."},
    "total_cloud_cover": {"name": "Total Cloud Cover", "units": "(0-1)", "short_name": "tcc", "description": "The fraction of the sky covered by cloud."},
}

# Create a reverse map from short_name to the full info
SHORT_NAME_MAP = {v.get('short_name', k): v for k, v in VARIABLE_MAP.items()}
