"""
Comprehensive import test to identify which module is causing dashboard crashes
"""

print("=== COMPREHENSIVE IMPORT TEST ===")
print("Testing all modules systematically...")

# Test 1: Basic imports
basic_imports = [
    ('streamlit', 'streamlit as st'),
    ('pandas', 'pandas as pd'),
    ('numpy', 'numpy as np'),
    ('xarray', 'xarray as xr'),
    ('matplotlib', 'matplotlib.pyplot as plt'),
    ('plotly', 'plotly.graph_objects as go'),
    ('folium', 'folium'),
    ('streamlit_folium', 'streamlit_folium'),
]

print("\n1. Testing basic imports...")
for name, import_str in basic_imports:
    try:
        exec(f"import {import_str}")
        print(f"✓ {name}")
    except Exception as e:
        print(f"✗ {name}: {e}")

# Test 2: Utils modules
print("\n2. Testing utils modules...")
utils_modules = [
    'utils.constants',
    'utils.helpers', 
    'utils.netcdf_utils'
]

for module in utils_modules:
    try:
        __import__(module)
        print(f"✓ {module}")
    except Exception as e:
        print(f"✗ {module}: {e}")

# Test 3: ML module
print("\n3. Testing ML module...")
try:
    import ml.models
    print("✓ ml.models")
except Exception as e:
    print(f"✗ ml.models: {e}")

# Test 4: Tab modules one by one
print("\n4. Testing tab modules...")
tab_modules = [
    'tabs.data_fetching',
    'tabs.file_inspection', 
    'tabs.prediction',
    'tabs.visualization',
    'tabs.report',
    'tabs.documentation'
]

for module in tab_modules:
    try:
        __import__(module)
        print(f"✓ {module}")
    except Exception as e:
        print(f"✗ {module}: {e}")

# Test 5: Main dashboard module
print("\n5. Testing main dashboard...")
try:
    import era5_dashboard
    print("✓ era5_dashboard")
except Exception as e:
    print(f"✗ era5_dashboard: {e}")

print("\n=== IMPORT TEST COMPLETE ===")
