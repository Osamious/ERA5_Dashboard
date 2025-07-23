"""
Debug version of ERA5 Dashboard to identify startup issues.
"""

import streamlit as st
import os

# Configure Streamlit page
st.set_page_config(
    page_title="ERA5 Data Dashboard - Debug",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("ERA5 Dashboard Debug Mode")
st.info("Testing module imports one by one to identify startup issues...")

# Test imports individually
import_results = {}

# Test 1: Basic imports
try:
    from utils.constants import TIMEZONE_OPTIONS, DEFAULT_TIMEZONE_OFFSET, TEMPERATURE_UNITS, DEFAULT_TEMPERATURE_UNIT
    import_results["utils.constants"] = "✅ SUCCESS"
    st.success("✅ Constants imported successfully")
except Exception as e:
    import_results["utils.constants"] = f"❌ ERROR: {str(e)}"
    st.error(f"❌ Constants import failed: {str(e)}")

# Test 2: Data fetching tab
try:
    from tabs.data_fetching import render_data_fetching_tab
    import_results["tabs.data_fetching"] = "✅ SUCCESS"
    st.success("✅ Data fetching tab imported successfully")
except Exception as e:
    import_results["tabs.data_fetching"] = f"❌ ERROR: {str(e)}"
    st.error(f"❌ Data fetching tab import failed: {str(e)}")

# Test 3: File inspection tab
try:
    from tabs.file_inspection import render_file_inspection_tab
    import_results["tabs.file_inspection"] = "✅ SUCCESS"
    st.success("✅ File inspection tab imported successfully")
except Exception as e:
    import_results["tabs.file_inspection"] = f"❌ ERROR: {str(e)}"
    st.error(f"❌ File inspection tab import failed: {str(e)}")

# Test 4: Visualization tab
try:
    from tabs.visualization import render_visualization_tab
    import_results["tabs.visualization"] = "✅ SUCCESS"
    st.success("✅ Visualization tab imported successfully")
except Exception as e:
    import_results["tabs.visualization"] = f"❌ ERROR: {str(e)}"
    st.error(f"❌ Visualization tab import failed: {str(e)}")

# Test 5: Report tab (likely culprit)
try:
    from tabs.report import render_report_tab
    import_results["tabs.report"] = "✅ SUCCESS"
    st.success("✅ Report tab imported successfully")
except Exception as e:
    import_results["tabs.report"] = f"❌ ERROR: {str(e)}"
    st.error(f"❌ Report tab import failed: {str(e)}")

# Test 6: Prediction tab
try:
    from tabs.prediction import render_prediction_tab
    import_results["tabs.prediction"] = "✅ SUCCESS"
    st.success("✅ Prediction tab imported successfully")
except Exception as e:
    import_results["tabs.prediction"] = f"❌ ERROR: {str(e)}"
    st.error(f"❌ Prediction tab import failed: {str(e)}")

# Summary
st.markdown("---")
st.markdown("### Import Summary")
for module, result in import_results.items():
    if "SUCCESS" in result:
        st.success(f"{module}: {result}")
    else:
        st.error(f"{module}: {result}")

st.markdown("### Next Steps")
if all("SUCCESS" in result for result in import_results.values()):
    st.info("🎉 All imports successful! The issue might be during tab rendering.")
    st.markdown("Try running the main dashboard again. If it still crashes, the issue is during tab execution, not import.")
else:
    st.warning("❗ Some imports failed. Fix these issues first:")
    for module, result in import_results.items():
        if "ERROR" in result:
            st.code(f"{module}: {result}")
