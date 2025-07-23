"""
Prediction tab for the ERA5 Dashboard.
"""

import streamlit as st
import xarray as xr
import os
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
import time
import shutil
import traceback
import uuid

from utils.constants import VARIABLE_MAP, SHORT_NAME_MAP
from utils.helpers import clear_prediction_results, get_time_dim_name, convert_time_array_to_local_timezone, format_datetime_with_timezone
from utils.netcdf_utils import save_prediction_to_netcdf
from ml.models import (
    train_and_predict_rf, 
    train_and_predict_gbr, 
    train_and_predict_convlstm_spatial
)

# Auto-tune functionality
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


def get_spatial_coord_names(dataset):
    """
    Get the correct latitude and longitude coordinate names from a spatial dataset.
    
    Args:
        dataset: xarray Dataset or DataArray
        
    Returns:
        tuple: (lat_name, lon_name) or (None, None) if not found
    """
    lat_names = ['lat', 'latitude', 'y']
    lon_names = ['lon', 'longitude', 'x']
    
    lat_name = None
    lon_name = None
    
    # Check dimensions first
    for name in lat_names:
        if name in dataset.dims:
            lat_name = name
            break
    
    for name in lon_names:
        if name in dataset.dims:
            lon_name = name
            break
    
    # If not found in dims, check coordinates
    if lat_name is None:
        for name in lat_names:
            if name in dataset.coords:
                lat_name = name
                break
    
    if lon_name is None:
        for name in lon_names:
            if name in dataset.coords:
                lon_name = name
                break
    
    return lat_name, lon_name

def save_auto_tune_profile(tune_results, model_choice, profile_name):
    """
    Save auto-tuned parameters as a profile.
    
    Args:
        tune_results: Results from auto-tune containing best_params
        model_choice: Model name (RandomForestRegressor or GradientBoostingRegressor)
        profile_name: Name for the profile
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import json
        import os
        
        # Load existing profiles
        PROFILES_FILE = "model_profiles.json"
        all_profiles = {}
        
        if os.path.exists(PROFILES_FILE):
            with open(PROFILES_FILE, 'r') as f:
                all_profiles = json.load(f)
        
        # Convert parameters to profile format
        best_params = tune_results['best_params']
        
        if model_choice == "RandomForestRegressor":
            profile_params = {
                'n_estimators_rf': best_params.get('n_estimators'),
                'max_depth_rf': best_params.get('max_depth'),
                'min_samples_split_rf': best_params.get('min_samples_split'),
                'min_samples_leaf_rf': best_params.get('min_samples_leaf'),
                'max_features_rf': best_params.get('max_features'),
                'criterion_rf': best_params.get('criterion'),
                'bootstrap_rf': best_params.get('bootstrap'),
                'oob_score_rf': best_params.get('oob_score')
            }
        else:  # GradientBoostingRegressor
            profile_params = {
                'n_estimators_gbr': best_params.get('n_estimators'),
                'max_depth_gbr': best_params.get('max_depth'),
                'min_samples_split_gbr': best_params.get('min_samples_split'),
                'min_samples_leaf_gbr': best_params.get('min_samples_leaf'),
                'learning_rate_gbr': best_params.get('learning_rate'),
                'subsample_gbr': best_params.get('subsample'),
                'loss_gbr': best_params.get('loss')
            }
        
        # Add to profiles
        if model_choice not in all_profiles:
            all_profiles[model_choice] = {}
        
        all_profiles[model_choice][profile_name] = profile_params
        
        # Save to file with backup
        if os.path.exists(PROFILES_FILE):
            import shutil
            shutil.copy2(PROFILES_FILE, f"{PROFILES_FILE}.backup")
        
        with open(PROFILES_FILE, 'w') as f:
            json.dump(all_profiles, f, indent=4)
        
        # Verify save was successful
        with open(PROFILES_FILE, 'r') as f:
            verify = json.load(f)
            return profile_name in verify.get(model_choice, {})
    
    except Exception as e:
        st.error(f"❌ Error saving profile: {str(e)}")
        import traceback
        st.error(f"🔍 Traceback: {traceback.format_exc()}")
        return False

def save_autotune_profile_robust(profile_name, tune_results, model_choice):
    """
    Robust function to save auto-tuned parameters as a profile.
    
    Args:
        profile_name: Name for the profile
        tune_results: Results from auto-tuning
        model_choice: Model type (RandomForestRegressor or GradientBoostingRegressor)
    
    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        # Input validation
        if not profile_name or not profile_name.strip():
            return False, "Profile name cannot be empty"
        
        if not tune_results or not isinstance(tune_results, dict):
            return False, "Invalid tune results provided"
        
        if 'best_params' not in tune_results:
            return False, "Best parameters not found in tune results"
        
        best_params = tune_results['best_params']
        if not best_params or not isinstance(best_params, dict):
            return False, "Best parameters are empty or invalid"
        
        # Load existing profiles
        PROFILES_FILE = "model_profiles.json"
        all_profiles = {}
        
        if os.path.exists(PROFILES_FILE):
            try:
                with open(PROFILES_FILE, 'r') as f:
                    content = f.read().strip()
                    if content:
                        all_profiles = json.loads(content)
                    else:
                        all_profiles = {}
            except (json.JSONDecodeError, Exception) as e:
                return False, f"Error reading existing profiles: {str(e)}"
        
        # Convert parameters based on model type
        if model_choice == "RandomForestRegressor":
            required_params = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 
                             'max_features', 'criterion', 'bootstrap', 'oob_score']
            profile_params = {}
            
            for param in required_params:
                if param not in best_params:
                    return False, f"Missing required parameter '{param}' for RandomForest"
                profile_params[f"{param}_rf"] = best_params[param]
                
        elif model_choice == "GradientBoostingRegressor":
            required_params = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 
                             'learning_rate', 'subsample', 'loss']
            profile_params = {}
            
            for param in required_params:
                if param not in best_params:
                    return False, f"Missing required parameter '{param}' for GradientBoosting"
                profile_params[f"{param}_gbr"] = best_params[param]
        else:
            return False, f"Unsupported model type: {model_choice}"
        
        # Ensure model section exists
        if model_choice not in all_profiles:
            all_profiles[model_choice] = {}
        
        # Check if profile name already exists
        if profile_name in all_profiles[model_choice]:
            return False, f"Profile '{profile_name}' already exists for {model_choice}. Choose a different name."
        
        # Save the profile
        all_profiles[model_choice][profile_name] = profile_params
        
        # Write to file with backup
        try:
            # Create backup if file exists
            if os.path.exists(PROFILES_FILE):
                backup_file = f"{PROFILES_FILE}.backup"
                import shutil
                shutil.copy2(PROFILES_FILE, backup_file)
            
            # Write new data
            with open(PROFILES_FILE, 'w') as f:
                json.dump(all_profiles, f, indent=4, sort_keys=True)
            
            # Verify the write was successful
            with open(PROFILES_FILE, 'r') as f:
                verify_data = json.load(f)
                if (model_choice in verify_data and 
                    profile_name in verify_data[model_choice] and
                    verify_data[model_choice][profile_name] == profile_params):
                    
                    profile_count = len(verify_data[model_choice])
                    return True, f"Profile '{profile_name}' saved successfully! You now have {profile_count} profiles for {model_choice}."
                else:
                    return False, "Profile save verification failed"
                    
        except Exception as e:
            return False, f"Error writing to file: {str(e)}"
            
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

def auto_tune_rf(file_path, target_variable, forecast_horizon, n_trials=50, timeout=300):
    """
    Auto-tune Random Forest hyperparameters using Optuna.
    
    Args:
        file_path: Path to the data file
        target_variable: Target variable name
        forecast_horizon: Number of future steps to predict
        n_trials: Number of optimization trials
        timeout: Maximum optimization time in seconds
      Returns:
        dict: Best parameters and performance metrics
    """
    from ml.models import _prepare_prediction_data
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestRegressor
    
    # Prepare data for optimization
    X, y, _, _, _, _, _, _, _ = _prepare_prediction_data(file_path, target_variable)
    
    def objective(trial):
        # Define hyperparameter search space
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'criterion': trial.suggest_categorical('criterion', ['squared_error', 'absolute_error', 'friedman_mse']),
        }
        
        # Create and evaluate model
        model = RandomForestRegressor(
            random_state=42,
            n_jobs=-1,
            **params
        )
        
        # Use cross-validation for robust evaluation
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        return -scores.mean()  # Minimize MSE
    
    # Create study and optimize
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
      # Get best parameters
    best_params = study.best_params
    best_params.update({
        'oob_score': best_params.get('bootstrap', True)  # Enable OOB if bootstrap is True
    })
    
    return {
        'best_params': best_params,
        'best_score': study.best_value,
        'n_trials': len(study.trials),
        'study': study
    }

def auto_tune_gbr(file_path, target_variable, forecast_horizon, n_trials=50, timeout=300):
    """
    Auto-tune Gradient Boosting Regressor hyperparameters using Optuna.
    
    Args:
        file_path: Path to the data file
        target_variable: Target variable name
        forecast_horizon: Number of future steps to predict
        n_trials: Number of optimization trials
        timeout: Maximum optimization time in seconds
      Returns:
        dict: Best parameters and performance metrics
    """
    from ml.models import _prepare_prediction_data
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import GradientBoostingRegressor
    
    # Prepare data for optimization
    X, y, _, _, _, _, _, _, _ = _prepare_prediction_data(file_path, target_variable)
    
    def objective(trial):
        # Define hyperparameter search space
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 2, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'loss': trial.suggest_categorical('loss', ['squared_error', 'absolute_error', 'huber']),
        }
        
        # Create and evaluate model
        model = GradientBoostingRegressor(
            random_state=42,
            **params
        )
        
        # Use cross-validation for robust evaluation
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        return -scores.mean()  # Minimize MSE
    
    # Create study and optimize
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
      # Get best parameters
    best_params = study.best_params
    # Note: All parameters including loss are now optimized, no need for manual updates
    
    return {
        'best_params': best_params,
        'best_score': study.best_value,
        'n_trials': len(study.trials),
        'study': study
    }


def render_prediction_tab():
    """Renders the Prediction tab content."""
    # Ensure plotly imports are available throughout the function
    import plotly.graph_objects as go
    import plotly.figure_factory as ff
    
    st.header("Weather Prediction")
    st.markdown("*Support for both Time Series (single-point) and Spatial (area) data*")

    with st.expander("About the Prediction Models"):
        st.markdown("""
        ### Time Series Models (for single-point data)
        
        #### Random Forest Regressor

        **How it works:**
        Random Forest is an *ensemble* learning method that operates by constructing a multitude of decision trees at training time. For a regression task like this, the final prediction is the average of the predictions from all the individual trees. This process of averaging helps to correct for the overfitting tendencies of single decision trees and generally results in better performance.

        **Best use for this project:**
        - **Capturing Complex Relationships:** Excellent for modeling the non-linear relationships between different weather variables (e.g., how temperature, pressure, and time of year collectively influence a future value).
        - **Robustness:** It is less sensitive to outliers in the data compared to other models.
        - **Feature Importance:** It naturally provides a ranking of which factors (e.g., 'hour', 'dayofyear', 'surface_pressure') were most influential in making the predictions.

        ---

        #### Gradient Boosting Regressor

        **How it works:**
        Gradient Boosting is another *ensemble* technique that builds models (typically decision trees) in a sequential, stage-wise fashion. Each new tree is trained to correct the errors made by the previously trained sequence of trees. It "boosts" the model's performance by iteratively focusing on the "hard-to-predict" cases, gradually building a single, highly accurate predictive model.

        **Best use for this project:**
        - **High Accuracy:** Often achieves state-of-the-art performance and can be more accurate than Random Forest, especially when hyperparameters are carefully tuned.
        - **Sequential Patterns:** Its sequential nature can be effective for time series data where patterns evolve over time.
        - **Flexibility:** It offers many hyperparameters to tune, allowing for fine-grained control over the model's behavior, though this also makes it more prone to overfitting if not configured correctly.

        ---

        ### Spatial Models (for area/grid data)

        #### Enhanced Convolutional LSTM (ConvLSTM) 🚀

        **🎯 BREAKTHROUGH: Anti-Damping Enhanced Version**
        
        **How it works:**
        ConvLSTM combines Convolutional Neural Networks (CNN) with Long Short-Term Memory (LSTM) networks to handle spatio-temporal data. The convolutional layers capture spatial patterns while LSTM cells learn temporal dependencies, making it ideal for forecasting spatial fields that evolve over time.

        **Enhanced Features (NEW):**
        - **🔥 Anti-Damping Technology:** Prevents forecasts from converging to flat lines
        - **📈 Multi-Step Training:** Trains on multiple forecast steps for better long-term accuracy  
        - **🌊 Trend Continuation:** Extrapolates recent trends with adaptive decay
        - **⏰ Seasonal Pattern Detection:** Automatically detects and applies daily/hourly cycles
        - **🎲 Adaptive Noise Injection:** Controlled randomness to maintain forecast variability
        - **🧠 Enhanced Normalization:** Preserves spatial correlations during training

        **Best use for this project:**
        - **Spatio-Temporal Patterns:** Specifically designed to capture both spatial correlations and temporal dynamics in climate data.
        - **Weather Forecasting:** Excellent for predicting how temperature, pressure, and wind patterns evolve across space and time.
        - **Non-linear Relationships:** Can learn complex, non-linear relationships that traditional methods like IDW cannot capture.
        - **Sequence Learning:** Naturally handles sequential data and can learn from multiple past time steps to predict future conditions.
        - **Multi-variable Support:** Can incorporate multiple meteorological variables simultaneously for more accurate predictions.
        
        ✅ **Performance Breakthrough:**
        The enhanced model solves the "damping" issue where long-term forecasts lose temporal variability. Advanced anti-damping mechanisms ensure forecasts maintain realistic patterns throughout the prediction horizon.
        """)

    # --- Prediction Setup ---
    st.subheader("1. Prediction Setup")
    database_path = "database"
    if not os.path.exists(database_path):
        st.warning("No database folder found. Please download some data first in the 'Data Fetching' tab.")
        return
        
    all_nc_files_list = sorted([f for f in os.listdir(database_path) if f.endswith('.nc')])

    if not all_nc_files_list:
        st.warning("No NetCDF files found. Please download a data file from the 'Data Fetching' tab.")
        return

    all_nc_files_dict = { f"File: {os.path.basename(f)}": f for f in all_nc_files_list }
    options = ["Select a file"] + list(all_nc_files_dict.keys())

    selected_pred_key = st.selectbox(
        "Select a data file for prediction",
        options=options,
        index=0,
        key="prediction_file_selector",
        on_change=clear_prediction_results
    )
    
    if selected_pred_key and selected_pred_key != "Select a file":
        file_name = all_nc_files_dict[selected_pred_key]
        file_path = os.path.join(database_path, file_name)
        
        # --- Variable Selection ---
        try:
            with xr.open_dataset(file_path) as ds:                # Get available variables
                available_vars = list(ds.data_vars.keys())
                
                if not available_vars:
                    st.error("No data variables found in the selected file.")
                    return
                
                # Variable selection
                st.subheader("🎯 Select Target Variable")
                target_variable = st.selectbox(
                    "Select target variable for prediction:",
                    available_vars,
                    format_func=lambda x: SHORT_NAME_MAP.get(x, {}).get('name', x),
                    key="target_variable_selector",
                    help="Choose which variable you want to predict."
                )
                
                var_info = SHORT_NAME_MAP.get(target_variable, {})
                  # Detect if this is a spatial (area) file by checking for lat/lon dimensions with multiple points
                # Check for various common coordinate dimension names
                lat_dims = ['lat', 'latitude', 'y']
                lon_dims = ['lon', 'longitude', 'x']
                
                has_lat = any(dim in ds.dims for dim in lat_dims)
                has_lon = any(dim in ds.dims for dim in lon_dims)
                
                # Find the actual lat/lon dimension names and sizes
                lat_dim = next((dim for dim in lat_dims if dim in ds.dims), None)
                lon_dim = next((dim for dim in lon_dims if dim in ds.dims), None)
                
                # Check if we have spatial data (multiple grid points)
                # Single point files (1x1 grid) should be treated as time series, not spatial
                is_spatial = False
                if has_lat and has_lon and lat_dim and lon_dim:
                    lat_size = len(ds[lat_dim])
                    lon_size = len(ds[lon_dim])
                    # Only consider it spatial if we have more than one point in at least one dimension
                    is_spatial = lat_size > 1 or lon_size > 1
                
                st.session_state['is_spatial'] = is_spatial
                
                if is_spatial:
                    spatial_info = f"🗺️ **Spatial Data Detected** - {lat_size} {lat_dim} × {lon_size} {lon_dim} grid points"
                    st.info(spatial_info)
                    
                    # Show spatial prediction info for the selected variable
                    target_name = var_info.get('name', target_variable)
                    target_units = var_info.get('units', 'units')
                    st.success(f"🎯 **Spatial Prediction Ready** - Will perform ConvLSTM spatio-temporal forecasting on **{target_name}** ({target_units})")
                else:
                    target_name = var_info.get('name', target_variable)
                    target_units = var_info.get('units', 'units')
                    if has_lat and has_lon and lat_dim and lon_dim:
                        # Single point case
                        st.info(f"📍 **Single Point Data Detected** - Location: {lat_dim}={ds[lat_dim].values.item():.3f}, {lon_dim}={ds[lon_dim].values.item():.3f}")
                    st.info(f"📈 **Time Series Data Detected** - Will forecast **{target_name}** ({target_units}) over time")

                # --- 2. Model & Hyperparameter Selection ---
                st.subheader("2. Configure Model")
                
                # Branch models by file type
                if is_spatial:
                    # Check TensorFlow availability for ConvLSTM
                    try:
                        import tensorflow as tf
                        tf_available = True
                        tf_version = tf.__version__
                    except ImportError:
                        tf_available = False
                        tf_version = "Not installed"
                    
                    if not tf_available:
                        st.error("❌ **TensorFlow Required for Spatial Prediction**")
                        st.markdown("""
                        ConvLSTM models require TensorFlow to be installed. Please install it using:
                        ```bash
                        pip install tf-nightly
                        ```
                        Then restart the application.
                        """)
                        st.info(f"**Current TensorFlow Status:** {tf_version}")
                        return
                    else:
                        st.success(f"✅ **TensorFlow Available:** {tf_version}")
                    
                    spatial_models = ["ConvLSTM"]
                    
                    # Get current target variable info for context
                    target_var_info = SHORT_NAME_MAP.get(target_variable, {})
                    target_name = target_var_info.get('name', target_variable)
                    
                    model_choice = st.selectbox(
                        f"Choose a Spatial Model for {target_name}",
                        spatial_models,
                        key="spatial_model_choice",
                        help=f"Select a spatial interpolation model for predicting {target_name} across the grid"
                    )
                else:
                    time_models = ["RandomForestRegressor", "GradientBoostingRegressor"]
                    model_choice = st.selectbox(
                        "Choose a Time Series Model",
                        time_models,
                        key="model_choice",
                        help="Select a model suitable for time series prediction"
                    )

        except Exception as e:
            st.error(f"Could not read or validate the selected file: {e}")
            return        
        # --- Auto-Tune Mode Selection (Time Series Only) ---
        if not st.session_state.get('is_spatial', False):
            use_auto_tune = st.checkbox(
                "🚀 Auto-Tune Mode (Bayesian Optimization)",
                value=False,
                help="Automatically optimize hyperparameters using Bayesian optimization. This will find the best parameters for your model."
            )
        else:
            use_auto_tune = False
            st.info("🔧 **Manual Configuration Only** - Auto-tune is not yet available for spatial models")
        
        # Clear auto-tune session state when auto-tune is disabled
        if 'previous_auto_tune_state' not in st.session_state:
            st.session_state.previous_auto_tune_state = use_auto_tune
        elif st.session_state.previous_auto_tune_state != use_auto_tune and not use_auto_tune:
            # Auto-tune was turned off, clear related session state
            keys_to_clear = ['auto_tune_session_id', 'current_tune_results', 'current_model_choice']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
        st.session_state.previous_auto_tune_state = use_auto_tune
        
        if not st.session_state.get('is_spatial', False) and not OPTUNA_AVAILABLE and use_auto_tune:
            st.error("⚠️ Auto-Tune requires the `optuna` package. Please install it with: `pip install optuna`")
            use_auto_tune = False

        if not st.session_state.get('is_spatial', False) and use_auto_tune:
            # --- Auto-Tune Configuration ---
            st.markdown("---")
            st.markdown("**🎯 Auto-Tune Configuration**")
            
            col1, col2 = st.columns(2)
            with col1:
                n_trials = st.number_input(
                    "Number of Trials", 
                    min_value=10, 
                    max_value=200, 
                    value=50,
                    help="Number of hyperparameter combinations to try. More trials = better optimization but longer time."
                )
            with col2:
                timeout = st.number_input(                    "Timeout (seconds)", 
                    min_value=60, 
                    max_value=1800, 
                    value=300,
                    help="Maximum time to spend on optimization. The process will stop after this time."
                )
            
            st.info("📊 Auto-Tune will use cross-validation to find the best hyperparameters for your selected model. This may take several minutes depending on your settings.")
        
        else:
            # --- Manual Mode: Profile Management and Hyperparameter Sliders ---
            st.markdown("---")
            st.markdown("**⚙️ Manual Hyperparameter Configuration**")
            
            # Profile Management
            PROFILES_FILE = "model_profiles.json"

            def load_profiles(model_key):
                if os.path.exists(PROFILES_FILE):
                    try:
                        with open(PROFILES_FILE, 'r') as f:
                            all_profiles = json.load(f)
                            return all_profiles.get(model_key, {})
                    except (json.JSONDecodeError, KeyError):
                        return {}
                return {}

            def save_profiles(model_key, model_profiles_to_save):
                all_profiles = {}
                if os.path.exists(PROFILES_FILE):
                    try:
                        with open(PROFILES_FILE, 'r') as f:
                            all_profiles = json.load(f)
                    except json.JSONDecodeError:
                        pass
                
                all_profiles[model_key] = model_profiles_to_save
                
                with open(PROFILES_FILE, 'w') as f:
                    json.dump(all_profiles, f, indent=4)
            
            model_profiles = load_profiles(model_choice)

            # Profile management UI in expander
            with st.expander("📂 Profile Management", expanded=False):
                st.markdown("**Load, Save, or Delete Model Parameter Profiles**")
                
                # Profile loading section
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.selectbox(
                        "Select a profile to load",
                        ["None"] + list(model_profiles.keys()),
                        key=f"profile_selector_{model_choice}",
                        help="Choose a saved profile to load hyperparameters."
                    )
                with col2:
                    if st.button("Load", use_container_width=True, key=f"load_profile_{model_choice}"):
                        selected_profile = st.session_state[f"profile_selector_{model_choice}"]
                        if selected_profile != "None" and selected_profile in model_profiles:
                            profile_data = model_profiles[selected_profile]
                            for key, value in profile_data.items():
                                st.session_state[key] = value
                            st.success(f"Loaded profile '{selected_profile}' for {model_choice}")
                            st.rerun()
                    
                with col3:
                    if st.button("Delete", use_container_width=True, type="primary", key=f"delete_profile_{model_choice}"):
                        selected_profile = st.session_state[f"profile_selector_{model_choice}"]
                        if selected_profile != "None" and selected_profile in model_profiles:
                            del model_profiles[selected_profile]
                            save_profiles(model_choice, model_profiles)
                            st.success(f"Deleted profile '{selected_profile}'")
                            st.rerun()

                # Profile saving section
                st.markdown("---")
                profile_name_input = st.text_input(
                    "Enter new profile name to save current settings",
                    key=f"new_profile_name_{model_choice}"
                )
                if st.button("Save Current Parameters", key=f"save_profile_{model_choice}"):
                    if profile_name_input:
                        current_params = {}
                        if model_choice == "RandomForestRegressor":
                            param_keys = ['n_estimators_rf', 'max_depth_rf', 'criterion_rf', 'bootstrap_rf', 
                                         'min_samples_split_rf', 'min_samples_leaf_rf', 'max_features_rf', 'oob_score_rf']
                        elif model_choice == "GradientBoostingRegressor":
                            param_keys = ['n_estimators_gbr', 'max_depth_gbr', 'learning_rate_gbr',
                                         'min_samples_split_gbr', 'min_samples_leaf_gbr', 'subsample_gbr', 'loss_gbr']
                        elif model_choice == "ConvLSTM":
                            param_keys = ['sequence_length', 'convlstm_filters', 'dropout_rate', 'epochs', 'batch_size', 'validation_split',
                                        'noise_std', 'teacher_forcing_ratio', 'multi_step_training', 'residual_connections']
                        else:
                            param_keys = []
                        
                        for key in param_keys:
                            if key in st.session_state:
                                current_params[key] = st.session_state[key]
                        
                        model_profiles[profile_name_input] = current_params
                        save_profiles(model_choice, model_profiles)
                        st.success(f"Saved profile '{profile_name_input}' for {model_choice}")
                        st.rerun()
                    else:
                        st.warning("Please enter a name for the profile.")
                
                # Show available profiles info
                if model_profiles:
                    st.markdown("---")
                    st.markdown("**Available Profiles:**")
                    profile_list = ", ".join(model_profiles.keys())
                    st.text(f"{profile_list}")

            # Hyperparameter Sliders based on model type
            if model_choice == "ConvLSTM":
                st.markdown("**ConvLSTM Hyperparameters**")
                col1, col2 = st.columns(2)
                with col1:
                    st.session_state['sequence_length'] = st.slider(
                        "Sequence Length", 3, 10,
                        value=st.session_state.get('sequence_length', 5),
                        key='slider_sequence_length',
                        help="Number of past time steps to use as input for prediction."
                    )
                    st.session_state['convlstm_filters'] = st.slider(
                        "ConvLSTM Filters", 8, 64,
                        value=st.session_state.get('convlstm_filters', 32),
                        step=8,
                        key='slider_convlstm_filters',
                        help="Number of filters in ConvLSTM layers. More filters can capture more complex patterns."
                    )
                    st.session_state['dropout_rate'] = st.slider(
                        "Dropout Rate", 0.0, 0.5,
                        value=st.session_state.get('dropout_rate', 0.1),
                        step=0.05,
                        key='slider_dropout_rate',
                        help="Dropout rate for regularization. Lower values (0.1) recommended for enhanced anti-damping model."
                    )
                with col2:
                    st.session_state['epochs'] = st.slider(
                        "Training Epochs", 10, 200,
                        value=st.session_state.get('epochs', 100),
                        step=10,
                        key='slider_epochs',
                        help="Number of training epochs. More epochs may improve accuracy but increase training time."
                    )
                    st.session_state['batch_size'] = st.selectbox(
                        "Batch Size",
                        options=[4, 8, 16, 32],
                        index=1,  # Default to 8
                        key='select_batch_size',
                        help="Number of samples processed before model weights are updated."
                    )
                    st.session_state['validation_split'] = st.slider(
                        "Validation Split", 0.1, 0.3,
                        value=st.session_state.get('validation_split', 0.2),
                        step=0.1,
                        key='slider_validation_split',
                        help="Fraction of data used for validation during training."
                    )
                
                # Advanced ConvLSTM parameters for better forecasting
                st.markdown("**Advanced Forecasting Parameters**")
                col3, col4 = st.columns(2)
                with col3:
                    st.session_state['noise_std'] = st.slider(
                        "Noise Injection", 0.0, 0.02,
                        value=st.session_state.get('noise_std', 0.005),
                        step=0.001,
                        key='slider_noise_std',
                        help="Small noise added during forecasting to prevent damping and improve robustness. Lower values recommended for enhanced model."
                    )
                    
                    st.session_state['multi_step_training'] = st.checkbox(
                        "Multi-Step Training",
                        value=st.session_state.get('multi_step_training', True),
                        key='checkbox_multi_step_training',
                        help="Train model to predict multiple steps ahead (1-6 steps) instead of just 1. Improves long-term forecasting but increases training time."
                    )
                    
                with col4:
                    st.session_state['teacher_forcing_ratio'] = st.slider(
                        "Teacher Forcing", 0.0, 1.0,
                        value=st.session_state.get('teacher_forcing_ratio', 0.5),
                        step=0.1,
                        key='slider_teacher_forcing',
                        help="Ratio of using actual vs predicted values during training. Higher values improve stability."
                    )
                    
                    st.session_state['residual_connections'] = st.checkbox(
                        "Residual Connections",
                        value=st.session_state.get('residual_connections', True),
                        key='checkbox_residual_connections',
                        help="Add skip connections to improve gradient flow and model performance. Recommended for better forecasting."
                    )
                
                with col1:
                    # Validation method selection
                    st.session_state['validation_method'] = st.selectbox(
                        "Validation Method",
                        options=["spatial_interpolation", "temporal_split", "spatiotemporal_cv"],
                        index=2,  # Default to spatio-temporal CV
                        key='select_validation_method',
                        help="Choose validation strategy: spatial interpolation (fast but optimistic), temporal split (realistic for forecasting), or spatio-temporal cross-validation (most rigorous)"
                    )
                    
                st.info("**Validation Methods:**\n"
                       "• **Spatial Interpolation**: Leave-one-out spatial interpolation (fast but overly optimistic)\n"
                       "• **Temporal Split**: Use earlier time periods for training, later for testing\n"
                       "• **Spatio-Temporal CV**: Rigorous cross-validation using both spatial and temporal hold-out")
                
                # Additional parameters for spatio-temporal CV
                if st.session_state.get('validation_method') == 'spatiotemporal_cv':
                    with st.expander("📊 Spatio-Temporal Cross-Validation Settings", expanded=True):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.session_state['cv_spatial_folds'] = st.slider(
                                "Spatial Folds", 2, 10,
                                value=st.session_state.get('cv_spatial_folds', 5),
                                key='slider_cv_spatial_folds',
                                help="Number of spatial regions to create for cross-validation"
                            )
                            st.session_state['cv_temporal_folds'] = st.slider(
                                "Temporal Folds", 2, 10,
                                value=st.session_state.get('cv_temporal_folds', 5),
                                key='slider_cv_temporal_folds',
                                help="Number of temporal periods to create for cross-validation"
                            )
                        with col2:
                            st.session_state['cv_method'] = st.selectbox(
                                "CV Strategy",
                                options=["blocked", "random"],
                                index=0,
                                key='select_cv_method',
                                help="Blocked: contiguous regions/periods, Random: scattered points"
                            )
                            st.session_state['cv_buffer_distance'] = st.number_input(
                                "Spatial Buffer (degrees)", 
                                min_value=0.0, 
                                max_value=5.0,
                                value=st.session_state.get('cv_buffer_distance', 0.5),
                                step=0.1,
                                key='input_cv_buffer',
                                help="Buffer distance around test regions to avoid spatial leakage"
                            )
                
                elif st.session_state.get('validation_method') == 'temporal_split':
                    with st.expander("⏰ Temporal Split Settings", expanded=True):
                        st.session_state['temporal_split_ratio'] = st.slider(
                            "Training Split Ratio", 0.5, 0.9,
                            value=st.session_state.get('temporal_split_ratio', 0.8),
                            step=0.05,
                            key='slider_temporal_split',
                            help="Fraction of time steps to use for training (rest for testing)"
                        )
            elif model_choice == "RandomForestRegressor":
                st.markdown("**Random Forest Regressor Hyperparameters**")
                col1, col2 = st.columns(2)
                with col1:
                    st.session_state['n_estimators_rf'] = st.slider(
                        "Number of Trees", 10, 500,
                        value=st.session_state.get('n_estimators_rf', 100),
                        key='slider_rf_n_estimators',
                        help="The number of trees in the forest. More trees can improve performance but also increase computation time."
                    )
                    st.session_state['max_depth_rf'] = st.slider(
                        "Max Depth", 1, 50,
                        value=st.session_state.get('max_depth_rf', 10),
                        key='slider_rf_max_depth',
                        help="The maximum depth of the trees. Deeper trees can model more complex patterns but may overfit."
                    )
                    st.session_state['min_samples_split_rf'] = st.slider(
                        "Min Samples Split", 2, 20,
                        value=st.session_state.get('min_samples_split_rf', 2),
                        key='slider_rf_min_samples_split',
                        help="The minimum number of samples required to split an internal node."
                    )
                    st.session_state['min_samples_leaf_rf'] = st.slider(
                        "Min Samples Leaf", 1, 50,
                        value=st.session_state.get('min_samples_leaf_rf', 1),
                        key='slider_rf_min_samples_leaf',
                        help="The minimum number of samples required to be at a leaf node."
                    )
                with col2:
                    st.session_state['max_features_rf'] = st.selectbox(
                        "Max Features", ['sqrt', 'log2', None],
                        index=0,
                        key='select_rf_max_features',
                        help="The number of features to consider when looking for the best split."
                    )
                    st.session_state['criterion_rf'] = st.selectbox(
                        "Criterion", ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
                        index=0,
                        key='select_rf_criterion',
                        help="The function to measure the quality of a split."
                    )
                    st.session_state['bootstrap_rf'] = st.checkbox(
                        "Bootstrap",
                        value=st.session_state.get('bootstrap_rf', True),
                        key='check_rf_bootstrap',
                        help="Whether bootstrap samples are used when building trees."
                    )
                    st.session_state['oob_score_rf'] = st.checkbox(
                        "OOB Score",
                        value=st.session_state.get('oob_score_rf', False),
                        key='check_rf_oob_score',
                        help="Whether to use out-of-bag samples to estimate the R^2 on unseen data. Only available if bootstrap=True."
                    )

            elif model_choice == "GradientBoostingRegressor":
                st.markdown("**Gradient Boosting Regressor Hyperparameters**")
                col1, col2 = st.columns(2)
                with col1:
                    st.session_state['n_estimators_gbr'] = st.slider(
                        "Number of Estimators", 10, 1000,
                        value=st.session_state.get('n_estimators_gbr', 100),
                        key='slider_gb_n_estimators',
                        help="The number of boosting stages to perform. More estimators can lead to overfitting."
                    )
                    st.session_state['max_depth_gbr'] = st.slider(
                        "Max Depth", 1, 50,
                        value=st.session_state.get('max_depth_gbr', 3),
                        key='slider_gb_max_depth',
                        help="Maximum depth of the individual regression estimators."
                    )
                    st.session_state['min_samples_split_gbr'] = st.slider(
                        "Min Samples Split", 2, 20,
                        value=st.session_state.get('min_samples_split_gbr', 2),
                        key='slider_gb_min_samples_split',
                        help="The minimum number of samples required to split an internal node."
                    )
                    st.session_state['min_samples_leaf_gbr'] = st.slider(
                        "Min Samples Leaf", 1, 20,
                        value=st.session_state.get('min_samples_leaf_gbr', 1),
                        key='slider_gb_min_samples_leaf',
                        help="The minimum number of samples required to be at a leaf node."
                    )
                with col2:
                    st.session_state['learning_rate_gbr'] = st.number_input(
                        "Learning Rate", 0.01, 1.0,
                        value=st.session_state.get('learning_rate_gbr', 0.1),
                        step=0.01, format="%.2f",
                        key='input_gb_learning_rate',
                        help="Shrinks the contribution of each tree. There is a trade-off between learning_rate and n_estimators."
                    )
                    st.session_state['subsample_gbr'] = st.slider(
                        "Subsample", 0.1, 1.0,
                        value=st.session_state.get('subsample_gbr', 1.0),
                        step=0.1,
                        key='slider_gb_subsample',
                        help="The fraction of samples to be used for fitting the individual base learners. If smaller than 1.0 this results in Stochastic Gradient Boosting."
                    )
                    st.session_state['loss_gbr'] = st.selectbox(
                        "Loss Function", ['squared_error', 'absolute_error', 'huber', 'quantile'],
                        index=0,
                        key='select_gb_loss',
                        help="The loss function to be optimized."                    )

        # --- 3. Training & Evaluation ---
        st.subheader("3. Train and Evaluate Model")
        
        # Smart forecast horizon suggestion based on available time steps
        with xr.open_dataset(file_path) as ds:
            time_dim = get_time_dim_name(ds)
            time_size = len(ds[time_dim]) if time_dim else 0
            
        max_horizon = max(1, time_size - 10)  # Leave enough for time series training
        default_horizon = min(24, max_horizon // 4)  # Conservative default
        
        st.session_state['forecast_horizon'] = st.number_input(
            f"Forecast Horizon (number of time steps to predict)",
            min_value=1,            max_value=max_horizon,
            value=min(st.session_state.get('forecast_horizon', default_horizon), max_horizon),
            help=f"Available time steps: {time_size}. Maximum recommended: {max_horizon}"
        )

        if use_auto_tune:
            button_text = "🚀 Start Auto-Tune & Generate Predictions"
            button_help = "This will automatically find the best hyperparameters and then generate predictions."
        else:
            button_text = "Generate Predictions"
            button_help = "Generate predictions using the current hyperparameter settings."
        
        if st.button(button_text, key="generate_predictions_button", help=button_help):
            try:                
                # Spatial prediction branch
                if st.session_state.get('is_spatial', False) and model_choice == "ConvLSTM":                    
                    with st.spinner(f"Training ConvLSTM model for spatial-temporal prediction..."):
                        # Prepare ConvLSTM parameters
                        filters = [st.session_state.get('convlstm_filters', 32), 16]  # Two-layer architecture
                        
                        try:
                            results = train_and_predict_convlstm_spatial(
                                file_path, target_variable,
                                sequence_length=st.session_state.get('sequence_length', 5),
                                forecast_horizon=st.session_state.get('forecast_horizon', 24),
                                filters=filters,
                                kernel_size=(3, 3),
                                dropout_rate=st.session_state.get('dropout_rate', 0.1),
                                epochs=st.session_state.get('epochs', 100),
                                batch_size=st.session_state.get('batch_size', 8),
                                validation_split=st.session_state.get('validation_split', 0.2),
                                teacher_forcing_ratio=st.session_state.get('teacher_forcing_ratio', 0.5),
                                noise_std=st.session_state.get('noise_std', 0.005),
                                multi_step_training=st.session_state.get('multi_step_training', True),
                                residual_connections=st.session_state.get('residual_connections', True)
                            )
                            
                            # Check if results contain error
                            if 'error' in results:
                                st.error(f"❌ ConvLSTM Error: {results['error']}")
                                if 'TensorFlow not available' in results['error']:
                                    st.info("💡 **Solution**: Please install TensorFlow by running:\n```\npip install tf-nightly\n```")
                                return
                            
                            # Validate results structure
                            required_keys = ['rmse', 'mae', 'r2', 'model_name']
                            missing_keys = [key for key in required_keys if key not in results]
                            if missing_keys:
                                st.error(f"❌ Incomplete results. Missing keys: {missing_keys}")
                                return
                            
                            results['model_name'] = "ConvLSTM"
                            results['validation_method'] = 'temporal_split'
                            results['auto_tuned'] = False
                            st.session_state.prediction_results = results
                            st.success("✅ ConvLSTM prediction completed successfully!")
                            
                        except Exception as e:
                            st.error(f"❌ Error during ConvLSTM training: {str(e)}")
                            st.write("**Error details:**")
                            st.code(str(e))
                            import traceback
                            st.write("**Full traceback:**")
                            st.code(traceback.format_exc())
                            return
                elif use_auto_tune:
                    # Auto-Tune Mode
                    # Clear any previous auto-tune session when starting a new one
                    if 'auto_tune_session_id' in st.session_state:
                        st.info("🔄 Starting new auto-tune session. Previous session data will be cleared.")
                        del st.session_state.auto_tune_session_id
                    if 'current_tune_results' in st.session_state:
                        del st.session_state.current_tune_results
                    if 'current_model_choice' in st.session_state:
                        del st.session_state.current_model_choice
                    if 'autotune_profile_section_key' in st.session_state:
                        del st.session_state.autotune_profile_section_key
                    
                    with st.spinner("🔍 Running Auto-Tune optimization... This may take several minutes."):
                        st.info("🎯 Searching for optimal hyperparameters using Bayesian optimization...")
                        
                        if model_choice == "RandomForestRegressor":
                            tune_results = auto_tune_rf(file_path, target_variable, st.session_state.forecast_horizon, n_trials, timeout)
                        else:  # GradientBoostingRegressor
                            tune_results = auto_tune_gbr(file_path, target_variable, st.session_state.forecast_horizon, n_trials, timeout)
                        
                        # Display optimization results
                        st.success(f"✅ Auto-Tune completed! Found optimal parameters after {tune_results['n_trials']} trials.")
                        
                        with st.expander("🎯 Optimization Results", expanded=True):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Best Score (MSE)", f"{tune_results['best_score']:.6f}")
                                st.metric("Trials Completed", tune_results['n_trials'])
                            with col2:
                                st.markdown("**📋 Best Parameters:**")
                                for param, value in tune_results['best_params'].items():
                                    st.text(f"• {param}: {value}")
                        
                        # Success messages right after optimization results
                        st.success("✅ Model trained with automatically optimized hyperparameters!")
                        st.info("💡 **Tip:** You can save these optimized parameters as a profile below for future use in Manual Mode.")
                        
                        # Store tune_results in session state for persistent access
                        st.session_state.current_tune_results = tune_results
                        st.session_state.current_model_choice = model_choice
                        st.session_state.auto_tune_completed = True  # Flag to show save section
                    
                    # Train with optimized parameters
                    with st.spinner("🏋️ Training final model with optimized parameters..."):
                        if model_choice == "RandomForestRegressor":
                            results = train_and_predict_rf(
                                file_path, target_variable,
                                tune_results['best_params']['n_estimators'],
                                tune_results['best_params']['max_depth'],
                                tune_results['best_params']['min_samples_split'],
                                tune_results['best_params']['min_samples_leaf'],
                                st.session_state.forecast_horizon,
                                tune_results['best_params']['max_features'],
                                tune_results['best_params']['criterion'],
                                tune_results['best_params']['bootstrap'],
                                tune_results['best_params']['oob_score']
                            )
                        else:  # GradientBoostingRegressor
                            results = train_and_predict_gbr(
                                file_path, target_variable,
                                tune_results['best_params']['n_estimators'],
                                tune_results['best_params']['max_depth'],
                                tune_results['best_params']['min_samples_split'],
                                tune_results['best_params']['min_samples_leaf'],
                                tune_results['best_params']['learning_rate'],
                                tune_results['best_params']['subsample'],
                                tune_results['best_params']['loss'],
                                st.session_state.forecast_horizon
                            )
                          # Add auto-tune info to results
                        results['auto_tuned'] = True
                        results['auto_tune_results'] = tune_results
                        st.session_state.prediction_results = results
                
                else:
                    # Manual Mode
                    with st.spinner("Preparing data, training model, and making predictions... (This may take a moment)"):
                        # Time series model training
                        if model_choice == "RandomForestRegressor":
                            results = train_and_predict_rf(
                                file_path, target_variable,
                                st.session_state.n_estimators_rf, st.session_state.max_depth_rf,
                                st.session_state.min_samples_split_rf, st.session_state.min_samples_leaf_rf,
                                st.session_state.forecast_horizon, st.session_state.max_features_rf,
                                st.session_state.criterion_rf, st.session_state.bootstrap_rf,
                                st.session_state.oob_score_rf
                            )
                        else: # GradientBoostingRegressor
                                results = train_and_predict_gbr(
                                    file_path, target_variable,
                                    st.session_state.n_estimators_gbr, st.session_state.max_depth_gbr,
                                    st.session_state.min_samples_split_gbr, st.session_state.min_samples_leaf_gbr,
                                    st.session_state.learning_rate_gbr, st.session_state.subsample_gbr,
                                    st.session_state.loss_gbr, st.session_state.forecast_horizon
                                )
                        
                        # Mark as manually tuned
                        results['auto_tuned'] = False
                        st.session_state.prediction_results = results
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.exception(e)
                st.session_state.prediction_results = None
        
        # --- PERSISTENT PROFILE SAVING SECTION ---
        # This section persists across Streamlit reruns and shows whenever auto-tune has been completed
        if st.session_state.get('auto_tune_completed', False) and st.session_state.get('current_tune_results'):
            st.markdown("---")
            st.markdown("**💾 Save Optimized Parameters as Profile**")
            
            tune_results = st.session_state.current_tune_results
            model_choice = st.session_state.current_model_choice
            
            # Debug information
            with st.expander("🔍 Debug Information", expanded=False):
                st.write(f"- Model choice: {model_choice}")
                st.write(f"- Tune results available: {tune_results is not None}")
                if tune_results:
                    st.write(f"- Best params: {tune_results.get('best_params', {})}")
              # Profile saving form
            with st.form("save_profile_form"):
                profile_name = st.text_input(
                    "Profile name for optimized parameters:",
                    value=f"AutoTune_{model_choice}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}",
                    help="Enter a descriptive name for this optimized parameter set"
                )
                
                save_submitted = st.form_submit_button("💾 Save Profile")
            
            # Handle form submission outside the form
            if save_submitted:
                if not profile_name.strip():
                    st.error("❌ Please enter a profile name")
                else:
                    # Save profile using a reliable helper function
                    success = save_auto_tune_profile(tune_results, model_choice, profile_name.strip())
                    if success:
                        st.success(f"✅ Profile '{profile_name}' saved successfully!")
                        st.info("💡 You can now use this profile in Manual Mode.")
                    else:
                        st.error("❌ Failed to save profile. Check the debug information above.")
            
            # Clear auto-tune session button (outside the form)
            if st.button("🧹 Clear Auto-Tune Session", key="clear_autotune", help="Clear the auto-tune results and hide this section"):
                st.session_state.auto_tune_completed = False
                st.session_state.current_tune_results = None
                st.session_state.current_model_choice = None
                st.rerun()        # --- 4. Display Results (if they exist in session state) ---
        if st.session_state.prediction_results:
            results = st.session_state.prediction_results
            st.subheader("4. Prediction Results")
            
            # Check if we have spatial prediction results (from ConvLSTM on spatial data)
            has_spatial_results = (
                st.session_state.get('is_spatial', False) and 
                model_choice == "ConvLSTM" and
                'actual_spatial_all_times' in results and 
                results['actual_spatial_all_times'] is not None
            )
            
            if has_spatial_results:
                # --- Spatial results visualization ---
                st.markdown("**Spatial Model Performance**")
                
                # Display validation method warning
                validation_method = results.get('validation_method', 'spatial_interpolation')
                if validation_method == 'spatial_interpolation':
                    st.warning("⚠️ **Validation Method: Spatial Interpolation (Leave-one-out)**\n\n"
                             "This method uses spatial interpolation for validation, which can be overly optimistic. "
                             "It tests the model's ability to interpolate between known points but does not test "
                             "true forecasting skill. Consider using temporal split or spatio-temporal CV for more realistic validation.")
                elif validation_method == 'temporal_split':
                    st.info("📅 **Validation Method: Temporal Split**\n\n"
                           "This method uses earlier time periods for training and later periods for testing. "
                           "This provides a more realistic assessment of forecasting performance.")
                elif validation_method == 'spatiotemporal_cv':
                    st.success("🎯 **Validation Method: Spatio-Temporal Cross-Validation**\n\n"
                              "This method uses rigorous cross-validation with both spatial and temporal hold-out. "
                              "This provides the most robust assessment of model performance for both interpolation and forecasting.")
                
                # Display metrics if available
                if 'rmse' in results:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("RMSE", f"{results['rmse']:.3f}")
                    col2.metric("MAE", f"{results['mae']:.3f}")
                    col3.metric("R-squared (R²)", f"{results['r2']:.3f}")
                
                # Display additional spatio-temporal CV metrics if available
                if validation_method == 'spatiotemporal_cv' and 'cv_std_rmse' in results:
                    st.markdown("**Cross-Validation Performance (Mean ± Std)**")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("CV RMSE", f"{results['rmse']:.3f} ± {results['cv_std_rmse']:.3f}")
                    col2.metric("CV MAE", f"{results['mae']:.3f} ± {results['cv_std_mae']:.3f}")
                    col3.metric("CV R²", f"{results['r2']:.3f} ± {results['cv_std_r2']:.3f}")
                    col4.metric("Validation Coverage", f"{results.get('validation_coverage', 0):.1%}")
                    
                    # Show CV fold details
                    with st.expander("📊 Cross-Validation Fold Details", expanded=False):
                        if 'cv_results' in results and results['cv_results']:
                            cv_df = pd.DataFrame(results['cv_results'])
                            st.dataframe(cv_df.round(4), use_container_width=True)
                            
                            # Plot CV performance across folds
                            import plotly.graph_objects as go
                            fig_cv = go.Figure()
                            fig_cv.add_trace(go.Scatter(
                                x=cv_df.index + 1,
                                y=cv_df['rmse'],
                                mode='lines+markers',
                                name='RMSE',
                                line=dict(color='red')
                            ))
                            fig_cv.add_trace(go.Scatter(
                                x=cv_df.index + 1,
                                y=cv_df['r2'],
                                mode='lines+markers',
                                name='R²',
                                yaxis='y2',
                                line=dict(color='blue')
                            ))
                            
                            fig_cv.update_layout(
                                title="Cross-Validation Performance Across Folds",
                                xaxis_title="Fold Number",
                                yaxis=dict(title="RMSE", side='left'),
                                yaxis2=dict(title="R²", side='right', overlaying='y'),
                                height=400
                            )
                            st.plotly_chart(fig_cv, use_container_width=True)
                        else:
                            st.info("CV fold details not available.")
                
                # Get variable info for proper labeling
                variable_info = SHORT_NAME_MAP.get(results['target_variable'], {})
                variable_full_name = variable_info.get('name', results['target_variable'])
                variable_units = variable_info.get('units', 'units')                  # Side-by-side comparison maps with time slider
                st.subheader("Spatial Field Comparison")
                
                # Get the spatial data arrays (now with all time slices)
                actual_all_times = results.get('actual_spatial_all_times')  # xarray DataArray with time dimension
                predicted_all_times = results.get('predicted_spatial_all_times')  # xarray DataArray with time dimension
                
                if actual_all_times is not None and predicted_all_times is not None:
                    # Get the correct coordinate names
                    lat_name, lon_name = get_spatial_coord_names(actual_all_times)
                    
                    if lat_name is None or lon_name is None:
                        st.error("Could not find latitude/longitude coordinates in the data.")
                        return
                      # Check if we have time dimension data
                    time_dim = results.get('time_dim')
                    n_times = results.get('n_times', 1)
                    n_actual_times = results.get('n_actual_times', n_times)  # Original actual times
                    times_available = results.get('times_available')
                    forecast_horizon = results.get('forecast_horizon', 0)
                    
                    if time_dim and n_times > 1 and times_available is not None:
                        # Add time slider for multi-time data (including forecast times)
                        st.markdown("#### Time Selection")
                        
                        # Convert to local timezone and format for display
                        time_options = [format_datetime_with_timezone(t) for t in times_available]
                        current_tz = st.session_state.get('timezone_offset', 3)
                        tz_str = f"UTC{'+' if current_tz >= 0 else ''}{current_tz}"
                        
                        # Show information about actual vs forecast data
                        if forecast_horizon > 0:
                            st.info(f"📊 **Data Range:** {n_actual_times} actual time steps + {forecast_horizon} forecast steps = {n_times} total")
                        
                        time_index = st.select_slider(
                            f"Time Index (Local Time {tz_str})",
                            options=list(range(len(time_options))),
                            value=0,  # Default to the first time step
                            format_func=lambda x: time_options[x] + (" [FORECAST]" if x >= n_actual_times else " [ACTUAL]"),
                            key="spatial_prediction_time_slider",
                            help=f"Choose a time slice from {n_times} available time steps. Times shown in {tz_str}. Forecast data shown where actual data is unavailable."
                        )                        
                        # Get the selected time slice
                        actual = actual_all_times.isel({time_dim: time_index})
                        predicted = predicted_all_times.isel({time_dim: time_index})
                        
                        # Show status information
                        if time_index < n_actual_times:
                            st.info(f"🕒 **Showing results for:** {time_options[time_index]} [ACTUAL DATA]")
                        else:
                            st.info(f"🔮 **Showing results for:** {time_options[time_index]} [FORECAST DATA]")
                    else:
                        # Single time slice
                        actual = actual_all_times.squeeze() if actual_all_times.values.ndim > 2 else actual_all_times
                        predicted = predicted_all_times.squeeze() if predicted_all_times.values.ndim > 2 else predicted_all_times
                    
                    # Create side-by-side maps
                    from plotly.subplots import make_subplots
                    import plotly.graph_objects as go
                    
                    fig_maps = make_subplots(
                        rows=1, cols=2, 
                        subplot_titles=["Actual", "Predicted"],
                        horizontal_spacing=0.05
                    )
                    
                    # Common color scale
                    vmin = min(actual.values.min(), predicted.values.min())
                    vmax = max(actual.values.max(), predicted.values.max())
                    
                    # Actual field
                    fig_maps.add_trace(go.Heatmap(
                        z=actual.values,
                        x=actual[lon_name],
                        y=actual[lat_name],
                        colorscale='Viridis',
                        zmin=vmin,
                        zmax=vmax,
                        showscale=False,
                        hovertemplate=f'Lat: %{{y}}<br>Lon: %{{x}}<br>Actual: %{{z:.3f}} {variable_units}<extra></extra>'
                    ), row=1, col=1)
                    
                    # Predicted field
                    fig_maps.add_trace(go.Heatmap(
                        z=predicted.values,
                        x=predicted[lon_name],
                        y=predicted[lat_name],
                        colorscale='Viridis',
                        zmin=vmin,
                        zmax=vmax,
                        showscale=True,
                        colorbar=dict(title=f"{variable_full_name}<br>({variable_units})"),
                        hovertemplate=f'Lat: %{{y}}<br>Lon: %{{x}}<br>Predicted: %{{z:.3f}} {variable_units}<extra></extra>'
                    ), row=1, col=2)
                    
                    fig_maps.update_layout(
                        title=f"Spatial Field Comparison: {variable_full_name}",
                        height=500,
                        xaxis_title="Longitude",
                        yaxis_title="Latitude",
                        xaxis2_title="Longitude",
                        yaxis2_title="Latitude"
                    )
                    
                    st.plotly_chart(fig_maps, use_container_width=True)
                    
                    # Error (Difference) Map - also controlled by the same time slider
                    st.subheader("Prediction Error Analysis")
                    
                    diff = predicted - actual
                    
                    fig_err = go.Figure()
                    fig_err.add_trace(go.Heatmap(
                        z=diff.values,
                        x=actual[lon_name],
                        y=actual[lat_name],
                        colorscale='RdBu',
                        zmid=0,
                        colorbar=dict(title=f"Error<br>({variable_units})"),
                        hovertemplate=f'Lat: %{{y}}<br>Lon: %{{x}}<br>Error: %{{z:.3f}} {variable_units}<extra></extra>'
                    ))
                    
                    fig_err.update_layout(
                        title=f"Prediction Error Map (Predicted - Actual): {variable_full_name}",
                        xaxis_title="Longitude",
                        yaxis_title="Latitude",
                        height=500
                    )
                    
                    st.plotly_chart(fig_err, use_container_width=True)
                    
                    # Scatter plot for overall agreement
                    st.subheader("Prediction Accuracy Assessment")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Scatter plot with 1:1 line
                        fig_scatter = go.Figure()
                        
                        # Flatten arrays for scatter plot
                        actual_flat = actual.values.flatten()
                        predicted_flat = predicted.values.flatten()
                        
                        fig_scatter.add_trace(go.Scatter(
                            x=actual_flat,
                            y=predicted_flat,
                            mode='markers',
                            name='Grid Points',
                            marker=dict(
                                opacity=0.6,
                                size=3,
                                color='blue'
                            ),
                            hovertemplate=f'Actual: %{{x:.3f}} {variable_units}<br>Predicted: %{{y:.3f}} {variable_units}<extra></extra>'
                        ))
                        
                        # Perfect prediction line (1:1)
                        min_val = min(actual_flat.min(), predicted_flat.min())
                        max_val = max(actual_flat.max(), predicted_flat.max())
                        fig_scatter.add_trace(go.Scatter(
                            x=[min_val, max_val],
                            y=[min_val, max_val],
                            mode='lines',
                            name='Perfect Prediction (1:1)',
                            line=dict(dash='dash', color='red', width=2)
                        ))
                        
                        fig_scatter.update_layout(
                            title="Predicted vs. Actual Values",
                            xaxis_title=f"Actual {variable_full_name} ({variable_units})",
                            yaxis_title=f"Predicted {variable_full_name} ({variable_units})",
                            height=400
                        )
                        
                        st.plotly_chart(fig_scatter, use_container_width=True)
                    
                    with col2:
                        # Error distribution histogram
                        fig_hist = go.Figure()
                        
                        error_flat = diff.values.flatten()
                        
                        fig_hist.add_trace(go.Histogram(
                            x=error_flat,
                            nbinsx=30,
                            name='Error Distribution',
                            marker_color='lightblue',
                            opacity=0.7
                        ))
                        
                        # Add vertical line at zero
                        fig_hist.add_vline(
                            x=0,
                            line_dash="dash",
                            line_color="red",
                            annotation_text="Perfect Prediction"
                        )
                        
                        fig_hist.update_layout(
                            title="Error Distribution",
                            xaxis_title=f"Prediction Error ({variable_units})",
                            yaxis_title="Frequency",
                            height=400,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    # Summary statistics
                    with st.expander("📊 Detailed Error Statistics", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Mean Error", f"{np.mean(error_flat):.4f}")
                            st.metric("Std Error", f"{np.std(error_flat):.4f}")
                        
                        with col2:
                            st.metric("Min Error", f"{np.min(error_flat):.4f}")
                            st.metric("Max Error", f"{np.max(error_flat):.4f}")
                        
                        with col3:
                            st.metric("Mean Abs Error", f"{np.mean(np.abs(error_flat)):.4f}")
                            correlation = np.corrcoef(actual_flat, predicted_flat)[0, 1]
                            st.metric("Correlation", f"{correlation:.4f}")
                      # Add time series info if available
                    if time_dim and n_times > 1:
                        with st.expander("📅 Time Series Overview", expanded=False):
                            st.markdown(f"""
                            **Time Dimension Information:**
                            - **Time dimension name:** {time_dim}
                            - **Number of actual time steps:** {n_actual_times}
                            - **Number of forecast time steps:** {forecast_horizon}
                            - **Total time steps:** {n_times}
                            - **Time range:** {time_options[0]} to {time_options[-1]}
                            - **Current selection:** {time_options[time_index]} (step {time_index + 1} of {n_times})
                            - **Data type:** {"ACTUAL" if time_index < n_actual_times else "FORECAST"}
                            """)
                            
                            # Calculate time series of spatial statistics (including forecast)
                            mean_actual_series = []
                            mean_predicted_series = []
                            rmse_series = []
                            time_labels = []
                            
                            for t in range(n_times):
                                actual_t = actual_all_times.isel({time_dim: t})
                                predicted_t = predicted_all_times.isel({time_dim: t})
                                
                                actual_flat_t = actual_t.values.flatten()
                                predicted_flat_t = predicted_t.values.flatten()
                                
                                # Only use valid values for actual data
                                valid_mask_t = ~np.isnan(actual_flat_t)
                                
                                if t < n_actual_times and np.sum(valid_mask_t) > 0:
                                    # Actual data exists
                                    mean_actual_series.append(np.mean(actual_flat_t[valid_mask_t]))
                                    mean_predicted_series.append(np.mean(predicted_flat_t[valid_mask_t]))
                                    rmse_series.append(np.sqrt(np.mean((actual_flat_t[valid_mask_t] - predicted_flat_t[valid_mask_t])**2)))
                                    time_labels.append("Actual")
                                else:
                                    # Forecast data (no actual data available)
                                    mean_actual_series.append(np.nan)
                                    predicted_valid_mask = ~np.isnan(predicted_flat_t)
                                    if np.sum(predicted_valid_mask) > 0:
                                        mean_predicted_series.append(np.mean(predicted_flat_t[predicted_valid_mask]))
                                    else:
                                        mean_predicted_series.append(np.nan)
                                    rmse_series.append(np.nan)
                                    time_labels.append("Forecast")
                            
                            # Plot time series of spatial means (including forecast)
                            fig_ts = go.Figure()
                            
                            # Split into actual and forecast parts for different styling
                            actual_indices = list(range(n_actual_times))
                            forecast_indices = list(range(n_actual_times, n_times))
                            
                            # Actual data
                            if len(actual_indices) > 0:
                                fig_ts.add_trace(go.Scatter(
                                    x=actual_indices,
                                    y=[mean_actual_series[i] for i in actual_indices],
                                    mode='lines+markers',
                                    name='Actual (Spatial Mean)',
                                    line=dict(color='blue'),
                                    marker=dict(symbol='circle')
                                ))
                                
                                fig_ts.add_trace(go.Scatter(
                                    x=actual_indices,
                                    y=[mean_predicted_series[i] for i in actual_indices],
                                    mode='lines+markers',
                                    name='Predicted (Spatial Mean)',
                                    line=dict(color='red'),
                                    marker=dict(symbol='circle')
                                ))
                            
                            # Forecast data
                            if len(forecast_indices) > 0:
                                fig_ts.add_trace(go.Scatter(
                                    x=forecast_indices,
                                    y=[mean_predicted_series[i] for i in forecast_indices],
                                    mode='lines+markers',
                                    name='Forecast (Spatial Mean)',
                                    line=dict(color='green', dash='dash'),
                                    marker=dict(symbol='diamond')
                                ))
                            
                            # Add vertical line to separate actual from forecast
                            if forecast_horizon > 0:
                                fig_ts.add_vline(
                                    x=n_actual_times - 0.5,
                                    line_dash="dot",
                                    line_color="gray",
                                    annotation_text="Actual | Forecast"
                                )
                            
                            # Highlight current time step
                            fig_ts.add_vline(
                                x=time_index,
                                line_dash="dash",
                                line_color="purple",
                                annotation_text=f"Current: {time_options[time_index]}"
                            )
                            
                            fig_ts.update_layout(
                                title=f"Time Series of Spatial Mean {variable_full_name}",
                                xaxis_title="Time Step Index",
                                yaxis_title=f"Spatial Mean {variable_full_name} ({variable_units})",
                                height=400
                            )
                            
                            st.plotly_chart(fig_ts, use_container_width=True)
                
                # --- Download Spatial Prediction Results ---
                st.markdown("---")
                st.subheader("📥 Download Prediction Results")
                
                st.markdown("**Export spatial prediction data as NetCDF file compatible with ERA5 format**")
                
                # Download options
                col1, col2 = st.columns(2)
                with col1:
                    include_test_data = st.checkbox(
                        "Include Predicted Test Data",
                        value=True,
                        help="Include the predicted values for the test period (where actual data exists for comparison)"
                    )
                with col2:
                    include_future_forecast = st.checkbox(
                        "Include Future Forecast",
                        value=True,
                        help="Include the future forecast predictions (beyond the available actual data)"
                    )
                
                if include_test_data or include_future_forecast:
                    try:
                        # Prepare data for download
                        download_data = {}
                        download_coords = {}
                        download_attrs = {}
                        
                        # Get coordinate information
                        if 'actual_spatial_all_times' in results and results['actual_spatial_all_times'] is not None:
                            actual_data = results['actual_spatial_all_times']
                            predicted_data = results['predicted_spatial_all_times']
                            
                            # Get coordinate names and values
                            time_dim = results.get('time_dim', 'time')
                            lat_name = None
                            lon_name = None
                            
                            # Find lat/lon coordinate names
                            for coord_name in actual_data.coords:
                                if coord_name.lower() in ['lat', 'latitude']:
                                    lat_name = coord_name
                                elif coord_name.lower() in ['lon', 'longitude']:
                                    lon_name = coord_name
                            
                            if lat_name and lon_name:
                                # Base coordinates (always include)
                                download_coords[lat_name] = actual_data.coords[lat_name]
                                download_coords[lon_name] = actual_data.coords[lon_name]
                                
                                # Determine which time slices to include
                                n_actual_times = results.get('n_actual_times', 0)
                                n_total_times = results.get('n_times', 0)
                                
                                time_coords_to_include = []
                                predicted_slices = []
                                actual_slices = []
                                
                                # Include test data if requested
                                if include_test_data and n_actual_times > 0:
                                    test_indices = list(range(n_actual_times))
                                    time_coords_to_include.extend(test_indices)
                                    
                                    for i in test_indices:
                                        predicted_slices.append(predicted_data.isel({time_dim: i}))
                                        actual_slices.append(actual_data.isel({time_dim: i}))
                                
                                # Include future forecast if requested
                                if include_future_forecast and n_total_times > n_actual_times:
                                    forecast_indices = list(range(n_actual_times, n_total_times))
                                    time_coords_to_include.extend(forecast_indices)
                                    
                                    for i in forecast_indices:
                                        predicted_slices.append(predicted_data.isel({time_dim: i}))
                                        # Create NaN-filled actual data for forecast period
                                        actual_slice = predicted_data.isel({time_dim: i}).copy()
                                        actual_slice.values[:] = np.nan
                                        actual_slices.append(actual_slice)
                                
                                if time_coords_to_include:
                                    # Combine selected time slices
                                    combined_times = [predicted_data.coords[time_dim].values[i] for i in time_coords_to_include]
                                    download_coords[time_dim] = (time_dim, combined_times)
                                    
                                    # Stack the spatial data
                                    predicted_values = np.stack([slice_data.values for slice_data in predicted_slices])
                                    actual_values = np.stack([slice_data.values for slice_data in actual_slices])
                                    
                                    # Create variable names
                                    target_var = results['target_variable']
                                    predicted_var_name = f"predicted_{target_var}"
                                    
                                    # Add data variables
                                    download_data[target_var] = (
                                        [time_dim, lat_name, lon_name], 
                                        actual_values,
                                        {
                                            'long_name': f'Actual {variable_full_name}',
                                            'units': variable_units,
                                            'description': 'Original/actual values (NaN for forecast period)'
                                        }
                                    )
                                    
                                    download_data[predicted_var_name] = (
                                        [time_dim, lat_name, lon_name], 
                                        predicted_values,
                                        {
                                            'long_name': f'Predicted {variable_full_name}',
                                            'units': variable_units,
                                            'description': 'Model predictions from ConvLSTM',
                                            'model': results.get('model_name', 'ConvLSTM'),
                                            'rmse': results.get('rmse', 'N/A'),
                                            'mae': results.get('mae', 'N/A'),
                                            'r2': results.get('r2', 'N/A')
                                        }
                                    )
                                    
                                    # Global attributes
                                    download_attrs = {
                                        'title': f'Spatial Prediction Results - {variable_full_name}',
                                        'description': 'ERA5-compatible spatial prediction data generated by ConvLSTM model',
                                        'model': results.get('model_name', 'ConvLSTM'),
                                        'source': 'ERA5 Dashboard - Spatial Prediction Module',
                                        'creation_date': pd.Timestamp.now().isoformat(),
                                        'includes_test_data': str(include_test_data),
                                        'includes_forecast_data': str(include_future_forecast),
                                        'n_test_timesteps': str(n_actual_times if include_test_data else 0),
                                        'n_forecast_timesteps': str(n_total_times - n_actual_times if include_future_forecast else 0),
                                        'performance_rmse': str(results.get('rmse', 'N/A')),
                                        'performance_mae': str(results.get('mae', 'N/A')),
                                        'performance_r2': str(results.get('r2', 'N/A'))
                                    }
                                    
                                    if st.button("📥 Generate Download File", key="download_spatial_predictions"):
                                        with st.spinner("Preparing NetCDF file for download..."):
                                            try:
                                                # Create xarray Dataset
                                                download_dataset = xr.Dataset(
                                                    data_vars=download_data,
                                                    coords=download_coords,
                                                    attrs=download_attrs
                                                )
                                                
                                                # Create filename
                                                timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                                                components = []
                                                if include_test_data:
                                                    components.append("test")
                                                if include_future_forecast:
                                                    components.append("forecast")
                                                
                                                filename = f"spatial_prediction_{target_var}_{timestamp}_{'_'.join(components)}.nc"
                                                
                                                # Create NetCDF file data directly in memory
                                                file_data = download_dataset.to_netcdf()
                                                
                                                # Provide download
                                                st.download_button(
                                                    label=f"💾 Download {filename}",
                                                    data=file_data,
                                                    file_name=filename,
                                                    mime="application/x-netcdf",
                                                    help="Download ERA5-compatible NetCDF file with spatial predictions"
                                                )
                                                
                                                # Show file info
                                                st.success("✅ Download file generated successfully!")
                                                
                                                with st.expander("📋 File Information", expanded=False):
                                                    st.write(f"**Filename:** `{filename}`")
                                                    st.write(f"**Variables:** {list(download_data.keys())}")
                                                    st.write(f"**Dimensions:** {list(download_coords.keys())}")
                                                    st.write(f"**Shape:** {download_dataset.dims}")
                                                    if include_test_data:
                                                        st.write(f"**Test period:** {n_actual_times} time steps")
                                                    if include_future_forecast:
                                                        st.write(f"**Forecast period:** {n_total_times - n_actual_times} time steps")
                                                    st.write("**Compatibility:** ERA5 format, compatible with dashboard visualization tab")
                                                
                                            except Exception as e:
                                                st.error(f"❌ Error generating download file: {str(e)}")
                                                st.info("💡 Try reducing the spatial extent or time range to reduce file size.")
                                
                                else:
                                    st.warning("⚠️ No time steps selected. Please enable at least one option above.")
                            
                            else:
                                st.error("❌ Could not find latitude/longitude coordinates in the results.")
                        
                        else:
                            st.error("❌ No spatial prediction data available for download.")
                    
                    except Exception as e:
                        st.error(f"❌ Error preparing download: {str(e)}")
                else:
                    st.info("💡 Please select at least one data type to include in the download file.")
                
                # Successfully completed spatial results display
                return  # Exit here after displaying spatial results
                
            else:
                # We don't have spatial results, so show time series results instead
                # (This happens for single-point data with RandomForest, GBR, etc.)
                pass  # Continue to time series results section below
            
            # --- Time Series Results (existing code) ---
            st.markdown("**Model Performance (on Test Set)**")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("RMSE", f"{results['rmse']:.3f}")
            col2.metric("MAE", f"{results['mae']:.3f}")
            col3.metric("R-squared (R²)", f"{results['r2']:.3f}")
            if results.get("oob_score") is not None:
                col4.metric("Out-of-Bag (OOB) Score", f"{results['oob_score']:.3f}")            # --- Main Prediction Plot ---
            st.subheader("Prediction Visualization")
            # Get variable info from SHORT_NAME_MAP using the target_variable (which is likely a short name like 't2m')
            variable_info = SHORT_NAME_MAP.get(results['target_variable'], {})
            variable_full_name = variable_info.get('name', results['target_variable'])
            variable_units = variable_info.get('units', 'units')
            
            # Time series prediction visualization
            fig_pred = go.Figure()
            
            # 1. Actual Data (convert to local timezone)
            actual_times = convert_time_array_to_local_timezone(results['df'][results['time_dim']].values)
            fig_pred.add_trace(go.Scatter(
                x=actual_times,
                y=results['df'][results['target_variable']],
                mode='lines',
                name='Actual Data',
                line=dict(color='skyblue')
            ))

            # 2. Test Predictions (convert to local timezone)
            test_time_index = results['df'][results['time_dim']].iloc[-len(results['y_test']):]
            test_times_local = convert_time_array_to_local_timezone(test_time_index.values)
            fig_pred.add_trace(go.Scatter(
                x=test_times_local,
                y=results['y_pred_test'],
                mode='lines',
                name='Test Predictions',
                line=dict(color='orange')
            ))
            
            # 3. Future Forecast (convert to local timezone)
            if 'future_datetimes' in results and 'y_pred_future' in results:
                future_times_local = convert_time_array_to_local_timezone(results['future_datetimes'])
                fig_pred.add_trace(go.Scatter(
                    x=future_times_local,
                    y=results['y_pred_future'],
                    mode='lines',
                    name='Future Forecast',
                    line=dict(color='green')
                ))            # Get current timezone for labels
            current_tz = st.session_state.get('timezone_offset', 3)
            tz_str = f"UTC{'+' if current_tz >= 0 else ''}{current_tz}"
            
            fig_pred.update_layout(
                title=f"Actual vs. Predicted {variable_full_name} (Local Time {tz_str})",
                xaxis_title=f"Time ({tz_str})",
                yaxis_title=f"{variable_full_name} ({variable_units})",
                legend_title="Legend"
            )
            st.plotly_chart(fig_pred, use_container_width=True)

            # --- Feature Importance Plot ---
            if 'feature_importances' in results and 'feature_names' in results:
                with st.expander("View Feature Importances", expanded=False):
                    feature_names = results['feature_names']
                    importances = results['feature_importances']
                    
                    # Validate data
                    if len(feature_names) != len(importances):
                        st.error("Mismatch between feature names and importance values")
                    elif len(feature_names) == 0:
                        st.warning("No features found for importance analysis")
                    else:
                        importance_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': importances
                        }).sort_values(by='Importance', ascending=True)
                        
                        # Display the data table as well
                        st.markdown("##### Feature Importance Values")
                        st.dataframe(importance_df.sort_values(by='Importance', ascending=False), use_container_width=True)
                    
                    fig_importance = go.Figure()
                    fig_importance.add_trace(go.Bar(
                        y=importance_df['Feature'],
                        x=importance_df['Importance'],
                        orientation='h',
                        marker_color='lightblue'
                    ))
                    
                    # Make title more specific
                    importance_type = "Gini-based" if results["model_name"] == "RandomForestRegressor" else "Feature"
                    importance_desc = "(Mean Decrease in Impurity)" if results["model_name"] == "RandomForestRegressor" else ""
                    fig_importance.update_layout(
                        title_text=f'{importance_type} Feature Importances ({results["model_name"]}) {importance_desc}',
                        xaxis_title="Importance Score",
                        yaxis_title="Feature",
                        height=max(400, 50 + len(feature_names) * 25),  # Dynamic height
                        margin=dict(l=200, r=50, t=80, b=50),  # Increased left margin
                        yaxis=dict(tickfont=dict(size=10)),  # Smaller font for feature names
                        showlegend=False
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
                    st.info(f"{importance_type} importance is calculated on the training set and can sometimes be biased. Permutation importance (below) is often more reliable.")
                    
                    # --- Permutation Importance Plot ---
                    if 'perm_importance' in results and results.get('perm_importance') is not None:
                        st.markdown("---")
                        st.markdown("#### Permutation Importance (on Test Set)")
                        st.markdown("""
                        Permutation importance measures the decrease in the model's R² score when a single feature's values are randomly shuffled.
                        This technique shows how much the model relies on that feature for accurate predictions on the *unseen test set*.
                        A larger drop in performance indicates a more important feature. The error bars represent the standard deviation of the importance score across multiple shuffles.
                        """)
                        perm_importance_result = results['perm_importance']
                        perm_importance_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': perm_importance_result.importances_mean,
                            'Std Dev': perm_importance_result.importances_std
                        }).sort_values(by='Importance', ascending=True)
                        
                        # Display the permutation importance data table
                        st.markdown("##### Permutation Importance Values")
                        display_perm_df = perm_importance_df.sort_values(by='Importance', ascending=False).copy()
                        display_perm_df['Importance'] = display_perm_df['Importance'].round(4)
                        display_perm_df['Std Dev'] = display_perm_df['Std Dev'].round(4)
                        st.dataframe(display_perm_df, use_container_width=True)
                        
                        fig_perm_importance = go.Figure()
                        fig_perm_importance.add_trace(go.Bar(
                            y=perm_importance_df['Feature'],
                            x=perm_importance_df['Importance'],
                            error_x=dict(type='data', array=perm_importance_df['Std Dev']),
                            orientation='h',
                            marker_color='lightcoral'
                        ))
                        fig_perm_importance.update_layout(
                            title_text='Permutation Feature Importances (on Test Set)',
                            xaxis_title="Importance Score (Decrease in R²)",
                            yaxis_title="Feature",
                            height=max(400, 50 + len(feature_names) * 25),  # Dynamic height
                            margin=dict(l=200, r=50, t=80, b=50),  # Increased left margin
                            yaxis=dict(tickfont=dict(size=10)),  # Smaller font for feature names
                            showlegend=False
                        )
                        st.plotly_chart(fig_perm_importance, use_container_width=True)            # --- Model Diagnostics Expander ---
            with st.expander("Show Model Diagnostics", expanded=False):
                st.subheader("Model Performance Diagnostics")
                
                # First row of diagnostic plots
                col1, col2 = st.columns(2)
                
                with col1:
                    # Predicted vs. Actual Scatter Plot
                    st.markdown("##### Predicted vs. Actual Values")
                    st.info("This scatter plot compares the actual values with the model's predictions. Points closer to the diagonal line indicate better predictions.", icon="📈")
                    
                    fig_scatter = go.Figure()
                    fig_scatter.add_trace(go.Scatter(
                        x=results['y_test'],
                        y=results['y_pred_test'],
                        mode='markers',
                        name='Predictions',
                        marker=dict(opacity=0.7)
                    ))
                    
                    # Add perfect prediction line
                    min_val = min(results['y_test'].min(), results['y_pred_test'].min())
                    max_val = max(results['y_test'].max(), results['y_pred_test'].max())
                    fig_scatter.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(dash='dash', color='red')                    ))
                    
                    fig_scatter.update_layout(
                        title="Predicted vs. Actual Values",
                        xaxis_title=f"Actual Values ({variable_units})",
                        yaxis_title=f"Predicted Values ({variable_units})",
                        height=400
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                with col2:
                    # Residuals vs. Predicted Values Plot
                    st.markdown("##### Residuals vs. Predicted Values")
                    st.info("This plot shows the model's errors (residuals) against the predicted values. For a good model, the points should be randomly scattered around the horizontal line at y=0, with no obvious pattern.", icon="📊")
                    
                    residuals = results['y_test'] - results['y_pred_test']
                    fig_residuals = go.Figure()
                    fig_residuals.add_trace(go.Scatter(
                        x=results['y_pred_test'],
                        y=residuals,
                        mode='markers',
                        name='Residuals',
                        marker=dict(opacity=0.7, color='lightblue')
                    ))
                    
                    # Add horizontal line at y=0
                    x_min, x_max = results['y_pred_test'].min(), results['y_pred_test'].max()
                    fig_residuals.add_trace(go.Scatter(
                        x=[x_min, x_max],
                        y=[0, 0],
                        mode='lines',
                        name='Zero Line',
                        line=dict(dash='dash', color='red')                    ))
                    
                    fig_residuals.update_layout(
                        title="Residuals vs. Predicted Values",
                        xaxis_title=f"Predicted Values ({variable_units})",
                        yaxis_title=f"Residuals ({variable_units})",
                        height=400
                    )
                    st.plotly_chart(fig_residuals, use_container_width=True)
                
                # Second row of diagnostic plots
                col3, col4 = st.columns(2)
                with col3:
                    # Residuals Histogram
                    st.markdown("##### Residuals Distribution")
                    st.info("This histogram shows the distribution of the model's errors. Ideally, the distribution should be normal (a bell curve) and centered at zero, indicating that the errors are not systematically biased.", icon="📊")
                    residuals = results['y_test'] - results['y_pred_test']
                    fig_resid_hist = ff.create_distplot([residuals], ['Residuals'], show_hist=True, show_rug=False)
                    fig_resid_hist.update_layout(
                        title="Distribution of Residuals",
                        xaxis_title=f"Residual Value ({variable_units})",
                        yaxis_title="Density",
                        height=400
                    )
                    st.plotly_chart(fig_resid_hist, use_container_width=True)
                
                with col4:
                    # Placeholder for future diagnostic plot
                    st.markdown("##### Additional Diagnostics")
                    st.info("Additional diagnostic plots may be added here in future versions.", icon="🔧")
                    # You can add more diagnostic plots here if needed            # --- Learning Curve Analysis ---
            with st.expander("Show Learning Curve Analysis", expanded=False):
                st.markdown("""
                #### Learning Curves
                This plot shows the model's performance on the training set and the validation (cross-validation) set as more data is added. It's a key tool for diagnosing bias vs. variance issues.

                - **If the training and validation scores converge to a low error**, the model is likely well-fitted.
                - **If there is a large gap between the high training score and the low validation score**, the model is likely overfitting (high variance). It has memorized the training data but doesn't generalize well.
                - **If both scores are low and converge**, the model may be underfitting (high bias). It's too simple to capture the patterns in the data.
                """ )
                if "learning_curve" in results and results["learning_curve"]:
                    lc_data = results["learning_curve"]
                    train_scores_mean = -np.mean(lc_data['train_scores'], axis=1)
                    validation_scores_mean = -np.mean(lc_data['validation_scores'], axis=1)
                    
                    fig_lc = go.Figure()
                    fig_lc.add_trace(go.Scatter(
                        x=lc_data['train_sizes'], 
                        y=train_scores_mean, 
                        mode='lines+markers', 
                        name='Training Score (MSE)',
                        line=dict(color='#1f77b4', width=3),
                        marker=dict(size=8, symbol='circle')
                    ))
                    fig_lc.add_trace(go.Scatter(
                        x=lc_data['train_sizes'], 
                        y=validation_scores_mean, 
                        mode='lines+markers', 
                        name='Validation Score (MSE)',
                        line=dict(color='#ff7f0e', width=3),
                        marker=dict(size=8, symbol='diamond')
                    ))
                    fig_lc.update_layout(
                        title="Learning Curves",
                        xaxis_title="Number of Training Samples",
                        yaxis_title="Mean Squared Error (MSE)",
                        height=500,
                        yaxis_type="log",
                        showlegend=True,
                        legend=dict(
                            x=0.7,
                            y=0.95,
                            bgcolor="rgba(255,255,255,0.8)",
                            bordercolor="rgba(0,0,0,0.2)",
                            borderwidth=1
                        )
                    )
                    st.plotly_chart(fig_lc, use_container_width=True)
                else:
                    st.warning("Learning curve data not available for this model run.")

            # --- Download Predictions ---
            st.subheader("5. Download Predictions")
            st.info("You can download the generated predictions as a NetCDF file, which can be re-loaded into this dashboard for visualization and comparison.")

            col1, col2 = st.columns(2)
            with col1:
                include_test = st.checkbox("Include Test Set Predictions", value=True, key="download_include_test")
            with col2:
                include_future = st.checkbox("Include Future Forecast", value=True, key="download_include_future")

            if st.button("Prepare Download", key="prepare_download_button"):
                if not include_test and not include_future:
                    st.warning("Please select at least one dataset to include in the download.")
                else:
                    try:
                        with st.spinner("Generating NetCDF file..."):
                            netcdf_bytes = save_prediction_to_netcdf(
                                results,
                                include_test,
                                include_future,
                                VARIABLE_MAP
                            )
                            st.session_state.download_data = {
                                "bytes": netcdf_bytes,
                                "filename": f"prediction_{results['model_name']}_{results['target_variable']}_{pd.Timestamp.now().strftime('%Y%m%d%H%M')}.nc"
                            }
                    except Exception as e:
                        st.error(f"Failed to create NetCDF file: {e}")
                        if 'download_data' in st.session_state:
                            del st.session_state.download_data

            if 'download_data' in st.session_state and st.session_state.download_data:
                st.download_button(
                    label="Download NetCDF File",
                    data=st.session_state.download_data["bytes"],
                    file_name=st.session_state.download_data["filename"],
                    mime="application/x-netcdf",
                    on_click=lambda: st.session_state.pop('download_data', None)
                )
