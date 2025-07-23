"""
Machine Learning models and prediction functions for the ERA5 Dashboard.
"""

import pandas as pd
import numpy as np
import xarray as xr
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Deep Learning imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import ConvLSTM2D, Conv2D, BatchNormalization, Dropout, Dense, Reshape, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.regularizers import l2
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. ConvLSTM models will not work.")

from utils.helpers import get_time_dim_name

def _prepare_prediction_data(file_path, target_variable):
    """Loads data, performs feature engineering, and splits into X and y."""
    with xr.open_dataset(file_path) as ds:
        time_dim = get_time_dim_name(ds)
        df = ds.to_dataframe().reset_index()
        all_vars = list(ds.data_vars.keys())
        lat = ds.latitude.item() if 'latitude' in ds.coords else None
        lon = ds.longitude.item() if 'longitude' in ds.coords else None

    time_col = pd.to_datetime(df[time_dim])
    df['time_numeric'] = (time_col - time_col.min()).dt.total_seconds()
    df['month'] = time_col.dt.month
    df['day'] = time_col.dt.day
    df['hour'] = time_col.dt.hour
    df['dayofweek'] = time_col.dt.dayofweek
    df['dayofyear'] = time_col.dt.dayofyear
    df['quarter'] = time_col.dt.quarter
    df['weekofyear'] = time_col.dt.isocalendar().week.astype(int)

    features = ['time_numeric', 'month', 'day', 'hour', 'dayofweek', 'dayofyear', 'quarter', 'weekofyear']
    
    if len(all_vars) > 1:
        other_vars = [v for v in all_vars if v != target_variable]
        features.extend(other_vars)

    X = df[features]
    y = df[target_variable]

    return X, y, df, time_dim, features, all_vars, lat, lon, time_col

def _generate_future_features(df, time_dim, time_col, forecast_horizon, features, all_vars, target_variable):
    """Generates a DataFrame with features for future predictions."""
    last_time = pd.to_datetime(df[time_dim].max())
    time_step_seconds = df['time_numeric'].diff().mean()

    if pd.isna(time_step_seconds) or time_step_seconds == 0:
        try:
            freq = pd.infer_freq(df[time_dim])
            time_step_seconds = pd.to_timedelta(freq).total_seconds()
            if time_step_seconds is None or time_step_seconds == 0: 
                raise ValueError
        except (TypeError, ValueError):
            time_step_seconds = 3600  # Default to 1 hour

    future_datetimes = pd.to_datetime([last_time + pd.to_timedelta(i * time_step_seconds, unit='s') for i in range(1, forecast_horizon + 1)])
    
    future_df = pd.DataFrame(index=future_datetimes)
    future_df['time_numeric'] = (future_df.index - time_col.min()).total_seconds()
    future_df['month'] = future_df.index.month
    future_df['day'] = future_df.index.day
    future_df['hour'] = future_df.index.hour
    future_df['dayofweek'] = future_df.index.dayofweek
    future_df['dayofyear'] = future_df.index.dayofyear
    future_df['quarter'] = future_df.index.quarter
    future_df['weekofyear'] = future_df.index.isocalendar().week.astype(int)

    if len(all_vars) > 1:
        other_vars = [v for v in all_vars if v != target_variable]
        for var in other_vars:
            # Use the last known value to project forward
            future_df[var] = df[var].iloc[-1]

    return future_df[features], future_datetimes

def train_and_predict_rf(file_path, target_variable, n_estimators, max_depth, min_samples_split, 
                        min_samples_leaf, forecast_horizon, max_features, criterion, bootstrap, oob_score):
    """
    Trains a RandomForestRegressor model and generates future predictions.
    """
    # 1. Load and Prepare Data
    X, y, df, time_dim, features, all_vars, lat, lon, time_col = _prepare_prediction_data(file_path, target_variable)

    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # 2. Train Model for Evaluation
    eval_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        criterion=criterion,
        bootstrap=bootstrap,
        oob_score=oob_score,
        random_state=42,
        n_jobs=-1
    )
    eval_model.fit(X_train, y_train)

    # 3. Evaluate on Test Set
    oob_score_value = eval_model.oob_score_ if oob_score and bootstrap else None
    y_pred_test = eval_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)

    # --- Permutation Importance ---
    perm_importance_result = permutation_importance(
        eval_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )

    # 4. Retrain on Full Data for Final Forecast
    final_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        criterion=criterion,
        bootstrap=bootstrap,
        oob_score=False,  # OOB score is not needed for the final forecasting model
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(X, y)  # Retrain on the entire dataset

    # 5. Generate Learning Curves
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator=eval_model,
        X=X, y=y,
        train_sizes=[0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
        cv=3,  # 3-fold cross-validation
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        shuffle=False  # Important for time series
    )

    # 6. Generate Future Predictions
    future_features, future_datetimes = _generate_future_features(
        df, time_dim, time_col, forecast_horizon, features, all_vars, target_variable
    )
    y_pred_future = final_model.predict(future_features)

    return {
        "rmse": rmse, "mae": mae, "r2": r2, "oob_score": oob_score_value,
        "y_pred_test": y_pred_test, "y_test": y_test,
        "y_pred_future": y_pred_future,
        "df": df, "time_dim": time_dim,
        "future_datetimes": future_datetimes,
        "latitude": lat,
        "longitude": lon,
        "target_variable": target_variable,
        "model_name": "RandomForestRegressor",
        "feature_importances": eval_model.feature_importances_,
        "perm_importance": perm_importance_result,
        "feature_names": features,
        "learning_curve": {
            "train_sizes": train_sizes,
            "train_scores": train_scores,
            "validation_scores": validation_scores
        },
        "model_params": {
            'n_estimators': n_estimators, 'max_depth': max_depth, 'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf, 'max_features': max_features, 'criterion': criterion,
            'bootstrap': bootstrap, 'oob_score': oob_score
        }
    }

def train_and_predict_gbr(file_path, target_variable, n_estimators, max_depth, min_samples_split, 
                         min_samples_leaf, learning_rate, subsample, loss, forecast_horizon):
    """
    Trains a GradientBoostingRegressor model and generates future predictions.
    """
    # 1. Load and Prepare Data
    X, y, df, time_dim, features, all_vars, lat, lon, time_col = _prepare_prediction_data(file_path, target_variable)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # 2. Train Model for Evaluation
    eval_model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        learning_rate=learning_rate,
        subsample=subsample,
        loss=loss,
        random_state=42
    )
    eval_model.fit(X_train, y_train)

    # 3. Evaluate on Test Set
    y_pred_test = eval_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)

    # --- Permutation Importance ---
    perm_importance_result = permutation_importance(
        eval_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )

    # 4. Retrain on Full Data for Final Forecast
    final_model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        learning_rate=learning_rate,
        subsample=subsample,
        loss=loss,
        random_state=42
    )
    final_model.fit(X, y)

    # 5. Generate Learning Curves
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator=eval_model,
        X=X, y=y,
        train_sizes=[0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
        cv=3,  # 3-fold cross-validation
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        shuffle=False  # Important for time series
    )

    # 6. Generate Future Predictions
    future_features, future_datetimes = _generate_future_features(
        df, time_dim, time_col, forecast_horizon, features, all_vars, target_variable
    )
    y_pred_future = final_model.predict(future_features)

    return {
        "rmse": rmse, "mae": mae, "r2": r2, "oob_score": None,
        "y_pred_test": y_pred_test, "y_test": y_test,
        "y_pred_future": y_pred_future,
        "df": df, "time_dim": time_dim,
        "future_datetimes": future_datetimes,
        "latitude": lat,
        "longitude": lon,
        "target_variable": target_variable,
        "model_name": "GradientBoostingRegressor",
        "feature_importances": eval_model.feature_importances_,
        "perm_importance": perm_importance_result,
        "feature_names": features,
        "learning_curve": {
            "train_sizes": train_sizes,
            "train_scores": train_scores,
            "validation_scores": validation_scores
        },
        "model_params": {
            'n_estimators': n_estimators, 'max_depth': max_depth, 'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf, 'learning_rate': learning_rate, 'subsample': subsample,
            'loss': loss
        }
    }


def train_and_predict_idw_spatial(file_path, target_variable, power=2.0, min_neighbors=5, max_neighbors=20):
    """
    Perform spatial prediction using Inverse Distance Weighting (IDW).
    
    Args:
        file_path: Path to the spatial NetCDF file
        target_variable: Variable to predict
        power: IDW power parameter (higher = more weight to closer points)
        min_neighbors: Minimum number of neighbors for interpolation
        max_neighbors: Maximum number of neighbors for interpolation
        
    Returns:
        dict: Results containing spatial predictions and metrics
    """
    import scipy.spatial.distance as distance
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Load spatial data
    with xr.open_dataset(file_path) as ds:
        # Get the target variable data
        data_array = ds[target_variable]
        
        # Get the correct coordinate names
        lat_name, lon_name = _get_spatial_coord_names(ds)
        
        # Check if we have lat/lon dimensions
        if lat_name is None or lon_name is None:
            raise ValueError(f"Spatial data must have latitude/longitude dimensions. Found: {list(ds.dims)}")
        
        # Check if this is actually spatial data (more than one grid point)
        lat_size = len(ds[lat_name])
        lon_size = len(ds[lon_name])
        if lat_size == 1 and lon_size == 1:
            raise ValueError(f"This appears to be single-point data (1x1 grid), not spatial data. Use time series models instead.")
        
        # Get coordinates
        lats = ds[lat_name].values
        lons = ds[lon_name].values
        
        # Create coordinate mesh
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        # Get the most recent time slice for prediction
        time_dim = get_time_dim_name(ds)
        if time_dim and len(ds[time_dim]) > 1:
            # Use the last time step as "actual" and second-to-last for training
            actual_data = data_array.isel({time_dim: -1})
            training_data = data_array.isel({time_dim: -2})
        else:
            # If only one time step, split spatially for demonstration
            actual_data = data_array.squeeze()
            training_data = actual_data.copy()
    
    # Flatten spatial data
    actual_flat = actual_data.values.flatten() if actual_data.values.ndim > 1 else actual_data.values
    training_flat = training_data.values.flatten() if training_data.values.ndim > 1 else training_data.values
    
    # Create coordinate arrays
    coords = np.column_stack([lat_grid.flatten(), lon_grid.flatten()])
    
    # Remove NaN values
    valid_mask = ~np.isnan(training_flat)
    valid_coords = coords[valid_mask]
    valid_values = training_flat[valid_mask]
    
    if len(valid_values) < min_neighbors:
        raise ValueError(f"Not enough valid data points ({len(valid_values)}) for IDW interpolation")
    
    # Perform proper IDW interpolation with leave-one-out validation
    predicted_flat = np.full_like(actual_flat, np.nan)
    
    # For each point, predict it using OTHER points (leave-one-out approach)
    for i in range(len(coords)):
        if not valid_mask[i]:
            # For invalid points, skip prediction
            continue
        
        target_coord = coords[i:i+1]
        
        # Create training set by excluding the current point
        current_point_idx = np.where((valid_coords == coords[i]).all(axis=1))[0]
        if len(current_point_idx) > 0:
            # Exclude current point from training data
            training_mask = np.ones(len(valid_coords), dtype=bool)
            training_mask[current_point_idx[0]] = False
            
            training_coords = valid_coords[training_mask]
            training_values = valid_values[training_mask]
        else:
            # Fallback if point not found
            training_coords = valid_coords
            training_values = valid_values
        
        if len(training_coords) >= min_neighbors:
            # Calculate distances to training points only
            distances_to_training = distance.cdist(target_coord, training_coords)[0]
            
            # Avoid division by zero for coincident points
            distances_to_training = np.where(distances_to_training == 0, 1e-10, distances_to_training)
            
            # Sort by distance and take closest neighbors
            sorted_indices = np.argsort(distances_to_training)
            n_neighbors = min(max_neighbors, len(training_coords))
            n_neighbors = max(n_neighbors, min_neighbors)
            
            neighbor_indices = sorted_indices[:n_neighbors]
            neighbor_distances = distances_to_training[neighbor_indices]
            neighbor_values = training_values[neighbor_indices]
            
            # Calculate IDW weights
            weights = 1.0 / (neighbor_distances ** power)
            weights_sum = np.sum(weights)
            
            if weights_sum > 0:
                predicted_flat[i] = np.sum(weights * neighbor_values) / weights_sum
            else:
                # Fallback to mean if all weights are zero
                predicted_flat[i] = np.mean(neighbor_values)
        else:
            # Not enough neighbors, use mean of all available training values
            predicted_flat[i] = np.mean(valid_values)
    
    # Reshape back to original spatial dimensions
    original_shape = actual_data.values.shape
    predicted_spatial = predicted_flat.reshape(original_shape)
    
    # Create xarray DataArrays for output
    predicted_da = xr.DataArray(
        predicted_spatial,
        coords=actual_data.coords,
        dims=actual_data.dims,
        name=f"{target_variable}_predicted"
    )
    
    # Calculate metrics on valid grid points only
    # Create mask for valid (non-NaN) values in the original actual data
    valid_mask = ~np.isnan(actual_flat)
    actual_valid = actual_flat[valid_mask]
    predicted_valid = predicted_flat[valid_mask]
    
    if len(actual_valid) > 0 and len(predicted_valid) > 0:
        rmse = np.sqrt(mean_squared_error(actual_valid, predicted_valid))
        mae = mean_absolute_error(actual_valid, predicted_valid)
        r2 = r2_score(actual_valid, predicted_valid)
    else:
        rmse = mae = r2 = np.nan
    
    return {
        "rmse": rmse,
        "mae": mae, 
        "r2": r2,
        "actual_spatial": actual_data,
        "predicted_spatial": predicted_da,
        "target_variable": target_variable,
        "model_name": "InverseDistanceWeighting",
        "model_params": {
            'power': power,
            'min_neighbors': min_neighbors,
            'max_neighbors': max_neighbors
        },
        "n_valid_points": len(valid_values),
        "spatial_coverage": len(valid_values) / len(actual_flat.flatten())
    }

def train_and_predict_idw_spatial_all_times(file_path, target_variable, power=2.0, min_neighbors=5, max_neighbors=20, forecast_horizon=24):
    """
    Perform spatial prediction using Inverse Distance Weighting (IDW) for all time slices.
    This function is designed to work with the time slider in the prediction tab.
    
    Args:
        file_path: Path to the spatial NetCDF file
        target_variable: Variable to predict
        power: IDW power parameter (higher = more weight to closer points)
        min_neighbors: Minimum number of neighbors for interpolation
        max_neighbors: Maximum number of neighbors for interpolation
        forecast_horizon: Number of future time steps to forecast
    
    Returns:
        dict: Results containing spatial predictions for all time slices and metrics
    """
    import scipy.spatial.distance as distance
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Load spatial data
    with xr.open_dataset(file_path) as ds:
        # Get the target variable data
        data_array = ds[target_variable]
        
        # Get the correct coordinate names
        lat_name, lon_name = _get_spatial_coord_names(ds)
        
        # Check if we have lat/lon dimensions
        if lat_name is None or lon_name is None:
            raise ValueError(f"Spatial data must have latitude/longitude dimensions. Found: {list(ds.dims)}")
        
        # Check if this is actually spatial data (more than one grid point)
        lat_size = len(ds[lat_name])
        lon_size = len(ds[lon_name])
        if lat_size == 1 and lon_size == 1:
            raise ValueError(f"This appears to be single-point data (1x1 grid), not spatial data. Use time series models instead.")
        
        # Get coordinates
        lats = ds[lat_name].values
        lons = ds[lon_name].values
        
        # Create coordinate mesh
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        # Get time dimension info
        time_dim = get_time_dim_name(ds)
        
        if time_dim and len(ds[time_dim]) > 1:
            # For multi-time data, process existing time steps and generate future forecasts
            n_times = len(ds[time_dim])
            
            # Get time information for forecasting
            times_available = ds[time_dim].values
            
            # Generate future time steps
            time_step_seconds = pd.to_datetime(times_available).to_series().diff().dt.total_seconds().median()
            if pd.isna(time_step_seconds) or time_step_seconds == 0:
                time_step_seconds = 3600  # Default to 1 hour
            
            last_time = pd.to_datetime(times_available[-1])
            future_times = [last_time + pd.to_timedelta(i * time_step_seconds, unit='s') 
                          for i in range(1, forecast_horizon + 1)]
            
            # Combine existing and future times
            all_times = list(times_available) + future_times
            total_times = n_times + forecast_horizon
            
            # Process existing data and generate predictions
            actual_all_times = data_array
              # Create predicted data by applying proper IDW with spatial validation
            # Use leave-one-out cross-validation approach for realistic predictions
            predicted_all_times = []
            
            for t in range(total_times):
                if t < n_times:
                    # Process existing time steps
                    time_slice = data_array.isel({time_dim: t})
                    current_time = times_available[t]
                else:
                    # Generate future time steps using improved temporal forecasting
                    current_time = future_times[t - n_times]
                    future_steps = t - n_times + 1
                    
                    # Use a more sophisticated forecasting approach
                    if n_times >= 24:  # If we have at least 24 time steps, try seasonal patterns
                        time_slice = _generate_seasonal_forecast(data_array, current_time, times_available, future_steps)
                    elif n_times >= 3:  # If we have at least 3 points, use damped trend
                        time_slice = _generate_damped_trend_forecast(data_array, future_steps)
                    else:
                        # Fallback: use persistence (last known value with small random variation)
                        last_slice = data_array.isel({time_dim: -1})
                        # Add small cyclical variation instead of linear trend
                        time_slice = _generate_persistence_forecast(last_slice, future_steps)
                
                # Proper IDW spatial prediction with hold-out validation
                actual_flat = time_slice.values.flatten()
                coords = np.column_stack([lat_grid.flatten(), lon_grid.flatten()])
                
                # Remove NaN values
                valid_mask = ~np.isnan(actual_flat)
                valid_coords = coords[valid_mask]
                valid_values = actual_flat[valid_mask]
                
                if len(valid_values) >= min_neighbors:
                    # Perform proper IDW interpolation with spatial hold-out
                    predicted_flat = np.full_like(actual_flat, np.nan)
                    
                    # For each point, predict it using OTHER points (leave-one-out approach)
                    for i in range(len(actual_flat)):
                        if not valid_mask[i]:
                            # For invalid points, skip prediction
                            continue
                        
                        target_coord = coords[i:i+1]
                        
                        # Create training set by excluding the current point
                        current_point_idx = np.where((valid_coords == coords[i]).all(axis=1))[0]
                        if len(current_point_idx) > 0:
                            # Exclude current point from training data
                            training_mask = np.ones(len(valid_coords), dtype=bool)
                            training_mask[current_point_idx[0]] = False
                            
                            training_coords = valid_coords[training_mask]
                            training_values = valid_values[training_mask]
                        else:
                            # Fallback if point not found
                            training_coords = valid_coords
                            training_values = valid_values
                        
                        if len(training_coords) >= min_neighbors:
                            # Calculate distances to training points only
                            distances = distance.cdist(target_coord, training_coords)[0]
                            
                            # Get nearest neighbors from training set
                            neighbor_indices = np.argsort(distances)[:max_neighbors]
                            neighbor_distances = distances[neighbor_indices]
                            neighbor_values = training_values[neighbor_indices]
                            
                            # Filter by minimum neighbors
                            if len(neighbor_indices) >= min_neighbors:
                                # Calculate IDW weights (avoid division by zero)
                                weights = np.where(neighbor_distances == 0, 1e10, 1 / (neighbor_distances ** power))
                                weights_sum = np.sum(weights)
                                
                                if weights_sum > 0:
                                    predicted_flat[i] = np.sum(weights * neighbor_values) / weights_sum
                                else:
                                    predicted_flat[i] = np.mean(neighbor_values)
                            else:
                                predicted_flat[i] = np.mean(training_values)
                        else:
                            predicted_flat[i] = np.mean(valid_values)
                
                # Reshape back to spatial dimensions
                predicted_spatial = predicted_flat.reshape(time_slice.values.shape)
                
                # Create coordinates for this time step
                if t < n_times:
                    # Use existing coordinates from the actual data
                    predicted_da = xr.DataArray(
                        predicted_spatial,
                        coords=time_slice.coords,
                        dims=time_slice.dims,
                        name=f"{target_variable}_predicted"
                    )
                else:
                    # Create coordinates for future time step (2D spatial data)
                    coords_dict = {lat_name: lats, lon_name: lons}
                    dims_list = [lat_name, lon_name]
                    
                    # Create the DataArray with spatial coordinates only
                    spatial_da = xr.DataArray(
                        predicted_spatial,
                        coords=coords_dict,
                        dims=dims_list,
                        name=f"{target_variable}_predicted"
                    )
                    
                    # Add the time coordinate to make it 3D
                    predicted_da = spatial_da.expand_dims({time_dim: [current_time]})
                
                predicted_all_times.append(predicted_da)
            
            # Combine all predicted time slices
            predicted_all_times = xr.concat(predicted_all_times, dim=time_dim, coords='minimal')
            
            # Create extended actual data that includes NaN for future times
            # This allows the time slider to show future predictions
            actual_data_list = []
            for t in range(total_times):
                if t < n_times:
                    # Use existing actual data
                    actual_slice = data_array.isel({time_dim: t})
                    actual_data_list.append(actual_slice)
                else:
                    # Create NaN data for future times (no actual data available)
                    future_time = future_times[t - n_times]
                    nan_data = np.full_like(data_array.isel({time_dim: 0}).values, np.nan)
                    
                    # Create 2D spatial coordinates first
                    coords_dict = {lat_name: lats, lon_name: lons}
                    dims_list = [lat_name, lon_name]
                    
                    # Create the DataArray with spatial coordinates only
                    spatial_da = xr.DataArray(
                        nan_data,
                        coords=coords_dict,
                        dims=dims_list,
                        name=target_variable
                    )
                    
                    # Add the time coordinate to make it 3D
                    actual_future_da = spatial_da.expand_dims({time_dim: [future_time]})
                    actual_data_list.append(actual_future_da)
            
            # Combine actual data (existing + future NaN)
            actual_all_times = xr.concat(actual_data_list, dim=time_dim, coords='minimal')
            
            # Create extended times array that includes future times for the time slider
            extended_times = np.array(all_times)
            
        else:
            # Single time slice - use the existing logic
            actual_all_times = data_array.squeeze()
            predicted_all_times = actual_all_times.copy()  # Simple fallback
            extended_times = None
        
        # Calculate overall metrics using the last actual time slice (not forecast)
        if time_dim and len(ds[time_dim]) > 1:
            # Use the last actual data for metrics (not forecast data)
            actual_last = data_array.isel({time_dim: -1})
            predicted_last = predicted_all_times.isel({time_dim: n_times-1})  # Last actual prediction
        else:
            actual_last = actual_all_times
            predicted_last = predicted_all_times
        
        actual_flat = actual_last.values.flatten()
        predicted_flat = predicted_last.values.flatten()
        
        # Calculate metrics on valid grid points only
        valid_mask = ~np.isnan(actual_flat)
        actual_valid = actual_flat[valid_mask]
        predicted_valid = predicted_flat[valid_mask]
        
        if len(actual_valid) > 0 and len(predicted_valid) > 0:
            rmse = np.sqrt(mean_squared_error(actual_valid, predicted_valid))
            mae = mean_absolute_error(actual_valid, predicted_valid)
            r2 = r2_score(actual_valid, predicted_valid)
        else:
            rmse = mae = r2 = np.nan
    
    return {
        "rmse": rmse,
        "mae": mae, 
        "r2": r2,
        "actual_spatial_all_times": actual_all_times,
        "predicted_spatial_all_times": predicted_all_times,
        "target_variable": target_variable,
        "model_name": "InverseDistanceWeighting",
        "model_params": {
            'power': power,
            'min_neighbors': min_neighbors,
            'max_neighbors': max_neighbors
        },
        "time_dim": time_dim,
        "n_times": total_times if time_dim and len(ds[time_dim]) > 1 else 1,  # Include forecast times
        "n_actual_times": len(ds[time_dim]) if time_dim else 1,  # Just actual times
        "times_available": extended_times if extended_times is not None else (ds[time_dim].values if time_dim else None),  # Include forecast times
        "forecast_horizon": forecast_horizon
    }

def _clean_coordinates_for_concat(data_array, reference_array):
    """
    Clean coordinates to ensure compatibility for concatenation.
    Removes coordinates that don't exist in the reference array.
    
    Args:
        data_array: xarray DataArray to clean
        reference_array: Reference xarray DataArray for coordinate matching
    
    Returns:
        xarray DataArray with cleaned coordinates
    """
    coords_to_remove = []
    for coord_name in data_array.coords:
        if coord_name not in reference_array.coords:
            coords_to_remove.append(coord_name)
    
    if coords_to_remove:
        return data_array.drop_vars(coords_to_remove)
    else:
        return data_array

def train_and_predict_idw_spatiotemporal_cv(file_path, target_variable, power=2.0, min_neighbors=5, max_neighbors=20, 
                                          forecast_horizon=24, spatial_folds=5, temporal_folds=5, 
                                          cv_method='blocked', buffer_distance=0.5):
    """
    Perform spatial prediction using IDW with spatio-temporal cross-validation.
    This provides the most rigorous validation by holding out both spatial regions and temporal periods.
    
    Args:
        file_path: Path to the spatial NetCDF file
        target_variable: Variable to predict
        power: IDW power parameter
        min_neighbors: Minimum number of neighbors for interpolation
        max_neighbors: Maximum number of neighbors for interpolation
        forecast_horizon: Number of future time steps to forecast
        spatial_folds: Number of spatial folds for cross-validation
        temporal_folds: Number of temporal folds for cross-validation
        cv_method: 'blocked' for contiguous regions/periods, 'random' for scattered
        buffer_distance: Spatial buffer around test regions (degrees)
    
    Returns:
        dict: Results with spatiotemporal validation metrics
    """
    import scipy.spatial.distance as distance
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.model_selection import KFold
    
    # Load spatial data
    with xr.open_dataset(file_path) as ds:
        data_array = ds[target_variable]
        
        # Clean up any problematic coordinates (like 'expver') early
        problematic_coords = ['expver', 'number']  # Common ERA5 coords that cause issues
        coords_to_drop = [coord for coord in problematic_coords if coord in data_array.coords]
        if coords_to_drop:
            data_array = data_array.drop_vars(coords_to_drop)
        
        # Get coordinate names
        lat_name, lon_name = _get_spatial_coord_names(ds)
        if lat_name is None or lon_name is None:
            raise ValueError(f"Spatial data must have latitude/longitude dimensions. Found: {list(ds.dims)}")
        
        # Check spatial dimensions
        lat_size = len(ds[lat_name])
        lon_size = len(ds[lon_name])
        if lat_size == 1 and lon_size == 1:
            raise ValueError("This appears to be single-point data, not spatial data.")
        
        # Get coordinates and time
        lats = ds[lat_name].values
        lons = ds[lon_name].values
        time_dim = get_time_dim_name(ds)
        
        if not time_dim or len(ds[time_dim]) < temporal_folds:
            raise ValueError(f"Need at least {temporal_folds} time steps for temporal cross-validation")
        
        n_times = len(ds[time_dim])
        times_available = ds[time_dim].values
        
        # Create coordinate mesh
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        coords = np.column_stack([lat_grid.flatten(), lon_grid.flatten()])
        
        # Generate spatio-temporal CV folds
        def create_spatial_folds(lats, lons, n_folds, method='blocked', buffer_dist=0.5):
            """Create spatial cross-validation folds"""
            if method == 'blocked':
                # Create contiguous spatial blocks
                lat_edges = np.linspace(lats.min(), lats.max(), int(np.sqrt(n_folds)) + 1)
                lon_edges = np.linspace(lons.min(), lons.max(), int(np.ceil(n_folds / np.sqrt(n_folds))) + 1)
                
                folds = []
                fold_id = 0
                for i in range(len(lat_edges) - 1):
                    for j in range(len(lon_edges) - 1):
                        if fold_id >= n_folds:
                            break
                        
                        # Test region
                        test_mask = ((lat_grid >= lat_edges[i]) & (lat_grid < lat_edges[i + 1]) &
                                   (lon_grid >= lon_edges[j]) & (lon_grid < lon_edges[j + 1]))
                        
                        # Buffer region (exclude from training)
                        buffer_mask = ((lat_grid >= lat_edges[i] - buffer_dist) & 
                                     (lat_grid < lat_edges[i + 1] + buffer_dist) &
                                     (lon_grid >= lon_edges[j] - buffer_dist) & 
                                     (lon_grid < lon_edges[j + 1] + buffer_dist))
                        
                        train_mask = ~buffer_mask
                        
                        folds.append({
                            'train_mask': train_mask.flatten(),
                            'test_mask': test_mask.flatten(),
                            'fold_id': fold_id
                        })
                        fold_id += 1
                        
            else:  # random
                # Randomly assign grid points to folds
                n_points = len(coords)
                point_indices = np.arange(n_points)
                np.random.shuffle(point_indices)
                
                folds = []
                points_per_fold = n_points // n_folds
                
                for fold_id in range(n_folds):
                    start_idx = fold_id * points_per_fold
                    end_idx = (fold_id + 1) * points_per_fold if fold_id < n_folds - 1 else n_points
                    
                    test_indices = point_indices[start_idx:end_idx]
                    train_indices = np.setdiff1d(point_indices, test_indices)
                    
                    test_mask = np.zeros(n_points, dtype=bool)
                    train_mask = np.zeros(n_points, dtype=bool)
                    test_mask[test_indices] = True
                    train_mask[train_indices] = True
                    
                    folds.append({
                        'train_mask': train_mask,
                        'test_mask': test_mask,
                        'fold_id': fold_id
                    })
            
            return folds
        
        def create_temporal_folds(n_times, n_folds, method='blocked'):
            """Create temporal cross-validation folds"""
            if method == 'blocked':
                # Create contiguous temporal blocks
                times_per_fold = n_times // n_folds
                folds = []
                
                for fold_id in range(n_folds):
                    start_idx = fold_id * times_per_fold
                    end_idx = (fold_id + 1) * times_per_fold if fold_id < n_folds - 1 else n_times
                    
                    test_indices = np.arange(start_idx, end_idx)
                    train_indices = np.setdiff1d(np.arange(n_times), test_indices)
                    
                    folds.append({
                        'train_indices': train_indices,
                        'test_indices': test_indices,
                        'fold_id': fold_id
                    })
            else:  # random
                # Random temporal assignment
                kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
                folds = []
                
                for fold_id, (train_indices, test_indices) in enumerate(kfold.split(np.arange(n_times))):
                    folds.append({
                        'train_indices': train_indices,
                        'test_indices': test_indices,
                        'fold_id': fold_id
                    })
            
            return folds
        
        # Create spatial and temporal folds
        spatial_folds_data = create_spatial_folds(lats, lons, spatial_folds, cv_method, buffer_distance)
        temporal_folds_data = create_temporal_folds(n_times, temporal_folds, cv_method)
        
        # Perform spatio-temporal cross-validation
        cv_results = []
        all_predictions = np.full_like(data_array.values, np.nan)
        validation_mask = np.zeros_like(data_array.values, dtype=bool)
        
        for spatial_fold in spatial_folds_data:
            for temporal_fold in temporal_folds_data:
                # Get train/test indices for this fold combination
                spatial_train_mask = spatial_fold['train_mask']
                spatial_test_mask = spatial_fold['test_mask']
                temporal_train_indices = temporal_fold['train_indices']
                temporal_test_indices = temporal_fold['test_indices']
                
                fold_predictions = []
                fold_actuals = []
                
                # For each test time step, train on spatial training data from training times
                for test_t_idx in temporal_test_indices:
                    test_time_slice = data_array.isel({time_dim: test_t_idx})
                    
                    # Collect training data from training time steps and training spatial locations
                    train_coords_list = []
                    train_values_list = []
                    
                    for train_t_idx in temporal_train_indices:
                        train_time_slice = data_array.isel({time_dim: train_t_idx})
                        train_values_flat = train_time_slice.values.flatten()
                        
                        # Use only spatial training locations
                        valid_train_mask = spatial_train_mask & ~np.isnan(train_values_flat)
                        
                        if np.sum(valid_train_mask) > 0:
                            train_coords_list.append(coords[valid_train_mask])
                            train_values_list.append(train_values_flat[valid_train_mask])
                    
                    if len(train_coords_list) == 0:
                        continue
                    
                    # Combine all training data
                    all_train_coords = np.vstack(train_coords_list)
                    all_train_values = np.concatenate(train_values_list)
                    
                    # Predict at test spatial locations
                    test_coords = coords[spatial_test_mask]
                    test_actual = test_time_slice.values.flatten()[spatial_test_mask]
                    
                    valid_test_mask = ~np.isnan(test_actual)
                    if np.sum(valid_test_mask) == 0:
                        continue
                    
                    test_coords_valid = test_coords[valid_test_mask]
                    test_actual_valid = test_actual[valid_test_mask]
                    
                    # IDW prediction for all test points
                    test_predicted = []
                    for target_coord in test_coords_valid:
                        # Calculate distances to all training points
                        distances = distance.cdist([target_coord], all_train_coords)[0]
                        
                        # Filter out training points that are too close (avoid overfitting)
                        min_distance_threshold = 1e-6  # Very small threshold to avoid numerical issues
                        valid_train_mask = distances > min_distance_threshold
                        
                        if np.sum(valid_train_mask) >= min_neighbors:
                            valid_distances = distances[valid_train_mask]
                            valid_values = all_train_values[valid_train_mask]
                            
                            # Select nearest neighbors from valid training points
                            nearest_indices = np.argsort(valid_distances)[:max_neighbors]
                            nearest_distances = valid_distances[nearest_indices]
                            nearest_values = valid_values[nearest_indices]
                            
                            # Calculate IDW weights
                            weights = 1.0 / (nearest_distances ** power)
                            predicted_value = np.average(nearest_values, weights=weights)
                        else:
                            # Fallback: use all training points if not enough valid ones
                            if len(all_train_values) > 0:
                                if distances[0] == 0:
                                    predicted_value = all_train_values[0]
                                else:
                                    # Select nearest neighbors
                                    nearest_indices = np.argsort(distances)[:min(max_neighbors, len(distances))]
                                    nearest_distances = distances[nearest_indices]
                                    nearest_values = all_train_values[nearest_indices]
                                    
                                    # Handle exact matches
                                    if nearest_distances[0] == 0:
                                        predicted_value = nearest_values[0]
                                    else:
                                        weights = 1.0 / (nearest_distances ** power)
                                        predicted_value = np.average(nearest_values, weights=weights)
                            else:
                                # Last resort: use NaN
                                predicted_value = np.nan
                        
                        test_predicted.append(predicted_value)
                    
                    test_predicted = np.array(test_predicted)
                    
                    # Store results for this fold
                    fold_predictions.extend(test_predicted)
                    fold_actuals.extend(test_actual_valid)
                    
                    # Store in full prediction array - FIXED indexing
                    test_spatial_indices = np.where(spatial_test_mask)[0]
                    valid_test_spatial_indices = test_spatial_indices[valid_test_mask]
                    
                    for i, pred_val in enumerate(test_predicted):
                        if not np.isnan(pred_val):  # Only store valid predictions
                            flat_idx = valid_test_spatial_indices[i]
                            lat_idx, lon_idx = np.unravel_index(flat_idx, (len(lats), len(lons)))
                            all_predictions[test_t_idx, lat_idx, lon_idx] = pred_val
                            validation_mask[test_t_idx, lat_idx, lon_idx] = True
                
                # Calculate metrics for this fold combination
                if len(fold_predictions) > 0:
                    fold_rmse = np.sqrt(mean_squared_error(fold_actuals, fold_predictions))
                    fold_mae = mean_absolute_error(fold_actuals, fold_predictions)
                    fold_r2 = r2_score(fold_actuals, fold_predictions)
                    
                    cv_results.append({
                        'spatial_fold': spatial_fold['fold_id'],
                        'temporal_fold': temporal_fold['fold_id'],
                        'rmse': fold_rmse,
                        'mae': fold_mae,
                        'r2': fold_r2,
                        'n_predictions': len(fold_predictions)
                    })
        
        # Calculate overall cross-validation metrics
        if cv_results:
            cv_rmse = np.mean([r['rmse'] for r in cv_results])
            cv_mae = np.mean([r['mae'] for r in cv_results])
            cv_r2 = np.mean([r['r2'] for r in cv_results])
            cv_std_rmse = np.std([r['rmse'] for r in cv_results])
            cv_std_mae = np.std([r['mae'] for r in cv_results])
            cv_std_r2 = np.std([r['r2'] for r in cv_results])
        else:
            cv_rmse = cv_mae = cv_r2 = np.nan
            cv_std_rmse = cv_std_mae = cv_std_r2 = np.nan
        
        # Fill in missing predictions using simple IDW on all available data
        # This ensures every grid point has a prediction
        print(f"Filling missing predictions for {n_times} time steps...")
        total_filled = 0
        
        for t_idx in range(n_times):
            time_slice = data_array.isel({time_dim: t_idx})
            time_slice_flat = time_slice.values.flatten()
            
            # Find points that still need predictions
            missing_mask = np.isnan(all_predictions[t_idx].flatten())
            valid_data_mask = ~np.isnan(time_slice_flat)
            
            # Get indices of missing points that have valid actual data
            points_to_predict = missing_mask & valid_data_mask
            
            if np.any(points_to_predict):
                print(f"Time {t_idx}: Filling {np.sum(points_to_predict)} missing predictions out of {len(time_slice_flat)} total points")
                
                # Use all valid data points as training for missing predictions
                train_coords = coords[valid_data_mask]
                train_values = time_slice_flat[valid_data_mask]
                
                if len(train_coords) == 0:
                    print(f"Warning: No valid training data for time {t_idx}")
                    continue
                
                missing_coords = coords[points_to_predict]
                
                for i, target_coord in enumerate(missing_coords):
                    # Calculate distances to all training points
                    distances = distance.cdist([target_coord], train_coords)[0]
                    
                    # Select nearest neighbors
                    n_neighbors = min(max_neighbors, len(train_coords))
                    nearest_indices = np.argsort(distances)[:n_neighbors]
                    nearest_distances = distances[nearest_indices]
                    nearest_values = train_values[nearest_indices]
                    
                    # Calculate prediction
                    if len(nearest_distances) > 0:
                        if nearest_distances[0] == 0:
                            predicted_value = nearest_values[0]
                        else:
                            weights = 1.0 / (nearest_distances ** power)
                            predicted_value = np.average(nearest_values, weights=weights)
                        
                        # Store the prediction
                        missing_point_idx = np.where(points_to_predict)[0][i]
                        lat_idx, lon_idx = np.unravel_index(missing_point_idx, (len(lats), len(lons)))
                        all_predictions[t_idx, lat_idx, lon_idx] = predicted_value
                        total_filled += 1
        
        print(f"Total missing predictions filled: {total_filled}")
        
        # Additional pass to ensure no grid points are left without predictions
        # Use the most recent valid data as reference for spatial interpolation
        print("Performing final check for completely missing grid points...")
        final_filled = 0
        
        # Find the most recent time step with good data coverage
        coverage_per_time = []
        for t_idx in range(n_times):
            time_slice = data_array.isel({time_dim: t_idx})
            valid_count = np.sum(~np.isnan(time_slice.values.flatten()))
            coverage_per_time.append(valid_count)
        
        if coverage_per_time:
            best_time_idx = np.argmax(coverage_per_time)
            reference_slice = data_array.isel({time_dim: best_time_idx})
            reference_flat = reference_slice.values.flatten()
            reference_valid_mask = ~np.isnan(reference_flat)
            
            if np.any(reference_valid_mask):
                reference_coords = coords[reference_valid_mask]
                reference_values = reference_flat[reference_valid_mask]
                
                # Check for any grid points that are still missing across all times
                for t_idx in range(n_times):
                    pred_flat = all_predictions[t_idx].flatten()
                    still_missing = np.isnan(pred_flat)
                    
                    if np.any(still_missing):
                        missing_coords = coords[still_missing]
                        
                        for i, target_coord in enumerate(missing_coords):
                            # Use reference time data for interpolation
                            distances = distance.cdist([target_coord], reference_coords)[0]
                            n_neighbors = min(max_neighbors, len(reference_coords))
                            nearest_indices = np.argsort(distances)[:n_neighbors]
                            nearest_distances = distances[nearest_indices]
                            nearest_values = reference_values[nearest_indices]
                            
                            if len(nearest_distances) > 0:
                                if nearest_distances[0] == 0:
                                    predicted_value = nearest_values[0]
                                else:
                                    weights = 1.0 / (nearest_distances ** power)
                                    predicted_value = np.average(nearest_values, weights=weights)
                                
                                missing_point_idx = np.where(still_missing)[0][i]
                                lat_idx, lon_idx = np.unravel_index(missing_point_idx, (len(lats), len(lons)))
                                all_predictions[t_idx, lat_idx, lon_idx] = predicted_value
                                final_filled += 1
        
        print(f"Final pass filled {final_filled} additional predictions")
        
        # Report final coverage
        total_predictions = all_predictions.size
        valid_predictions = np.sum(~np.isnan(all_predictions))
        coverage_pct = (valid_predictions / total_predictions) * 100
        print(f"Final prediction coverage: {valid_predictions}/{total_predictions} ({coverage_pct:.2f}%)")
        
        # Create prediction arrays with proper coordinates
        predicted_all_times = xr.DataArray(
            all_predictions,
            dims=data_array.dims,
            coords=data_array.coords
        )
        
        # Generate future forecasts using simple temporal extrapolation
        time_step_seconds = pd.to_datetime(times_available).to_series().diff().dt.total_seconds().median()
        if pd.isna(time_step_seconds) or time_step_seconds == 0:
            time_step_seconds = 3600
        
        last_time = pd.to_datetime(times_available[-1])
        future_times = [last_time + pd.to_timedelta(i * time_step_seconds, unit='s') 
                       for i in range(1, forecast_horizon + 1)]
        
        # Simple future prediction using trend from last few time steps
        n_trend_steps = min(5, n_times)
        last_slices = data_array.isel({time_dim: slice(-n_trend_steps, None)})
        
        future_predictions = []
        print(f"Generating {forecast_horizon} future predictions...")
        
        for i in range(forecast_horizon):
            # Simple linear extrapolation
            if n_trend_steps >= 2:
                trend = (last_slices.isel({time_dim: -1}) - last_slices.isel({time_dim: -2}))
                future_slice = last_slices.isel({time_dim: -1}) + trend * (i + 1)
            else:
                future_slice = last_slices.isel({time_dim: -1})
            
            # Ensure future predictions have complete spatial coverage
            # Fill any NaN values using spatial interpolation from the last valid prediction
            future_flat = future_slice.values.flatten()
            missing_future = np.isnan(future_flat)
            
            if np.any(missing_future) and n_times > 0:
                # Use the last time step with valid predictions as reference
                last_pred_flat = all_predictions[-1].flatten()
                valid_last_pred = ~np.isnan(last_pred_flat)
                
                if np.any(valid_last_pred):
                    last_pred_coords = coords[valid_last_pred]
                    last_pred_values = last_pred_flat[valid_last_pred]
                    
                    missing_coords = coords[missing_future]
                    
                    for j, target_coord in enumerate(missing_coords):
                        distances = distance.cdist([target_coord], last_pred_coords)[0]
                        n_neighbors = min(max_neighbors, len(last_pred_coords))
                        nearest_indices = np.argsort(distances)[:n_neighbors]
                        nearest_distances = distances[nearest_indices]
                        nearest_values = last_pred_values[nearest_indices]
                        
                        if len(nearest_distances) > 0:
                            if nearest_distances[0] == 0:
                                predicted_value = nearest_values[0]
                            else:
                                weights = 1.0 / (nearest_distances ** power)
                                predicted_value = np.average(nearest_values, weights=weights)
                            
                            # Update the future slice
                            future_values = future_slice.values.copy()
                            future_values.flat[missing_point_idx] = predicted_value
                            future_slice = xr.DataArray(
                                future_values,
                                dims=future_slice.dims,
                                coords=future_slice.coords
                            )
            
            future_predictions.append(future_slice)
        
        if future_predictions:
            # Create future coordinates, ensuring compatibility with original data
            future_coords = {}
            for dim in data_array.dims:
                if dim == time_dim:
                    future_coords[dim] = future_times
                else:
                    future_coords[dim] = data_array.coords[dim]
            
            # Only include coordinates that exist in the original data_array
            # This prevents issues with mismatched coordinates like 'expver'
            filtered_future_coords = {
                coord_name: coord_val for coord_name, coord_val in future_coords.items()
                if coord_name in data_array.coords
            }
            
            future_array = xr.concat(future_predictions, dim=time_dim, coords='minimal').assign_coords(**filtered_future_coords)
            predicted_extended = xr.concat([predicted_all_times, future_array], dim=time_dim, coords='minimal')
            
            # Extend actual data with NaN for future
            nan_shape = list(data_array.shape)
            nan_shape[data_array.get_axis_num(time_dim)] = forecast_horizon
            
            # Create future NaN array with only coordinates that exist in original data
            future_nan_coords = {}
            for dim in data_array.dims:
                if dim == time_dim:
                    future_nan_coords[dim] = future_times
                else:
                    future_nan_coords[dim] = data_array.coords[dim]
            
            future_nan = xr.DataArray(
                np.full(nan_shape, np.nan),
                dims=data_array.dims,
                coords=future_nan_coords
            )
            
            # Before concatenating, ensure coordinate compatibility
            # Remove any coordinates from future_nan that don't exist in data_array
            coords_to_remove = []
            for coord_name in future_nan.coords:
                if coord_name not in data_array.coords:
                    coords_to_remove.append(coord_name)
            
            if coords_to_remove:
                future_nan = future_nan.drop_vars(coords_to_remove)
            
            actual_extended = xr.concat([data_array, future_nan], dim=time_dim, coords='minimal')
            
            all_times = list(times_available) + future_times
        else:
            predicted_extended = predicted_all_times
            actual_extended = data_array
            all_times = list(times_available)
        
        return {
            'actual_spatial_all_times': actual_extended,
            'predicted_spatial_all_times': predicted_extended,
            'rmse': cv_rmse,
            'mae': cv_mae,
            'r2': cv_r2,
            'cv_std_rmse': cv_std_rmse,
            'cv_std_mae': cv_std_mae,
            'cv_std_r2': cv_std_r2,
            'cv_results': cv_results,
            'target_variable': target_variable,
            'model_name': 'InverseDistanceWeighting_SpatioTemporalCV',
            'time_dim': time_dim,
            'n_times': len(all_times),
            'n_actual_times': n_times,
            'forecast_horizon': forecast_horizon,
            'times_available': all_times,
            'validation_method': 'spatiotemporal_cv',
            'validation_coverage': float(np.sum(validation_mask)) / validation_mask.size,
            'hyperparameters': {
                'power': power,
                'min_neighbors': min_neighbors,
                'max_neighbors': max_neighbors,
                'spatial_folds': spatial_folds,
                'temporal_folds': temporal_folds,
                'cv_method': cv_method,
                'buffer_distance': buffer_distance
            }
        }

def _get_spatial_coord_names(dataset):
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

def _prepare_spatial_sequences(data_array, time_dim, sequence_length=5, forecast_horizon=1):
    """
    Prepare sequences for ConvLSTM training.
    
    Args:
        data_array: xarray DataArray with spatial-temporal data
        time_dim: name of time dimension
        sequence_length: number of time steps to use as input
        forecast_horizon: number of time steps to predict
    
    Returns:
        X: input sequences (samples, time_steps, height, width, channels)
        y: target sequences (samples, height, width, channels)
        time_coords: time coordinates for each sample
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is required for ConvLSTM models")
    
    # Get spatial dimensions
    dims = list(data_array.dims)
    spatial_dims = [dim for dim in dims if dim != time_dim]
    
    if len(spatial_dims) != 2:
        raise ValueError(f"Expected 2 spatial dimensions, got {len(spatial_dims)}: {spatial_dims}")
    
    lat_dim, lon_dim = spatial_dims
    n_times, n_lat, n_lon = data_array.shape
    
    # Prepare sequences
    X, y, time_coords = [], [], []
    
    for i in range(sequence_length, n_times - forecast_horizon + 1):
        # Input sequence: past sequence_length time steps
        input_seq = data_array.isel({time_dim: slice(i - sequence_length, i)})
        
        # Target: next forecast_horizon time steps
        if forecast_horizon == 1:
            target = data_array.isel({time_dim: i})
        else:
            target = data_array.isel({time_dim: slice(i, i + forecast_horizon)})
        
        # Skip if any NaN values
        if not (np.isnan(input_seq.values).any() or np.isnan(target.values).any()):
            # Reshape to (time_steps, height, width, channels)
            X.append(input_seq.values[..., np.newaxis])  # Add channel dimension
            
            if forecast_horizon == 1:
                y.append(target.values[..., np.newaxis])
            else:
                y.append(target.values[..., np.newaxis])
            
            time_coords.append(data_array.coords[time_dim].values[i])
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y, time_coords


def _prepare_spatial_sequences_enhanced(data_array, time_dim, sequence_length=5, forecast_horizon=1):
    """
    Enhanced sequence preparation for ConvLSTM with multi-step training capability.
    
    Args:
        data_array: xarray DataArray with spatial-temporal data
        time_dim: name of time dimension
        sequence_length: number of time steps to use as input
        forecast_horizon: number of time steps to predict (can be > 1 for multi-step)
    
    Returns:
        X: input sequences (samples, time_steps, height, width, channels)
        y: target sequences (samples, forecast_horizon, height, width, channels) or (samples, height, width, channels)
        time_coords: time coordinates for each sample
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is required for ConvLSTM models")
    
    # Get spatial dimensions
    dims = list(data_array.dims)
    spatial_dims = [dim for dim in dims if dim != time_dim]
    
    if len(spatial_dims) != 2:
        raise ValueError(f"Expected 2 spatial dimensions, got {len(spatial_dims)}: {spatial_dims}")
    
    lat_dim, lon_dim = spatial_dims
    n_times, n_lat, n_lon = data_array.shape
    
    # Prepare sequences
    X, y, time_coords = [], [], []
    
    for i in range(sequence_length, n_times - forecast_horizon + 1):
        # Input sequence: past sequence_length time steps
        input_seq = data_array.isel({time_dim: slice(i - sequence_length, i)})
        
        # Target: next forecast_horizon time steps
        if forecast_horizon == 1:
            target = data_array.isel({time_dim: i})
        else:
            target = data_array.isel({time_dim: slice(i, i + forecast_horizon)})
        
        # Skip if any NaN values in input
        if np.isnan(input_seq.values).any():
            continue
            
        # For targets, be more lenient with NaN values
        if forecast_horizon == 1:
            if np.isnan(target.values).any():
                continue
        else:
            # For multi-step, skip only if more than 50% are NaN
            nan_ratio = np.isnan(target.values).sum() / target.values.size
            if nan_ratio > 0.5:
                continue
        
        # Reshape to (time_steps, height, width, channels)
        X.append(input_seq.values[..., np.newaxis])  # Add channel dimension
        
        if forecast_horizon == 1:
            y.append(target.values[..., np.newaxis])
        else:
            # For multi-step, return the full sequence with time dimension
            # Shape will be (forecast_horizon, height, width, 1)
            y.append(target.values[..., np.newaxis])
        
        time_coords.append(data_array.coords[time_dim].values[i])
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y, time_coords


def _normalize_spatial_data_enhanced(X, y):
    """
    Enhanced normalization that preserves spatial structure and provides better statistics.
    
    Args:
        X: input sequences (samples, time_steps, height, width, channels)
        y: target sequences (samples, [time_steps,] height, width, channels)
    
    Returns:
        X_scaled: normalized input sequences
        y_scaled: normalized target sequences
        scaler_stats: statistics for denormalization
    """
    # Calculate statistics on the entire spatial-temporal dataset
    all_data = np.concatenate([X.flatten(), y.flatten()])
    data_mean = np.nanmean(all_data)
    data_std = np.nanstd(all_data)
    
    # Avoid division by zero
    if data_std == 0:
        data_std = 1.0
    
    # Normalize
    X_scaled = (X - data_mean) / data_std
    y_scaled = (y - data_mean) / data_std
    
    # Store statistics for denormalization
    scaler_stats = {
        'mean': data_mean,
        'std': data_std
    }
    
    return X_scaled, y_scaled, scaler_stats


def _denormalize_spatial_data_enhanced(data_scaled, scaler_stats):
    """
    Denormalize data using stored statistics.
    
    Args:
        data_scaled: normalized data
        scaler_stats: statistics from normalization
    
    Returns:
        data_original: denormalized data
    """
    return data_scaled * scaler_stats['std'] + scaler_stats['mean']


def _create_convlstm_model_enhanced(input_shape, filters=[32, 16], kernel_size=(3, 3), dropout_rate=0.1,
                                   multi_step_output=False, residual_connections=True):
    """
    Create enhanced ConvLSTM model architecture with anti-damping features.
    
    Args:
        input_shape: (time_steps, height, width, channels)
        filters: list of filter numbers for ConvLSTM layers
        kernel_size: convolution kernel size
        dropout_rate: dropout rate for regularization (reduced for better forecasting)
        multi_step_output: whether to output multiple time steps
        residual_connections: whether to use residual connections
    
    Returns:
        Compiled Keras model
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is required for ConvLSTM models")
    
    # Use functional API for more flexibility
    inputs = Input(shape=input_shape)
    
    # First ConvLSTM layer
    x = ConvLSTM2D(
        filters=filters[0],
        kernel_size=kernel_size,
        padding='same',
        return_sequences=len(filters) > 1,
        activation='tanh',
        recurrent_activation='sigmoid',
        kernel_regularizer=l2(0.0005),  # Reduced regularization
        recurrent_regularizer=l2(0.0005),
        bias_regularizer=l2(0.0005)
    )(inputs)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Store for potential residual connection
    if residual_connections and len(filters) > 1:
        residual = x
    
    # Additional ConvLSTM layers
    for i, filter_count in enumerate(filters[1:], 1):
        return_seq = i < len(filters) - 1
        
        x = ConvLSTM2D(
            filters=filter_count,
            kernel_size=kernel_size,
            padding='same',
            return_sequences=return_seq,
            activation='tanh',
            recurrent_activation='sigmoid',
            kernel_regularizer=l2(0.0005),
            recurrent_regularizer=l2(0.0005),
            bias_regularizer=l2(0.0005)
        )(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        # Add residual connection if possible
        if residual_connections and i == 1 and x.shape[-1] == residual.shape[-1]:
            # Simple residual connection (requires same number of filters)
            if not return_seq:  # Only for the last layer that doesn't return sequences
                # Reduce residual to match
                residual_reduced = ConvLSTM2D(
                    filters=filter_count,
                    kernel_size=(1, 1),
                    padding='same',
                    return_sequences=False,
                    activation='linear'
                )(residual)
                x = tf.keras.layers.Add()([x, residual_reduced])
    
    # Additional processing layers for better representation
    x = Conv2D(
        filters=16,  # Intermediate representation
        kernel_size=(3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=l2(0.0005)
    )(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate * 0.5)(x)
    
    # Another processing layer
    x = Conv2D(
        filters=8,
        kernel_size=(1, 1),
        activation='relu',
        padding='same'
    )(x)
    x = BatchNormalization()(x)
    
    # Output layer
    if multi_step_output:
        # For multi-step output, we need to reshape and use TimeDistributed
        # This is more complex, for now use single-step approach
        outputs = Conv2D(
            filters=1,
            kernel_size=(1, 1),
            activation='linear',
            padding='same'
        )(x)
    else:
        outputs = Conv2D(
            filters=1,
            kernel_size=(1, 1),
            activation='linear',
            padding='same'
        )(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Enhanced optimizer settings
    optimizer = Adam(
        learning_rate=0.0015,  # Slightly higher learning rate
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        clipnorm=0.5  # Reduced gradient clipping
    )
    
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    return model


def _train_with_teacher_forcing(model, X_train, y_train, X_val, y_val,
                               epochs=100, batch_size=8, callbacks=None,
                               teacher_forcing_ratio=0.5, noise_std=0.005):
    """
    Train model with teacher forcing and noise injection for better robustness.
    
    Args:
        model: Keras model to train
        X_train, y_train: training data
        X_val, y_val: validation data
        epochs: number of training epochs
        batch_size: batch size
        callbacks: training callbacks
        teacher_forcing_ratio: ratio of using actual vs predicted values
        noise_std: standard deviation of noise to inject
    
    Returns:
        Training history
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is required")
    
    # Add noise to training data for robustness
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, X_train.shape)
        X_train_noisy = X_train + noise
        
        # Ensure we don't go too far from original data
        X_train_noisy = np.clip(X_train_noisy, 
                               X_train.mean() - 3*X_train.std(), 
                               X_train.mean() + 3*X_train.std())
    else:
        X_train_noisy = X_train
    
    # Standard training for now (teacher forcing is more complex for spatial data)
    # TODO: Implement proper spatial teacher forcing in future versions
    history = model.fit(
        X_train_noisy, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks if callbacks else [],
        verbose=1
    )
    
    return history


def _generate_future_predictions_enhanced(model, last_sequence, forecast_horizon, scaler_stats,
                                        data_array, sequence_length, noise_std=0.005,
                                        multi_step_model=False):
    """
    Generate future predictions with advanced anti-damping techniques.
    
    Args:
        model: trained ConvLSTM model
        last_sequence: last sequence from training data
        forecast_horizon: number of steps to forecast
        scaler_stats: normalization statistics
        data_array: original data array for statistical reference
        sequence_length: length of input sequences
        noise_std: noise standard deviation
        multi_step_model: whether model predicts multiple steps
    
    Returns:
        future_predictions: array of future predictions
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is required")
    
    future_predictions = []
    current_input = last_sequence.copy()
    
    # Calculate adaptive statistics from recent history
    recent_history = data_array.isel({data_array.dims[0]: slice(-sequence_length*3, None)})
    recent_mean = float(recent_history.mean())
    recent_std = float(recent_history.std())
    recent_trend = float(recent_history.isel({data_array.dims[0]: -1}).mean() - 
                        recent_history.isel({data_array.dims[0]: 0}).mean())
    
    # Seasonal/cyclical patterns
    seasonal_component = _extract_seasonal_component(data_array, sequence_length)
    
    for step in range(forecast_horizon):
        # Predict next step
        next_pred_scaled = model.predict(current_input, verbose=0)
        
        if step == 0:  # Debug info for first prediction only
            print(f"Debug - Future prediction shapes:")
            print(f"  current_input shape: {current_input.shape}")
            print(f"  next_pred_scaled shape: {next_pred_scaled.shape}")

        # Denormalize prediction
        next_pred = _denormalize_spatial_data_enhanced(next_pred_scaled, scaler_stats)
        
        # Anti-damping techniques
        
        # 1. Trend continuation (gradually weakening)
        trend_strength = max(0.1, 1.0 - step / forecast_horizon)
        trend_adjustment = recent_trend * trend_strength * 0.1
        next_pred += trend_adjustment
        
        # 2. Seasonal/cyclical adjustment
        if seasonal_component is not None and step < len(seasonal_component):
            seasonal_strength = max(0.2, 1.0 - step / (forecast_horizon * 2))
            next_pred += seasonal_component[step] * seasonal_strength
        
        # 3. Noise injection (controlled randomness)
        if noise_std > 0:
            # Adaptive noise - more early on, less later
            adaptive_noise_std = noise_std * (1.0 + step * 0.1)
            noise = np.random.normal(0, adaptive_noise_std, next_pred.shape)
            next_pred += noise
        
        # 4. Constraint to reasonable bounds (prevent extreme values)
        # Use recent statistics rather than global ones
        lower_bound = recent_mean - 4 * recent_std
        upper_bound = recent_mean + 4 * recent_std
        next_pred = np.clip(next_pred, lower_bound, upper_bound)
        
        # 5. Variability injection - prevent convergence to mean
        if step > 3:  # After a few steps
            # Add controlled variability based on recent history
            variability = recent_std * 0.05 * np.random.randn(*next_pred.shape)
            next_pred += variability
        
        future_predictions.append(next_pred[0])
        
        if step == 0:  # Debug info for first prediction only
            print(f"  next_pred shape: {next_pred.shape}")
            print(f"  next_pred[0] shape: {next_pred[0].shape}")

        # Update input sequence for next prediction
        # Renormalize the prediction for the next input
        next_pred_scaled_for_input = (next_pred - scaler_stats['mean']) / scaler_stats['std']
        
        # Sliding window update
        current_input = np.roll(current_input, -1, axis=1)
        current_input[0, -1] = next_pred_scaled_for_input[0]
        
        # Anti-feedback-loop measure: slight perturbation of input
        if step > 0 and step % 3 == 0:  # Every 3 steps
            perturbation_strength = min(0.02, noise_std * 2)
            perturbation = np.random.normal(0, perturbation_strength, current_input[0, -1].shape)
            current_input[0, -1] += perturbation
    
    return np.array(future_predictions)


def _extract_seasonal_component(data_array, sequence_length):
    """
    Extract seasonal/cyclical component from time series for forecasting.
    
    Args:
        data_array: original data array
        sequence_length: sequence length used for training
    
    Returns:
        seasonal_component: array of seasonal adjustments
    """
    try:
        # Simple approach: look for daily cycles in hourly data
        time_coords = data_array.coords[data_array.dims[0]]
        times = pd.to_datetime(time_coords.values)
        
        if len(times) < 48:  # Need at least 2 days for daily pattern
            return None
        
        # Calculate spatial mean for each time step
        spatial_means = data_array.mean(dim=[d for d in data_array.dims if d != data_array.dims[0]])
        
        # Detect if we have hourly data (common time step)
        time_diff = times[1] - times[0]
        if time_diff <= pd.Timedelta(hours=1):
            # Look for daily patterns
            hours = [t.hour for t in times]
            daily_pattern = {}
            
            for hour in range(24):
                hour_values = [spatial_means.values[i] for i, h in enumerate(hours) if h == hour]
                if hour_values:
                    daily_pattern[hour] = np.mean(hour_values)
            
            if len(daily_pattern) > 12:  # Have enough hours for pattern
                # Create seasonal component for next 24 hours
                last_hour = times[-1].hour
                seasonal_comp = []
                for i in range(min(24, sequence_length)):
                    target_hour = (last_hour + i + 1) % 24
                    if target_hour in daily_pattern:
                        # Difference from daily mean
                        overall_mean = np.mean(list(daily_pattern.values()))
                        seasonal_comp.append(daily_pattern[target_hour] - overall_mean)
                    else:
                        seasonal_comp.append(0.0)
                
                return np.array(seasonal_comp) if seasonal_comp else None
        
        return None
    except Exception:
        return None

# Missing functions for IDW compatibility
def _generate_seasonal_forecast(data_array, current_time, times_available, future_steps):
    """Simple seasonal forecast for IDW compatibility."""
    # Simple implementation - use last available data
    return data_array.isel({data_array.dims[0]: -1})

def _generate_damped_trend_forecast(data_array, future_steps):
    """Simple damped trend forecast for IDW compatibility."""
    # Simple implementation - use last available data
    return data_array.isel({data_array.dims[0]: -1})

def _generate_persistence_forecast(last_slice, future_steps):
    """Simple persistence forecast for IDW compatibility."""
    # Simple implementation - return the last slice
    return last_slice

def train_and_predict_convlstm_spatial(file_path, target_variable, sequence_length=5, forecast_horizon=24,
                                     filters=[32, 16], kernel_size=(3, 3), dropout_rate=0.1,
                                     epochs=100, batch_size=8, validation_split=0.2,
                                     teacher_forcing_ratio=0.5, noise_std=0.005,
                                     multi_step_training=True, residual_connections=True):
    """
    Train ConvLSTM model for spatial-temporal prediction with advanced anti-damping techniques.
    
    Args:
        file_path: path to NetCDF file
        target_variable: variable to predict
        sequence_length: number of past time steps to use
        forecast_horizon: number of future time steps to predict
        filters: list of filter numbers for ConvLSTM layers
        kernel_size: convolution kernel size
        dropout_rate: dropout rate (reduced default for better forecasting)
        epochs: training epochs
        batch_size: batch size
        validation_split: fraction for validation
        teacher_forcing_ratio: ratio of using actual vs predicted values during training (0-1)
        noise_std: standard deviation of noise to add during training for robustness
        multi_step_training: whether to train on multi-step sequences
        residual_connections: whether to use residual connections in the model
    
    Returns:
        Dictionary with predictions and metrics
    """
    if not TENSORFLOW_AVAILABLE:
        return {
            'error': 'TensorFlow not available. Please install tensorflow>=2.13.0',
            'model_name': 'ConvLSTM_Enhanced',
            'target_variable': target_variable
        }
    
    try:
        print("Loading data for ConvLSTM training...")
        with xr.open_dataset(file_path) as ds:
            print(f"Dataset dimensions: {list(ds.dims.keys())}")
            print(f"Dataset coordinates: {list(ds.coords.keys())}")
            print(f"Dataset variables: {list(ds.data_vars.keys())}")
            
            # Clean coordinates to avoid compatibility issues
            coords_to_remove = []
            for coord_name in ds.coords:
                if coord_name not in ds.dims and coord_name not in ['latitude', 'longitude', 'lat', 'lon']:
                    if coord_name in ['expver', 'number']:  # Common problematic coordinates
                        coords_to_remove.append(coord_name)
            
            if coords_to_remove:
                print(f"Removing problematic coordinates: {coords_to_remove}")
                ds = ds.drop_vars(coords_to_remove)
            
            time_dim = get_time_dim_name(ds)
            print(f"Time dimension found: {time_dim}")
            
            if target_variable not in ds:
                available_vars = list(ds.data_vars.keys())
                return {
                    'error': f'Target variable "{target_variable}" not found in dataset. Available variables: {available_vars}',
                    'model_name': 'ConvLSTM_Enhanced',
                    'target_variable': target_variable
                }
            
            data_array = ds[target_variable]
            
            # Get spatial coordinate names
            lat_name, lon_name = _get_spatial_coord_names(ds)
            print(f"Spatial coordinates found: lat={lat_name}, lon={lon_name}")
            
            if not lat_name or not lon_name:
                return {
                    'error': f'Could not find latitude/longitude coordinates. Available coords: {list(ds.coords.keys())}',
                    'model_name': 'ConvLSTM_Enhanced',
                    'target_variable': target_variable
                }
        
        print(f"Data shape: {data_array.shape}")
        print(f"Time dimension: {time_dim}")
        print(f"Spatial dimensions: {lat_name}, {lon_name}")
        
        # Enhanced sequence preparation with multi-step training
        print("Preparing sequences with anti-damping strategy...")
        if multi_step_training:
            # For multi-step training, create multiple training samples at different forecast horizons
            # But keep each sample as single-step to match model output shape
            print("Using multi-step training strategy with variable forecast horizons...")
            all_X, all_y, all_time_coords = [], [], []
            
            # Train on 1, 2, 3, 4, 5, 6 steps ahead to improve long-term forecasting
            max_train_steps = min(forecast_horizon, 6)
            for step_ahead in range(1, max_train_steps + 1):
                X_step, y_step, time_coords_step = _prepare_spatial_sequences_enhanced(
                    data_array, time_dim, sequence_length, forecast_horizon=step_ahead
                )
                if len(X_step) > 0:
                    # For multi-step training, we reshape y to single step
                    if step_ahead == 1:
                        y_step_reshaped = y_step
                    else:
                        # Take the last time step from multi-step target
                        y_step_reshaped = y_step[:, -1:, :, :, :]  # Take last time step
                        y_step_reshaped = y_step_reshaped.squeeze(axis=1)  # Remove time dimension
                    
                    all_X.extend(X_step)
                    all_y.extend(y_step_reshaped)
                    all_time_coords.extend(time_coords_step)
            
            X = np.array(all_X)
            y = np.array(all_y)
            time_coords = np.array(all_time_coords)
            print(f"Multi-step training: collected {len(X)} samples from {max_train_steps} forecast horizons")
        else:
            X, y, time_coords = _prepare_spatial_sequences_enhanced(
                data_array, time_dim, sequence_length, forecast_horizon=1
            )
        
        if len(X) == 0:
            raise ValueError("No valid sequences found. Data may contain too many NaN values.")
        
        print(f"Prepared {len(X)} sequences")
        print(f"Input shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        # Enhanced normalization strategy - preserve spatial structure
        print("Applying enhanced normalization...")
        X_scaled, y_scaled, scaler_stats = _normalize_spatial_data_enhanced(X, y)
        
        # Split data temporally (important for time series)
        split_idx = int(len(X_scaled) * (1 - validation_split))
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y_scaled[:split_idx], y_scaled[split_idx:]
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # Create enhanced model with anti-damping features
        print("Creating enhanced ConvLSTM model with anti-damping features...")
        input_shape = X_train.shape[1:]  # (time_steps, height, width, channels)
        print(f"Input shape for model: {input_shape}")
        
        model = _create_convlstm_model_enhanced(
            input_shape, filters, kernel_size, dropout_rate, 
            multi_step_output=False,  # Always single-step output with our approach
            residual_connections=residual_connections
        )
        print("Enhanced ConvLSTM model created successfully")
        
        print("Model summary:")
        model.summary()
        
        # Enhanced callbacks with better learning rate scheduling
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True, min_delta=1e-6),
            ReduceLROnPlateau(patience=8, factor=0.7, min_lr=1e-7, verbose=1)
        ]
        
        # Enhanced training with teacher forcing
        print("Training model with teacher forcing and noise injection...")
        history = _train_with_teacher_forcing(
            model, X_train, y_train, X_val, y_val,
            epochs=epochs, batch_size=batch_size, callbacks=callbacks,
            teacher_forcing_ratio=teacher_forcing_ratio, noise_std=noise_std
        )
          # Make predictions on validation set
        print("Making predictions...")
        y_pred_scaled = model.predict(X_val)
        print(f"Debug - Model prediction output shape: {y_pred_scaled.shape}")
        print(f"Debug - X_val shape: {X_val.shape}")
        print(f"Debug - y_val shape: {y_val.shape}")

        # Inverse transform predictions using enhanced denormalization
        y_pred = _denormalize_spatial_data_enhanced(y_pred_scaled, scaler_stats)
        y_val_original = _denormalize_spatial_data_enhanced(y_val, scaler_stats)
        
        # Calculate metrics
        y_pred_flat = y_pred.flatten()
        y_val_flat = y_val_original.flatten()
        
        # Remove NaN values for metrics
        valid_mask = ~(np.isnan(y_pred_flat) | np.isnan(y_val_flat))
        y_pred_clean = y_pred_flat[valid_mask]
        y_val_clean = y_val_flat[valid_mask]
        
        if len(y_pred_clean) > 0:
            rmse = np.sqrt(mean_squared_error(y_val_clean, y_pred_clean))
            mae = mean_absolute_error(y_val_clean, y_pred_clean)
            r2 = r2_score(y_val_clean, y_pred_clean)
        else:
            rmse = mae = r2 = np.nan
        
        # Generate future forecasts with advanced anti-damping strategy
        print(f"Generating {forecast_horizon} future predictions with advanced anti-damping...")
        
        # Use the last sequence from original data (not scaled validation data)
        last_sequence_idx = len(X_scaled) - 1
        last_sequence = X_scaled[last_sequence_idx:last_sequence_idx+1]
        
        future_predictions = _generate_future_predictions_enhanced(
            model, last_sequence, forecast_horizon, scaler_stats,
            data_array, sequence_length, noise_std,
            multi_step_model=False  # Always single-step with our approach
        )
        print(f"Debug - Future predictions shape after generation: {future_predictions.shape}")

        # Create time coordinates for future predictions
        last_time = pd.to_datetime(time_coords[-1])
        if len(time_coords) > 1:
            time_step = pd.to_datetime(time_coords[-1]) - pd.to_datetime(time_coords[-2])
        else:
            time_step = pd.Timedelta(hours=1)  # Default 1 hour
        future_times = [last_time + (i + 1) * time_step for i in range(forecast_horizon)]
        
        # Create xarray DataArrays for results
        spatial_coords = {
            lat_name: data_array.coords[lat_name],
            lon_name: data_array.coords[lon_name]
        }
        
        # Validation predictions
        val_times = [time_coords[split_idx + i] for i in range(len(y_val))]
        
        # Ensure shapes match for creating DataArrays
        # ConvLSTM Enhanced model outputs (samples, height, width, channels=1)
        print(f"Debug - Original prediction shapes:")
        print(f"  y_pred shape before squeeze: {y_pred.shape}")
        print(f"  y_val_original shape before squeeze: {y_val_original.shape}")
        
        # Handle ConvLSTM output: (samples, height, width, 1) -> (samples, height, width)
        if y_pred.ndim == 4 and y_pred.shape[-1] == 1:
            y_pred_single = y_pred[:, :, :, 0]  # Remove channel dimension
        else:
            y_pred_single = y_pred.squeeze()
        
        if y_val_original.ndim == 4 and y_val_original.shape[-1] == 1:
            y_val_single = y_val_original[:, :, :, 0]  # Remove channel dimension
        else:
            y_val_single = y_val_original.squeeze()
            
        print(f"Debug - Processed shapes:")
        print(f"  y_pred_single shape: {y_pred_single.shape}")
        print(f"  y_val_single shape: {y_val_single.shape}")
        
        predicted_val = xr.DataArray(
            y_pred_single,
            dims=[time_dim, lat_name, lon_name],
            coords={time_dim: val_times, **spatial_coords}
        )
        
        actual_val = xr.DataArray(
            y_val_single,
            dims=[time_dim, lat_name, lon_name],
            coords={time_dim: val_times, **spatial_coords}
        )
        
        # Future predictions - squeeze out channel dimension if present
        # future_predictions shape: (forecast_horizon, height, width, 1) -> remove channel to get (forecast_horizon, height, width)
        if future_predictions.ndim == 4 and future_predictions.shape[-1] == 1:
            future_predictions_squeezed = future_predictions[:, :, :, 0]  # Remove only channel dimension
        else:
            future_predictions_squeezed = future_predictions
        
        # Debug: Print shapes for troubleshooting
        print(f"Debug - Shape info:")
        print(f"  y_pred_single shape: {y_pred_single.shape}")
        print(f"  y_val_single shape: {y_val_single.shape}")
        print(f"  future_predictions original shape: {future_predictions.shape}")
        print(f"  future_predictions_squeezed shape: {future_predictions_squeezed.shape}")
        
        predicted_future = xr.DataArray(
            future_predictions_squeezed,
            dims=[time_dim, lat_name, lon_name],
            coords={time_dim: future_times, **spatial_coords}
        )
        
        # Combine validation and future predictions
        all_pred_times = val_times + future_times
        
        # Ensure consistent shapes before concatenation
        print(f"Debug - Concatenation shapes:")
        print(f"  y_pred_single shape: {y_pred_single.shape}")
        print(f"  future_predictions_squeezed shape: {future_predictions_squeezed.shape}")
        
        # Verify shapes are compatible
        if y_pred_single.ndim != future_predictions_squeezed.ndim:
            raise ValueError(f"Shape mismatch: y_pred_single has {y_pred_single.ndim} dimensions, "
                           f"future_predictions_squeezed has {future_predictions_squeezed.ndim} dimensions")
        
        all_predictions = np.concatenate([y_pred_single, future_predictions_squeezed], axis=0)
        
        predicted_all = xr.DataArray(
            all_predictions,
            dims=[time_dim, lat_name, lon_name],
            coords={time_dim: all_pred_times, **spatial_coords}
        )
        
        # Create actual data with NaN for future (to match prediction structure)
        future_nan = np.full(future_predictions_squeezed.shape, np.nan)
        
        # Debug: Check shapes before concatenation
        print(f"Debug - Actual data concatenation shapes:")
        print(f"  y_val_single shape: {y_val_single.shape}")
        print(f"  future_nan shape: {future_nan.shape}")
        
        # Ensure both arrays have the same number of dimensions
        if y_val_single.ndim != future_nan.ndim:
            # If dimensions don't match, try to make them match
            if y_val_single.ndim == 3 and future_nan.ndim == 4:
                # Remove the last dimension if it's size 1
                if future_nan.shape[-1] == 1:
                    future_nan = future_nan[:, :, :, 0]
                    print(f"  future_nan shape after squeeze: {future_nan.shape}")
            elif y_val_single.ndim == 4 and future_nan.ndim == 3:
                # Add a dimension to future_nan
                future_nan = future_nan[..., np.newaxis]
                print(f"  future_nan shape after expand: {future_nan.shape}")
        
        # Verify shapes are compatible
        if y_val_single.ndim != future_nan.ndim:
            raise ValueError(f"Shape mismatch: y_val_single has {y_val_single.ndim} dimensions, "
                           f"future_nan has {future_nan.ndim} dimensions")

        all_actual = np.concatenate([y_val_single, future_nan], axis=0)
        
        actual_all = xr.DataArray(
            all_actual,
            dims=[time_dim, lat_name, lon_name],
            coords={time_dim: all_pred_times, **spatial_coords}
        )
        
        return {
            'actual_spatial_all_times': actual_all,
            'predicted_spatial_all_times': predicted_all,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'model_name': 'ConvLSTM_Enhanced',
            'target_variable': target_variable,
            'time_dim': time_dim,
            'n_times': len(all_pred_times),
            'n_actual_times': len(val_times),
            'forecast_horizon': forecast_horizon,
            'times_available': all_pred_times,
            'validation_method': 'temporal_split',
            'training_history': history.history if hasattr(history, 'history') else history,
            'hyperparameters': {
                'sequence_length': sequence_length,
                'filters': filters,
                'kernel_size': kernel_size,
                'dropout_rate': dropout_rate,
                'epochs': len(history.history['loss']) if hasattr(history, 'history') else epochs,
                'batch_size': batch_size,
                'validation_split': validation_split,
                'teacher_forcing_ratio': teacher_forcing_ratio,
                'noise_std': noise_std,
                'multi_step_training': multi_step_training,
                'residual_connections': residual_connections
            }
        }
        
    except Exception as e:
        import traceback
        error_details = f'ConvLSTM training failed: {str(e)}\n{traceback.format_exc()}'
        print(error_details)
        return {
            'error': error_details,
            'model_name': 'ConvLSTM_Enhanced',
            'target_variable': target_variable
        }
