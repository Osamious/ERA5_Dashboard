"""
NetCDF file utilities for the ERA5 Dashboard.
"""

import xarray as xr
import pandas as pd
import numpy as np
import json
from .helpers import convert_time_array_to_local_timezone

def save_prediction_to_netcdf(results, include_test, include_future, variable_map):
    """
    Saves prediction results to a NetCDF file in memory.

    Args:
        results (dict): The dictionary containing prediction results.
        include_test (bool): Whether to include the test set predictions.
        include_future (bool): Whether to include the future forecast.
        variable_map (dict): The map of variable names and metadata.

    Returns:
        bytes: The NetCDF file content as bytes.
    """
    target_var = results['target_variable']
    predicted_var_name = f"{target_var}_predicted"
    time_dim = results['time_dim']    # Combine time coordinates and data (convert to local timezone)
    all_times = []
    all_preds = []

    if include_test:
        test_time_index = results['df'][time_dim].iloc[-len(results['y_test']):]
        # Convert to local timezone
        test_times_local = convert_time_array_to_local_timezone(test_time_index.values)
        all_times.extend(test_times_local)
        all_preds.extend(results['y_pred_test'])

    if include_future and 'future_datetimes' in results:
        # Convert future times to local timezone
        future_times_local = convert_time_array_to_local_timezone(results['future_datetimes'])
        all_times.extend(future_times_local)
        all_preds.extend(results['y_pred_future'])

    if not all_times:
        raise ValueError("No data selected for download.")

    # Create a combined DataFrame to handle sorting and duplicate removal
    combined_df = pd.DataFrame({time_dim: all_times, predicted_var_name: all_preds}).set_index(time_dim)
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')].sort_index()

    # Create the xarray Dataset
    ds_pred = xr.Dataset(
        {
            predicted_var_name: ([time_dim], combined_df[predicted_var_name].values)
        },
        coords={
            time_dim: combined_df.index,
            'latitude': results['latitude'],
            'longitude': results['longitude']
        }
    )    # Add metadata to conform to ERA5 standards
    ds_pred.attrs['history'] = f"Created {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} by ERA5 Dashboard. Model: {results['model_name']}."
    ds_pred.attrs['source'] = "Machine learning prediction based on ERA5 data."
    ds_pred.attrs['timezone'] = "UTC+3 (local time conversion applied)"
    ds_pred.attrs['model_parameters'] = json.dumps(results.get('model_params', {}))

    var_info = variable_map.get(target_var, {})
    ds_pred[predicted_var_name].attrs['long_name'] = f"Predicted {var_info.get('name', target_var)}"
    ds_pred[predicted_var_name].attrs['units'] = var_info.get('units', 'unknown')
    
    # Add lat/lon attributes
    ds_pred['latitude'].attrs['long_name'] = 'latitude'
    ds_pred['latitude'].attrs['units'] = 'degrees_north'
    ds_pred['longitude'].attrs['long_name'] = 'longitude'
    ds_pred['longitude'].attrs['units'] = 'degrees_east'    # Return bytes directly - use default engine for maximum compatibility
    return ds_pred.to_netcdf()

def save_spatial_prediction_to_netcdf(results, variable_map, include_test=True, include_future=True):
    """
    Saves spatial prediction results to a NetCDF file in ERA5-compatible format.

    Args:
        results (dict): The dictionary containing spatial prediction results.
        variable_map (dict): The map of variable names and metadata.
        include_test (bool): Whether to include the test set predictions.
        include_future (bool): Whether to include the future forecast.

    Returns:
        bytes: The NetCDF file content as bytes.
    """
    target_var = results['target_variable']
    predicted_var_name = f"{target_var}_predicted"
    
    # Get spatial dimensions
    lats = results['lats']
    lons = results['lons']
    
    # Combine time coordinates and predictions based on user selection
    all_times = []
    all_predictions = []
    
    if include_test and 'y_test' in results and 'y_pred_test' in results:
        # For spatial test predictions, we need to reconstruct the spatial grid
        # This is a simplified approach - in practice, test predictions might need more complex handling
        test_predictions = results['y_pred_test']
        if hasattr(test_predictions, '__len__') and len(test_predictions) > 0:
            # Create a single time step for test predictions (using the last available time)
            if 'future_times' in results and len(results['future_times']) > 0:
                if hasattr(results['future_times'], 'values'):
                    first_time = results['future_times'].values[0]
                else:
                    first_time = results['future_times'][0]
                # Make test time slightly before future predictions
                test_time = pd.Timestamp(first_time) - pd.Timedelta(hours=1)
            else:
                test_time = pd.Timestamp.now()
            
            # For spatial predictions, test data is typically aggregated - create a representative spatial field
            if len(test_predictions) == len(lats) * len(lons):
                # Test predictions are already in spatial format
                test_spatial = test_predictions.reshape(len(lats), len(lons))
            else:
                # Create a simplified spatial representation
                test_spatial = np.full((len(lats), len(lons)), np.mean(test_predictions))
            
            all_times.append(test_time)
            all_predictions.append(test_spatial[np.newaxis, :, :])  # Add time dimension
    
    if include_future and 'future_times' in results and 'future_predictions' in results:
        future_times = results['future_times']
        future_predictions = results['future_predictions']
        
        # Convert future times to local timezone
        if hasattr(future_times, 'values'):
            future_times_local = convert_time_array_to_local_timezone(future_times.values)
        else:
            future_times_local = convert_time_array_to_local_timezone(future_times)
        
        all_times.extend(future_times_local)
        all_predictions.append(future_predictions)
    
    if not all_times:
        raise ValueError("No data selected for download.")
    
    # Concatenate all predictions along time dimension
    if len(all_predictions) > 1:
        combined_predictions = np.concatenate(all_predictions, axis=0)
    else:
        combined_predictions = all_predictions[0]
    
    # Ensure times are pandas DatetimeIndex for proper xarray handling
    if not isinstance(all_times, pd.DatetimeIndex):
        all_times = pd.DatetimeIndex(all_times)
      # Create the xarray Dataset in ERA5-compatible format
    # Ensure dimensions are in the correct order: (time, latitude, longitude)
    ds_spatial = xr.Dataset(
        {
            predicted_var_name: (['time', 'latitude', 'longitude'], combined_predictions)
        },
        coords={
            'time': all_times,
            'latitude': lats,
            'longitude': lons
        }
    )
    
    # Add comprehensive metadata to conform to ERA5 standards
    ds_spatial.attrs['Conventions'] = 'CF-1.6'
    ds_spatial.attrs['title'] = f"Spatial prediction of {target_var} using {results['model_name']}"
    ds_spatial.attrs['institution'] = "ERA5 Dashboard - Machine Learning Prediction System"
    ds_spatial.attrs['source'] = "Machine learning spatial prediction based on ERA5 reanalysis data"
    ds_spatial.attrs['history'] = f"Created {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} by ERA5 Dashboard. Model: {results['model_name']}."
    ds_spatial.attrs['references'] = "Based on ERA5 reanalysis data from ECMWF"
    ds_spatial.attrs['comment'] = f"Spatial predictions generated using {results['model_name']} with {len(lats)}x{len(lons)} grid resolution"
    ds_spatial.attrs['timezone'] = "UTC+3 (local time conversion applied)"
    
    # Add model performance metrics as global attributes
    ds_spatial.attrs['model_rmse'] = float(results['rmse'])
    ds_spatial.attrs['model_mae'] = float(results['mae'])
    ds_spatial.attrs['model_r2'] = float(results['r2'])
    ds_spatial.attrs['training_time_seconds'] = float(results['training_time'])
    
    # Add model-specific parameters
    if 'model_params' in results:
        ds_spatial.attrs['model_parameters'] = json.dumps(results['model_params'])
    
    # Add variable metadata
    var_info = variable_map.get(target_var, {})
    ds_spatial[predicted_var_name].attrs['long_name'] = f"Predicted {var_info.get('name', target_var)}"
    ds_spatial[predicted_var_name].attrs['units'] = var_info.get('units', 'unknown')
    ds_spatial[predicted_var_name].attrs['standard_name'] = var_info.get('standard_name', predicted_var_name)
    ds_spatial[predicted_var_name].attrs['cell_methods'] = 'time: forecasted'
    ds_spatial[predicted_var_name].attrs['grid_mapping'] = 'latitude_longitude'
    
    # Add coordinate metadata for ERA5 compatibility
    ds_spatial['latitude'].attrs['long_name'] = 'latitude'
    ds_spatial['latitude'].attrs['units'] = 'degrees_north'
    ds_spatial['latitude'].attrs['standard_name'] = 'latitude'
    ds_spatial['latitude'].attrs['axis'] = 'Y'
    
    ds_spatial['longitude'].attrs['long_name'] = 'longitude'
    ds_spatial['longitude'].attrs['units'] = 'degrees_east'
    ds_spatial['longitude'].attrs['standard_name'] = 'longitude'
    ds_spatial['longitude'].attrs['axis'] = 'X'    
    ds_spatial['time'].attrs['long_name'] = 'time'
    ds_spatial['time'].attrs['standard_name'] = 'time'
    ds_spatial['time'].attrs['axis'] = 'T'
    # Note: calendar attribute is automatically handled by xarray for datetime coordinates
    # Add grid mapping information
    ds_spatial.attrs['grid_mapping_name'] = 'latitude_longitude'
    ds_spatial.attrs['longitude_of_prime_meridian'] = 0.0
    ds_spatial.attrs['semi_major_axis'] = 6378137.0
    ds_spatial.attrs['inverse_flattening'] = 298.257223563
    
    # Return bytes directly for maximum compatibility
    # Let xarray handle encoding automatically to avoid conflicts
    return ds_spatial.to_netcdf()
