import os
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from scipy.ndimage import zoom

class Event:
    '''
    Stores a 3D array of tuples (cmorph, humidity, surface_temp)
    data[date][lat][long]
    0.25 deg granularity, 1 measurement per day
    '''
    def __init__(
        self, 
        lat_min, 
        lat_max, 
        lon_min, 
        lon_max,
        start_date, 
        end_date,
        data_directory: str = "./data"
    ):
        self._validate_dates(start_date, end_date)

        # Convert longitudes to 0–360
        if lon_min < 0: lon_min += 360
        if lon_max < 0: lon_max += 360

        self.data, self.dates, self.lats, self.lons = self._load_data(
            lat_min, 
            lat_max, 
            lon_min, 
            lon_max,
            start_date,
            end_date,
            data_directory
        )
        print(f"Initialized event with:\ndates: {start_date} to {end_date}\nlon bounds ({lon_min}, {lon_max})\nlat bounds ({lat_min}, {lat_max}")
        self.out_string = f"Event(start_date={start_date}, end_date={end_date}, lat_min={lat_min}, lat_max={lat_max}, lon_min={lon_min}, lon_max={lon_max})"
        
    def __repr__(self):
        return self.out_string

    def _validate_dates(self, start, end):
        fmt = "%Y-%m-%d"
        min_date = datetime.strptime("1998-01-01", fmt)
        max_date = datetime.strptime("2022-12-31", fmt)
        start_dt = datetime.strptime(start, fmt)
        end_dt = datetime.strptime(end, fmt)

        if not (min_date <= start_dt <= max_date):
            raise ValueError(f"Start date must be between 1998-01-01 and 2022-12-31")
        if not (min_date <= end_dt <= max_date):
            raise ValueError(f"End date must be between 1998-01-01 and 2022-12-31")
        if start_dt > end_dt:
            raise ValueError("Start date must be earlier than or equal to end date")
        
    def _get_daily_surface_temp(self, date, target_shape, data_directory):
        """Loads the surface temperature file, interpolates to target (lat, lon) shape."""
        file_name = "temperature/Mean-Layer-Temperature-NOAA_v05r00_TLS_S197812_E202503_C20250402.nc"
        file_path = os.path.join(data_directory, file_name)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Surface temperature file not found: {file_path}")

        with xr.open_dataset(file_path) as ds:
            temp = ds["tcdr_MSU_AMSUA_ATMS_TLS"].squeeze()  # (month, lat, lon)

            # Find closest month
            time_values = ds["time"].values
            target_month = np.datetime64(date.strftime("%Y-%m"))
            closest_idx = np.argmin(np.abs(time_values - target_month))

            temp_slice = temp[closest_idx].values  # (lat, lon)

            # Zoom to match CMORPH slice size
            zoom_factors = (
                target_shape[0] / temp_slice.shape[0],
                target_shape[1] / temp_slice.shape[1],
            )
            resized_temp = zoom(temp_slice, zoom_factors, order=1)
            return resized_temp
        
    def _get_daily_humidity(self, date, lat_range, lon_range, data_directory):
        """Loads the humidity file for the given year and slices/interpolates it to the requested lat/lon range."""
        file_name = f"humidity/ir-sounder-upper-trop-humidity-bt_v04r00_monthlygrid_s{date.strftime('%Y')}0101_e{date.strftime('%Y')}1231_c20230801.nc"
        file_path = os.path.join(data_directory, file_name)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Humidity file not found: {file_path}")

        with xr.open_dataset(file_path) as ds:
            humidity = ds["ch12"].squeeze()  # shape: (months, lat, lon)
            h_lat = ds["lat"].values
            h_lon = ds["lon"].values

            # Pick closest time slice (month)
            target_month = np.datetime64(date.strftime("%Y-%m"))
            time_values = ds["time"].values
            closest_idx = np.argmin(np.abs(time_values - target_month))
            humidity_slice = humidity[closest_idx].values  # shape (lat, lon)

            # Find lat/lon indices using np.ix_
            lat_min, lat_max = lat_range
            lon_min, lon_max = lon_range

            padding = 1.25  # half the grid spacing
            lat_inds = np.where((h_lat >= lat_min - padding) & (h_lat <= lat_max + padding))[0]
            lon_inds = np.where((h_lon >= lon_min - padding) & (h_lon <= lon_max + padding))[0]

            if len(lat_inds) == 0 or len(lon_inds) == 0:
                raise ValueError("No matching lat/lon indices found in humidity grid")

            sliced = humidity_slice[np.ix_(lat_inds, lon_inds)]  # safe slice
            return sliced

    def _load_data(self, lat_min, lat_max, lon_min, lon_max, start_date, end_date, data_directory):
        data_stack = []
        dates = []
        lats = lons = None

        # Parse date range
        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            file_name = f"cmorph/CMORPH_V1.0_ADJ_0.25deg-DLY_00Z_{current.strftime('%Y%m%d')}.nc"
            file_path = os.path.join(data_directory, file_name)

            if os.path.exists(file_path):
                try:
                    print(f"Processing file: {file_path}")
                    with xr.open_dataset(file_path) as ds:
                        if lats is None or lons is None:
                            all_lats = ds["lat"].values
                            all_lons = ds["lon"].values
                            lat_mask = (all_lats >= lat_min) & (all_lats <= lat_max)
                            lon_mask = (all_lons >= lon_min) & (all_lons <= lon_max)
                            lats = all_lats[lat_mask]
                            lons = all_lons[lon_mask]
                            self.lat_mask = lat_mask
                            self.lon_mask = lon_mask

                        cmorph = ds["cmorph"].squeeze().values  # (lat, lon)
                        sliced = cmorph[self.lat_mask][:, self.lon_mask]  # (lat, lon)
                        humidity_raw = self._get_daily_humidity(current, (lat_min, lat_max), (lon_min, lon_max), data_directory)
                        # Resize humidity to match CMORPH resolution
                        zoom_factors = (
                            sliced.shape[0] / humidity_raw.shape[0],
                            sliced.shape[1] / humidity_raw.shape[1],
                        )
                        humidity = zoom(humidity_raw, zoom_factors, order=1)  # bilinear interp
                        temperature = self._get_daily_surface_temp(current, sliced.shape, data_directory)

                        tuple_grid = np.empty_like(sliced, dtype=object)
                        for i in range(sliced.shape[0]):
                            for j in range(sliced.shape[1]):
                                tuple_grid[i, j] = (float(sliced[i, j]), float(humidity[i, j]), float(temperature[i, j]))  # Add surface_temp later if needed

                        data_stack.append(tuple_grid)
                        dates.append(date_str)
                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")
            else:
                print(f"Missing file: {file_path}")

            current += timedelta(days=1)

        if not data_stack:
            raise RuntimeError("No data loaded for given time and spatial bounds.")

        data = np.stack(data_stack, axis=0)  # (time, lat, lon)
        return data, dates, lats, lons
    
    def get_data(self):
        """Returns the 3D data array of tuples (precip, surf_temp, humidity): shape (time, lat, lon)."""
        return self.data

    def get_dates(self):
        """Returns the list of date strings corresponding to the time dimension."""
        return self.dates

    def get_lats(self):
        """Returns the 1D array of latitudes used in the spatial slice."""
        return self.lats

    def get_lons(self):
        """Returns the 1D array of longitudes used in the spatial slice."""
        return self.lons