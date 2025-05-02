import requests
import time
import os
import json
from datetime import datetime, timedelta

YEARS_OF_DATA = 20 

NOAA_TOKEN = 'znkMxUqZUSIRgcBAfybGeAvietAtVWEu'
BASE_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2"

STATIONS = {
    "Steamboat": "GHCND:USC00057936",
    "Vail": "GHCND:USC00058575", 
    "Breckenridge": "GHCND:USC00050909", 
    "Winter Park": "GHCND:USC00059175",
    "Copper Mountain": "GHCND:USC00051959"
}

def get_noaa_data_for_year(station_id, start_date, end_date):
    """Fetch snowfall data from NOAA API for a specific station and date range within one year."""
    all_data = []
    offset = 0
    limit = 1000
    max_retries = 2 
    retry_delay = 2 
    
    while True:
        for attempt in range(max_retries):
            try:
                params = {
                    "datasetid": "GHCND",
                    "stationid": station_id,
                    "startdate": start_date,
                    "enddate": end_date,
                    "datatypeid": "SNOW",
                    "limit": limit,
                    "offset": offset
                }
                
                headers = {"token": NOAA_TOKEN}
                response = requests.get(f"{BASE_URL}/data", params=params, headers=headers)
                
                if response.status_code == 429: 
                    print(f"Rate limit reached, waiting {retry_delay * (attempt + 1)} seconds...")
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                    
                if response.status_code != 200:
                    print(f"Error fetching data: {response.status_code}")
                    print(f"Response: {response.text}")
                    time.sleep(retry_delay)
                    continue
                    
                data = response.json()
                break
            except Exception as e:
                print(f"Error on attempt {attempt+1}: {str(e)}")
                if attempt == max_retries - 1:
                    print(f"Failed to fetch data after {max_retries} attempts: {str(e)}")
                    return None
                time.sleep(retry_delay * (attempt + 1))
        
        if 'results' not in data or not data['results']:
            break
            
        all_data.extend(data['results'])
        
        if len(data['results']) < limit:
            break
            
        offset += limit
        time.sleep(0.5) 
    
    return all_data

def get_noaa_data(station_id, start_date, end_date):
    """Fetch snowfall data from NOAA API for multiple years."""
    all_data = []
    current_start = datetime.strptime(start_date, "%Y-%m-%d")
    end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
    
    while current_start < end_datetime:
        current_end = min(current_start + timedelta(days=364), end_datetime)
        
        print(f"Fetching {current_start.year} data...")
        year_data = get_noaa_data_for_year(
            station_id,
            current_start.strftime("%Y-%m-%d"),
            current_end.strftime("%Y-%m-%d")
        )
        
        if year_data:
            all_data.extend(year_data)
        else:
            print(f"Warning: No data available for {current_start.year}")
            
        current_start = current_end + timedelta(days=1)
        time.sleep(1) 
    
    return {"results": all_data} if all_data else None

def main():
    current_date = datetime.now()
    current_month = current_date.month
    current_year = current_date.year
    
    if current_month >= 7:
        end_year = current_year + 1
        start_year = current_year - YEARS_OF_DATA
    else:
        end_year = current_year
        start_year = current_year - YEARS_OF_DATA - 1
    
    start_date = f"{start_year}-07-01"
    end_date = f"{end_year}-06-30"
    
    data_dir = 'snowfall_data'
    os.makedirs(data_dir, exist_ok=True)
    
    date_info = {
        'start_date': start_date,
        'end_date': end_date,
        'download_date': current_date.strftime("%Y-%m-%d")
    }
    
    with open(f'{data_dir}/date_info.json', 'w') as f:
        json.dump(date_info, f)
    
    for resort, station_id in STATIONS.items():
        print(f"\nDownloading data for {resort}...")
        raw_data = get_noaa_data(station_id, start_date, end_date)
        
        if raw_data and 'results' in raw_data:
            print(f"Successfully downloaded {len(raw_data['results'])} records for {resort}")
            
            with open(f'{data_dir}/{resort.lower().replace(" ", "_")}_raw.json', 'w') as f:
                json.dump(raw_data, f)
        else:
            print(f"No data available for {resort}")
    
    print("\nData download complete! Data saved to the snowfall_data directory.")

if __name__ == "__main__":
    main()
