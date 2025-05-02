import os
import tarfile
import requests
from datetime import datetime, timedelta
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup



def download_mean_layer_temp(save_dir):
    filename = "Mean-Layer-Temperature-NOAA_v05r00_TLS_S197812_E202503_C20250402.nc"
    url = f"https://www.ncei.noaa.gov/data/mean-layer-temperature-noaa/access/" + filename
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    if os.path.exists(save_path):
        print(f"temp(1/1) File already exists: {save_path}")
        return

    try:
        print(f"Downloading {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"temp(1/1) Saved to {save_path}")

    except requests.HTTPError as e:
        print(f"Failed to download {url}: {e}")



# Download humidity data
#   monthly, 2.5deg resolution, brightness_temperature ()
def download_upper_trop_humidity(save_dir, start_year=1998, end_year=2022):
    files_downloaded = 0
    num_files = end_year-start_year + 1
    year = start_year

    while year <= end_year:
        filename = f"ir-sounder-upper-trop-humidity-bt_v04r00_monthlygrid_s{year}0101_e{year}1231_c20230801.nc"
        url = f"https://www.ncei.noaa.gov/data/ir-sounder-upper-trop-humidity-bt/access/" + filename

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)

        if os.path.exists(save_path):
            files_downloaded += 1
            print(f"humidity({files_downloaded}/{num_files}) File already exists: {save_path}")
            year += 1
            continue

        try:
            print(f"Downloading {url}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            files_downloaded += 1
            print(f"humidity({files_downloaded}/{num_files}) Saved to {save_path}")

        except requests.HTTPError as e:
            print(f"Failed to download {url}: {e}")

        year += 1


# Download cmorph .zip files for a date range
#   daily, 0.25deg resolution, mm/day precipitation rate

def get_cmorph_tar(year, month):
    base_url = f"https://www.ncei.noaa.gov/data/cmorph-high-resolution-global-precipitation-estimates/archive/daily/0.25deg/{year}/{int(month):02d}/"

    # Step 1: Get HTML page contents
    resp = requests.get(base_url)
    if resp.status_code != 200:
        print(f"Failed to access {base_url} (status {resp.status_code})")
        return

    # Step 2: Parse HTML and find the .tar link
    soup = BeautifulSoup(resp.text, "html.parser")
    tar_links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].endswith('.tar')]

    if not tar_links:
        print("No .tar files found.")
        return

    # Usually only one .tar file per month
    tar_name = tar_links[0]
    full_url = base_url + tar_name
    return full_url

def download_cmorph_daily(start_date, end_date, save_dir):
    base_url = "https://www.ncei.noaa.gov/data/cmorph-high-resolution-global-precipitation-estimates/archive/daily/0.25deg"
    date = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    files_downloaded = 0
    num_files = (end.year - date.year) * 12 + (end.month - date.month) + 1

    os.makedirs(save_dir, exist_ok=True)

    while date <= end:
        year = date.strftime("%Y")
        month = date.strftime("%m")
        start_str = f"{year}{month}01"
        # Get the last day of the month
        next_month = date.replace(day=28) + timedelta(days=4)
        last_day = (next_month - timedelta(days=next_month.day)).day
        end_str = f"{year}{month}{last_day:02d}"

        filename = f"cmorph_v1.0_0.25deg_daily_s{start_str}_e{end_str}_c20180808.tar"
        url = get_cmorph_tar(year, month)
        save_path = os.path.join(save_dir, filename)

        if os.path.exists(save_path):
            files_downloaded += 1
            print(f"precip({files_downloaded}/{num_files}) File already exists: {save_path}")
            date = (date.replace(day=28) + timedelta(days=4)).replace(day=1)
            continue

        try:
            print(f"Downloading {url}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            files_downloaded += 1
            print(f"precip({files_downloaded}/{num_files}) Saved to {save_path}")

        except requests.HTTPError as e:
            print(f"Failed to download {url}: {e}")

        # Move to first day of next month
        date = (date.replace(day=28) + timedelta(days=4)).replace(day=1)

def extract_nc_files(data_folder):
    for file in os.listdir(data_folder):
        if file.endswith(".tar"):
            filepath = os.path.join(data_folder, file)
            with tarfile.open(filepath, "r") as tar:
                for member in tar.getmembers():
                    target_path = os.path.join(data_folder, member.name)
                    if not os.path.exists(target_path):
                        try:
                            tar.extract(member, path=data_folder)
                        except PermissionError as e:
                            print(f"Permission denied extracting {member.name}: {e}")



def download_and_extract_data():
    download_mean_layer_temp("./data/temperature")
    download_upper_trop_humidity("./data/humidity")
    download_cmorph_daily("1998-01-01", "2022-12-31", "./data/cmorph")
    extract_nc_files("./data/cmorph")

def extract_data_only():
    extract_nc_files("./data/cmorph")