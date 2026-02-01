import os
import argparse
import requests
import zipfile
import tarfile
from tqdm import tqdm
import gdown
import shutil

# Configuration
DATA_ROOT = os.path.join(os.getcwd(), 'data', 'raw')
OSCD_URL = "https://rc.eodatabase.ch/oscd/OSCD.zip"
# LEVIR-CD Official Drive ID (Example placeholder, will update if found)
# Often 1p55bwy5k82437y is cited but might be old.
LEVIR_CD_DRIVE_ID = "1p55bwy5k82437y" 

def download_file(url, dest_path):
    if os.path.exists(dest_path):
        print(f"{dest_path} already exists. Skipping.")
        return
    
    print(f"Downloading {url} to {dest_path}...")
    try:
        response = requests.get(url, stream=True, verify=False) # Verify false for some academic sites
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True) as bar:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    except Exception as e:
        print(f"Download failed: {e}")

def download_gdown(id, dest_path):
    if os.path.exists(dest_path):
        print(f"{dest_path} already exists. Skipping.")
        return
    
    print(f"Downloading from Google Drive (ID: {id}) to {dest_path}...")
    try:
        url = f'https://drive.google.com/uc?id={id}'
        gdown.download(url, dest_path, quiet=False, fuzzy=True)
    except Exception as e:
        print(f"Gdown failed: {e}")

def extract_file(file_path, extract_to):
    print(f"Extracting {file_path}...")
    try:
        if file_path.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif file_path.endswith('.tar.gz') or file_path.endswith('.tar'):
            with tarfile.open(file_path, 'r') as tar_ref:
                tar_ref.extractall(extract_to)
        print("Extraction complete.")
    except Exception as e:
        print(f"Extraction failed: {e}")

def process_oscd():
    print("--- Processing OSCD ---")
    oscd_zip = os.path.join(DATA_ROOT, "OSCD.zip")
    download_file(OSCD_URL, oscd_zip)
    if os.path.exists(oscd_zip):
        extract_file(oscd_zip, os.path.join(DATA_ROOT, "OSCD"))

def process_levir():
    print("\n--- Processing LEVIR-CD ---")
    levir_zip = os.path.join(DATA_ROOT, "LEVIR-CD-256.zip") # Assuming 256 version or similar
    
    # Try GDrive
    download_gdown(LEVIR_CD_DRIVE_ID, levir_zip)
    
    if os.path.exists(levir_zip):
         extract_file(levir_zip, os.path.join(DATA_ROOT, "LEVIR-CD"))
    else:
        print("Automatic download failed.")
        print(f"Please manually download LEVIR-CD and place it at {levir_zip} or extract to data/raw/LEVIR-CD")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['levircd', 'oscd', 'all'], default='all')
    parser.add_argument('--drive_id', type=str, help='Google Drive File ID for LEVIR-CD', default="1dLuzldMRmbBNKPpUkX8Z53hi6NHLrWim")
    parser.add_argument('--output', type=str, default=None, help='Output directory for raw data')
    args = parser.parse_args()

    global DATA_ROOT
    if args.output:
        DATA_ROOT = args.output
    else:
        DATA_ROOT = os.path.join(os.getcwd(), 'data', 'raw')

    os.makedirs(DATA_ROOT, exist_ok=True)
    
    if args.dataset in ['oscd', 'all']:
        process_oscd()
    
    if args.dataset in ['levircd', 'all']:
        print(f"\n--- Processing LEVIR-CD (Drive ID: {args.drive_id}) ---")
        levir_zip = os.path.join(DATA_ROOT, "LEVIR-CD-256.zip")
        # Try GDrive with provided ID
        download_gdown(args.drive_id, levir_zip)
        
        if os.path.exists(levir_zip):
             extract_file(levir_zip, os.path.join(DATA_ROOT, "LEVIR-CD"))
        else:
            print("Automatic download failed.")
            print(f"Please manually download LEVIR-CD and place it at {levir_zip}")

if __name__ == "__main__":
    main()
