import os
import requests
import zipfile
import tarfile
from tqdm import tqdm

# Configuration
DATA_ROOT = os.path.join(os.getcwd(), 'data')
LEVIR_CD_URL = "https://www.dropbox.com/s/1p55bwy5k82437y/LEVIR-CD-256.zip?dl=1" # Example/Placeholder URL - real one might differ or require authentication
OSCD_URL = "https://rc.eodatabase.ch/oscd/" # Base URL, specific files needed
# HRSCD link usually from IEEE Dataport, requires login. 

def download_file(url, dest_path):
    if os.path.exists(dest_path):
        print(f"{dest_path} already exists. Skipping.")
        return
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True) as bar:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

def extract_file(file_path, extract_to):
    print(f"Extracting {file_path}...")
    if file_path.endswith('.zip'):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif file_path.endswith('.tar.gz') or file_path.endswith('.tar'):
        with tarfile.open(file_path, 'r') as tar_ref:
            tar_ref.extractall(extract_to)
    print("Extraction complete.")

def main():
    os.makedirs(DATA_ROOT, exist_ok=True)
    
    print("NOTE: Some datasets (HRSCD) require manual download due to authentication.")
    print(f"Data directory: {DATA_ROOT}")
    
    # Placeholder for LEVIR-CD download logic
    # levir_zip = os.path.join(DATA_ROOT, "LEVIR-CD.zip")
    # download_file(LEVIR_CD_URL, levir_zip)
    # extract_file(levir_zip, os.path.join(DATA_ROOT, "LEVIR-CD"))

    # Placeholder for OSCD
    # oscd_url = ...
    
    # Placeholder for HRSCD
    
    print("Please manually place downloaded datasets in the 'data' folder if automatic download fails or is not possible.")

if __name__ == "__main__":
    main()
