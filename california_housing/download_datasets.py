import os, tarfile, urllib
import pandas as pd

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = "/" + os.path.join(*os.getcwd().split('/')[:-1] + ["datasets"])
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
CSV_PATH = os.path.join(HOUSING_PATH, "housing.csv")

os.makedirs(HOUSING_PATH, exist_ok=True)

def fetch_housing_data():
    tgz_path = os.path.join(HOUSING_PATH, "housing.tgz")
    urllib.request.urlretrieve(HOUSING_URL, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=HOUSING_PATH)
    housing_tgz.close()
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError("CSV file was not downloaded or extracted")
    
    file_size = os.stat(CSV_PATH).st_size
    if file_size < 1_000_000:
        raise ImportError(f"CSV file was extracted but is only {file_size} bytes")

def load_housing_data():
    """Loads the housing data. """
    
    # If the csv file does not exist, we fetch it
    if not os.path.exists(CSV_PATH):
        fetch_housing_data()

    # We expect the file to be over 1MB, if not, something is wrong, we'll fetch it again
    if os.stat(CSV_PATH).st_size < 1_000_000:
        fetch_housing_data()

    return pd.read_csv(CSV_PATH)
 