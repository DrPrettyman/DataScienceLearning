import os, tarfile, urllib
import pandas as pd

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = "/" + os.path.join(*os.getcwd().split('/')[:-1] + ["datasets"])
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

os.makedirs(HOUSING_PATH, exist_ok=True)

def fetch_housing_data():
    tgz_path = os.path.join(HOUSING_PATH, "housing.tgz")
    urllib.request.urlretrieve(HOUSING_URL, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=HOUSING_PATH)
    housing_tgz.close()

def load_housing_data():
    return pd.read_csv(os.path.join(HOUSING_PATH, "housing.csv"))
