import os, tarfile, urllib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

DIR = os.path.dirname(os.path.abspath(__file__))
TGZ_PATH = os.path.join(DIR, "housing.tgz")
CSV_PATH = os.path.join(DIR, "housing.csv")


def fetch_housing_data():
    urllib.request.urlretrieve(HOUSING_URL, TGZ_PATH)
    housing_tgz = tarfile.open(TGZ_PATH)
    housing_tgz.extractall(path=DIR)
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
 

def split_data():
    # Import the data
    housing = load_housing_data()

    # Our `y` value (labels) is the column 'median_house_value'. We're going to dropna for that column
    housing.dropna(subset=['median_house_value'], inplace=True)

    # Using stratified sampling
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0, 1.5, 3, 4.5, 6, np.inf],
        labels=[1, 2, 3, 4, 5]
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        train = housing.iloc[train_index]
        test = housing.iloc[test_index]
        
    # now remove the `income_cat` field
    train = train.drop(columns=["income_cat"]).to_numpy()
    test = test.drop(columns=["income_cat"]).to_numpy()
    housing.drop(columns=["income_cat"], inplace=True)

    # now take the 'median_house_value' as the y value
    y_indx = list(housing.columns).index('median_house_value')

    y_train = train[:, y_indx]
    y_test = test[:, y_indx]

    # Fit & transform the X values
    x_indx = tuple(set(range(len(housing.columns))) - {y_indx})

    X_train = train[:, x_indx]
    X_test = test[:, x_indx]

    return X_train, X_test, y_train, y_test


def split_data_pd():
    # Import the data
    housing = load_housing_data()

    # Our `y` value (labels) is the column 'median_house_value'. We're going to dropna for that column
    housing.dropna(subset=['median_house_value'], inplace=True)

    # Using stratified sampling
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0, 1.5, 3, 4.5, 6, np.inf],
        labels=[1, 2, 3, 4, 5]
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        train = housing.iloc[train_index]
        test = housing.iloc[test_index]
        
    # now remove the `income_cat` field
    train = train.drop(columns=["income_cat"])
    test = test.drop(columns=["income_cat"])
    housing.drop(columns=["income_cat"], inplace=True)

    # now take the 'median_house_value' as the y value
    y_train = train[["median_house_value"]]
    y_test = test[["median_house_value"]]

    # Fit & transform the X values
    x_cols = list(housing.columns)
    x_cols.remove("median_house_value")

    X_train = train[x_cols]
    X_test = test[x_cols]

    return X_train, X_test, y_train, y_test
