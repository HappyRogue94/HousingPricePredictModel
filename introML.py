import os
import tarfile
import requests
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from zlib import crc32

DOWNLOAD_ROOT= "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets","housing")
HOUSING_URL  = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    #set mask for permissions
    os.umask(0)
    os.makedirs(housing_path, exist_ok=True, mode=0o777)
    tgz_path=os.path.join(housing_path, "housing.tgz")
    #must set stream=True to get raw data
    downloaded_data = requests.get(housing_url, stream=True)
    with open(tgz_path, 'wb') as f:
        f.write(downloaded_data.raw.read())
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier) & 0xffffffff < test_ratio * 2**32)

def plot_data(data, kind, x, y, alpha):
    return data.plot(kind=kind, x=x, y=y, alpha=alpha)



if __name__ == '__main__':
    train_set, test_set = train_test_split(load_housing_data(), test_size=0.2, random_state=42) 
    housing = load_housing_data()

    #create new income category for the housing data set so that we can perform starified sampling
    housing["income_cat"] = pd.cut(housing["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1,2,3,4,5]) 

    # extracting statified train & test sets
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_idx]
        strat_test_set  = housing.loc[test_idx]
    #drop the income_cat to revert data back to its original form
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    

    ###EXPLORING THE DATA

    #here we are creating a copy of the training set so we dont harm the original data
    housing = strat_train_set.copy()

    #plot the data
    plot_data(housing, kind="scatter", x="longitude", y="latitude", alpha=0.1)



