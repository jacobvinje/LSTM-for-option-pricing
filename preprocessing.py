import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pathlib import Path

def read_files(path, filenames):
    """Reads all files and returns a dataframe"""
    return pd.concat((pd.read_csv(path + f, skipinitialspace=True) for f in filenames))

def read_file(file):
    """Read a single file and return a dataframe"""
    return pd.read_csv(file, skipinitialspace=True)

def process_options(df_opt, call = True):
    """Cleans up column names and add time to live (Ttl) and volatility column to the dataframe"""
    keys = {key: key[key.find("[")+1:key.find("]")][0] + key[key.find("[")+1:key.find("]")][1:].lower()  for key in df_opt.keys()}
    df_opt = df_opt.rename(columns=keys)

    if call:
        keys = {"C_ask": "Ask", "C_bid": "Bid"}
    else:
        keys = {"P_ask": "Ask", "P_bid": "Bid"}
    df_opt = df_opt.rename(columns=keys)

    df_opt["Quote_date"] = pd.to_datetime(df_opt["Quote_date"])
    df_opt["Expire_date"] = pd.to_datetime(df_opt["Expire_date"])
    df_opt["Ttl"] = df_opt.apply(lambda row: (row.Expire_date - row.Quote_date).days, axis = 1)
    df_opt["Price"] = (df_opt["Ask"] + df_opt["Bid"])/2

    #df_opt["Moneyness"] = df_opt["Underlying_last"] / df_opt["Strike"]
    #df_opt["Bid_strike"] = df_opt["Bid"] / df_opt["Strike"]
    #df_opt["Ask_strike"] = df_opt["Ask"] / df_opt["Strike"]
     
    df_vol = calculate_volatility(df_opt)
    df_opt = pd.merge(df_opt, df_vol, on ="Quote_date", how = "left")

    #columns = ["Quote_date", "Expire_date",  "Underlying_last", "Strike", "Ask", "Bid",  "Bid_strike", "Ask_strike", "Moneyness", "Ttl", "Volatility"]
    #columns = ["Quote_date", "Expire_date", "Ask", "Bid", "Underlying_last", "Strike", "Ttl", "Volatility"]
    columns = ["Quote_date", "Expire_date", "Price", "Underlying_last", "Strike", "Ttl", "Volatility"]
    df_opt = df_opt[columns]
    df_opt = df_opt[(df_opt["Ttl"] != 0) & (df_opt["Ttl"] <= 365*3)]
    return df_opt[columns]

def calculate_volatility(df):
    """Calculate underlying 90 days annualized moving average volatility from dataset of options"""
    df_vol = df[["Quote_date", "Underlying_last"]].drop_duplicates()
    df_vol["Volatility"] = np.log(df_vol["Underlying_last"] / df_vol["Underlying_last"].shift()).rolling(90).std()*(252**0.5)
    return df_vol[["Quote_date", "Volatility"]]

def process_rates(df_r):
    """Renames date column and rate duration to days"""
    df_r["Date"] = pd.to_datetime(df_r["Date"])
    keys = {"Date" : "Quote_date",
            "1 Mo": 30,
            "3 Mo": 90,
            "6 Mo": 180,
            "1 Yr": 365,
            "2 Yr": 365*2,
            "3 Yr": 365*3,
            "5 Yr": 365*5,
            "7 Yr": 365*7,
            "10 Yr": 365*10}
    df_r = df_r.rename(columns = keys)
    return df_r[keys.values()]

def combine_opt_rates(df_opt, df_r):
    """Combines dataframes for options and rates matching the Ttl of the option to the closest R"""
    df_opt = pd.merge(df_opt, df_r, on ="Quote_date", how = "left")
    rates = list(df_r.columns)
    rates.remove("Quote_date")
    df_opt["Ttl_diff"] = df_opt["Ttl"].apply(lambda x: (np.abs(np.array(rates) - x)).argmin())
    df_opt["R"] = df_opt[["Ttl_diff"] + rates].values.tolist()
    df_opt["R"] = df_opt["R"].apply(lambda x: x[int(x[0]+1)])
    df_opt = df_opt.drop(rates + ["Ttl_diff"], axis=1)
    return df_opt.dropna()

def get_model_dataset(path_opt, filenames_opt, path_r, filenames_r, call = True):
    """Wrapper function to extract option data and rates. Returns a combined dataframe"""
    df_opt = read_files(path_opt, filenames_opt)
    df_r = read_files(path_r, filenames_r)
    df_opt = process_options(df_opt, call)
    df_r = process_rates(df_r)
    df = combine_opt_rates(df_opt, df_r)
    return df #TODO: Fix handling of nan values

def lag_features(df, features, seq_length):
    """Transforms a raw 2D dataframe of option data into 2D dataframe ofsequence data.
    Last 2 indexes per sequence are bid and ask price. The len(features)*seq_length
    features before are sequences of features"""
    df = df.sort_values(["Expire_date", "Strike", "Ttl"], ascending = [True, True, False])

    for step in range(seq_length)[::-1]:
        for feature in features:
            df[feature + "-" + str(step)] = df[feature].shift(step)
    
    df["Check_strike"] = df["Strike"] == df["Strike"].shift(seq_length-1)
    df["Check_expire"] = df["Expire_date"] == df["Expire_date"].shift(seq_length-1)
    df = df[(df["Check_strike"] == True) & (df["Check_expire"] == True)]
    df = df.drop(["Check_strike", "Check_expire"], axis=1)
    #df[["Bid_strike_last", "Ask_strike_last"]] = df[["Bid_strike", "Ask_strike"]]
    #df[["Bid_last", "Ask_last"]] = df[["Bid", "Ask"]]
    df["Price_last"] = df["Price"]
    df = df.sort_values(["Quote_date"], ascending = [True])
    return df

def create_train_test(df, split_date):
    """Splits data in training and test set, and transforms data to right 2D format"""
    return df[df["Quote_date"] < split_date], df[df["Quote_date"] >= split_date]

def df_to_xy(df, num_features, seq_length, num_outputs):
    """Transforms a dataframe into two arrays of explanatory variables x and explained variables y"""
    array = df.to_numpy()
    array_x, array_y = array[:, -num_features*seq_length - num_outputs:-num_outputs].astype(np.float32), array[:,-num_outputs:].astype(np.float32)
    return array_x, array_y

def min_max_scale(train, test):
    """Scales a training and test set using MinMaxScaler. The scaler is calibrated on the training set"""
    scaler = MinMaxScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)
    return train, test

def create_csv(first_year, last_year):
    path_opt = "./data/options/"
    filenames_opt = ["spx_eod_" + str(year) + (str(month) if month >= 10 else "0"+str(month)) +".txt" for year in range(first_year-1, last_year+1) for month in range(1, 13)]
    path_r = "./data/rates/"
    filenames_r = ["yield-curve-rates-2022.csv", "yield-curve-rates-1990-2021.csv"]
    call = True
    df = get_model_dataset(path_opt, filenames_opt, path_r, filenames_r, call)
    print("Data read")

    df = lag_features(df, features = ["Underlying_last", "Strike", "Ttl", "Volatility", "R"], seq_length = 5)

    df = df[df["Quote_date"] >= f"{str(first_year)}-01-01"]
    df = df[df["Quote_date"] <= f"{str(last_year)}-12-31"]

    filename = f"./data/processed_data/{first_year}-{last_year}_underlying-strike_only-price_inc.lags.csv"

    filepath = Path(filename)  
    filepath.parent.mkdir(parents=True, exist_ok=True)  
    df.to_csv(filename)
    print("Data written")

"""first_year = 2021
last_year = 2021
create_csv(first_year, last_year)"""


"""
path_opt = "./data/options/"
#filenames = ["spx_eod_" + str(year) + (str(month) if month >= 10 else "0"+str(month)) +".txt" for year in range(2010, 2022) for month in range(1, 13)] + ["spx_eod_2022" + (str(month) if month >= 10 else "0"+str(month)) +".txt" for month in range(1, 10)]
filenames_opt = ["spx_eod_202209.txt"]
path_r = "./data/rates/"
filenames_r = ["yield-curve-rates-2022.csv", "yield-curve-rates-1990-2021.csv"]

df_read = get_model_dataset(path_opt, filenames_opt, path_r, filenames_r, True)
print(df_read)
df_read.info()

features = ["Underlying_last", "Moneyness", "Ttl", "R"]
train_x, train_y, test_x, test_y = create_train_test(df_read, features,  "2022-09-18", 5)
"""