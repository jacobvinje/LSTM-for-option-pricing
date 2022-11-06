import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def read_files(path, filenames):
    """Reads all files and returns a dataframe"""
    return pd.concat((pd.read_csv(path + f, skipinitialspace=True) for f in filenames))

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

    df_opt["Moneyness"] = df_opt["Underlying_last"] / df_opt["Strike"]
    
    df_vol = calculate_volatility(df_opt)
    df_vol.info()
    df_opt = pd.merge(df_opt, df_vol, on ="Quote_date", how = "left")

    columns = ["Quote_date", "Expire_date",  "Underlying_last", "Strike", "Moneyness", "Ask", "Bid", "Ttl", "Volatility"]
    df_opt = df_opt[columns]
    df_opt = df_opt[df_opt["Ttl"] != 0]
    return df_opt[columns]

def calculate_volatility(df):
    """Calculate underlying 90 days annualized moving average volatility from dataset of options"""
    df_vol = df[["Quote_date", "Underlying_last"]].drop_duplicates()
    df_vol["Volatility"] = np.log(df_vol["Underlying_last"] / df_vol["Underlying_last"].shift()).rolling(90).std()*(252**0.5)
    return df_vol[["Quote_date", "Volatility"]]

def process_rates(df_r):
    """Renames rate duration"""
    df_r["Date"] = pd.to_datetime(df_r["Date"])
    df_r = df_r.rename(columns = {"Date" : "Quote_date", "3 Mo": "R"})
    return df_r[["Quote_date", "R"]]

def get_model_dataset(path_opt, filenames_opt, path_r, filenames_r, call = True):
    """Wrapper function to extract option data and rates. Returns a combined dataframe"""
    df_opt = read_files(path_opt, filenames_opt)
    df_r = read_files(path_r, filenames_r)
    df_opt = process_options(df_opt, call)
    df_r = process_rates(df_r)
    df = df_opt = pd.merge(df_opt, df_r, on ="Quote_date", how = "left")
    return df.dropna() #TODO: Fix handling of nan values

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
    df[["Bid_last", "Ask_last"]] = df[["Bid", "Ask"]]
    return df

def create_train_test(df, features, split_date, seq_length):
    """Splits data in training and test set, and transforms data to right 2D format"""
    train = lag_features(df[df["Quote_date"] < split_date], features, seq_length).to_numpy()
    test = lag_features(df[df["Quote_date"] >= split_date], features, seq_length).to_numpy()
    train[:, -len(features)*seq_length - 2:], test[:, -len(features)*seq_length - 2:] = min_max_scale(train[:, -len(features)*seq_length-2:], test[:, -len(features)*seq_length-2:])
    return train, test #TODO: Move reshaping to modell

def min_max_scale(train, test):
    """Scales a training and test set using MinMaxScaler. The scaler is calibrated on the training set"""
    scaler = MinMaxScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)
    return train, test

def split_x_y(data, num_features, seq_length):
    """Splits data in explanatory data x and explained data y"""
    data_x, data_y = data[:, -num_features*seq_length - 2:-2].astype(np.float32), data[:,-2:].astype(np.float32)
    data_x = np.reshape(data_x, (len(data_x), seq_length, num_features))
    return data_x, data_y


    


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