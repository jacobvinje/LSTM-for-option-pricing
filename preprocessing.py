import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def read_files(path, filenames):
    """
    Reads all files and returns a dataframe with only the specificed columns
    """
    return pd.concat((pd.read_csv(path + f, skipinitialspace=True) for f in filenames))

def process_options(df_opt, call = True):
    """
    Cleans up column names and add time to live (Ttl) column to the dataframe
    """

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

    columns = ["Quote_date", "Expire_date",  "Underlying_last", "Strike", "Ask", "Bid", "Ttl"]
    df_opt = df_opt[columns]
    df_opt = df_opt[df_opt["Ttl"] != 0]
    return df_opt[columns]

def process_rates(df_r):
    """
    Rename rate duration
    """
    df_r["Date"] = pd.to_datetime(df_r["Date"])
    df_r = df_r.rename(columns = {"Date" : "Quote_date", "3 Mo": "R"})
    #rate_keys = {key: key if key == "Date" else int(key.split(" ")[0])*30 if key.split(" ")[1] == "Mo" else int(key.split(" ")[0])*365  for key in df_r.keys()}
    #df_r = df_r.rename(columns=rate_keys)
    columns = ["Quote_date", "R"]
    return df_r[columns]

def combine_opt_r(df_opt, df_r):
    """
    Combines the dataset for options and rates
    """
    #df_opt["R"] = df_opt.apply(lambda row : df_rates[str(min(df_r.drop(["Date"], axis = 1).keys(), key = lambda x:abs(int(x)-row.Ttl)))][row.Quote_date], axis = 1)
    df_opt = pd.merge(df_opt, df_r, on ="Quote_date", how = "left")
    return df_opt

def get_model_dataset(path_opt, filenames_opt, path_r, filenames_r, call = True):
    """
    Wrapper function to extract option data and rates. Returns a combined dataframe
    """
    df_opt = read_files(path_opt, filenames_opt)
    df_r = read_files(path_r, filenames_r)
    df_opt = process_options(df_opt, call)
    df_r = process_rates(df_r)
    df = combine_opt_r(df_opt, df_r)
    return df.dropna() #TODO: Fix handling of nan values

def lstm_format(df, features, seq_length):
    """
    Transforms a raw 2D list of option data into a explanatory variable x and explained y
    x: 3D matrix of lagged features per sequence
    y: 2D array of bid-ask prices per sequence
    """
    df = df.sort_values(["Expire_date", "Strike", "Ttl"], ascending = [True, True, False])

    for step in range(seq_length):
        for feature in features:
            df[feature + " - " + str(step)] = df[feature].shift(step)
    
    df["Check_strike"] = df["Strike"] == df["Strike"].shift(seq_length-1)
    df["Check_expire"] = df["Expire_date"] == df["Expire_date"].shift(seq_length-1)
    df = df[(df["Check_strike"] == True) & (df["Check_expire"] == True)]

    x = df[[feature + " - " + str(step) for step in range(seq_length)[::-1] for feature in features]].to_numpy()
    y = df[["Bid", "Ask"]].to_numpy()
    return x, y

def create_train_test(df, features, split_date, seq_length):
    train_x, train_y = lstm_format(df[df["Quote_date"] < split_date], features, seq_length)
    test_x, test_y = lstm_format(df[df["Quote_date"] >= split_date], features, seq_length)
    train_x, test_x = min_max_scale(train_x, test_x)
    train_y, test_y = min_max_scale(train_y, test_y)
    return np.reshape(train_x, (len(train_x), seq_length, len(features))), train_y, np.reshape(test_x, (len(test_x), seq_length, len(features))), test_y

def min_max_scale(train, test):
    scaler = MinMaxScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)
    return train, test

features = ["Underlying_last", "Moneyness", "Ttl", "R"]
train_x, train_y, test_x, test_y = create_train_test(df_read, features,  "2022-09-18", 5)
    

path_opt = "./data/options/"
#filenames = ["spx_eod_" + str(year) + (str(month) if month >= 10 else "0"+str(month)) +".txt" for year in range(2010, 2022) for month in range(1, 13)] + ["spx_eod_2022" + (str(month) if month >= 10 else "0"+str(month)) +".txt" for month in range(1, 10)]
filenames_opt = ["spx_eod_202209.txt"]
path_r = "./data/rates/"
filenames_r = ["yield-curve-rates-2022.csv", "yield-curve-rates-1990-2021.csv"]

df = get_model_dataset(path_opt, filenames_opt, path_r, filenames_r, True)
print(df)

df.info()
df = lstm_format(df, 5)
