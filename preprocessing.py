import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

def lstm_format(df, seq_length):
    """
    HIGHLY WIP
    Transforms a raw 2D list of option data into a 3D format of sequential data for LSTM model.
    """
    df["Option_key"] = df["Expire_date"].astype(str) + " | " + df["Strike"].astype(str)
    features = ["Underlying_last", "Strike", "Ttl", "R"]
    df["F"] = df[features].values.tolist()
    sequenses = []

    for option in df.Option_key.unique():
        print("Key:", option)
        print(df[df["Option_key" == option]])
        df_opt = specific_option(df[df["Option_key" == option]], seq_length)

def specific_option(df, seq_length):
    """
    HIGHLY WIP
    Creates the sequential 3D format for a single option from a 2D list of all its quotes
    """
    for step in range(1,seq_length):
        df["F-"+str(step)] = df["F"].shift(step)
    df.info()
    return df
    

path_opt = "./data/options/"
#filenames = ["spx_eod_" + str(year) + (str(month) if month >= 10 else "0"+str(month)) +".txt" for year in range(2010, 2022) for month in range(1, 13)] + ["spx_eod_2022" + (str(month) if month >= 10 else "0"+str(month)) +".txt" for month in range(1, 10)]
filenames_opt = ["spx_eod_202209.txt"]
path_r = "./data/rates/"
filenames_r = ["yield-curve-rates-2022.csv", "yield-curve-rates-1990-2021.csv"]

df = get_model_dataset(path_opt, filenames_opt, path_r, filenames_r, True)
print(df)

df.info()
df = lstm_format(df, 5)
