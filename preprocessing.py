import pandas as pd
import matplotlib.pyplot as plt

def read_files(path, filenames):
    """
    Reads all files and returns a dataframe with only the specificed columns
    """
    return pd.concat((pd.read_csv(path + f, skipinitialspace=True) for f in filenames))

def process_options(df_options, columns, call = True):
    """
    WIP
    Cleans up column names and add time to live (TTL) column to the dataframe
    """
    
    keys = {key: key[key.find("[")+1:key.find("]")] for key in df_options.keys()}
    df_options = df_options.rename(columns=keys)

    df_options["QUOTE_DATE"] = pd.to_datetime(df_options["QUOTE_DATE"])
    df_options["EXPIRE_DATE"] = pd.to_datetime(df_options["EXPIRE_DATE"])
    df_options["TTL"] = df_options.apply(lambda row: (row.EXPIRE_DATE - row.QUOTE_DATE).days, axis = 1)

    if call:
        columns = ["QUOTE_DATE", "EXPIRE_DATE", "UNDERLYING_LAST", "STRIKE", "C_ASK", "C_BID", "TTL"]
    else:
        columns = ["QUOTE_DATE", "EXPIRE_DATE", "UNDERLYING_LAST", "STRIKE", "P_ASK", "P_BID", "TTL"]
    
    keys = {key: key[key.find("[")+1:key.find("]")] for key in df_options.keys()}
    df_options = df_options.rename(columns=keys)

    return df_options[columns]

def process_rates(df_r, columns):
    """
    WIP
    Rename rate duration as days
    """
    df_r = df_r[columns]
    df_r["Date"] = pd.to_datetime(df_r["Date"])
    df_r = df_r.rename(columns = {"Date" : "QUOTE_DATE"})
    #rate_keys = {key: key if key == "Date" else int(key.split(" ")[0])*30 if key.split(" ")[1] == "Mo" else int(key.split(" ")[0])*365  for key in df_r.keys()}
    #df_r = df_r.rename(columns=rate_keys)
    return df_r

def combine_opt_r(df_opt, df_r):
    """
    WIP
    Combines the dataset for options and rates
    """
    #df_opt["R"] = df_opt.apply(lambda row : df_rates[str(min(df_r.drop(["Date"], axis = 1).keys(), key = lambda x:abs(int(x)-row.TTL)))][row.QUOTE_DATE], axis = 1)
    df_opt = pd.merge(df_opt, df_r, on ="QUOTE_DATE", how = "left")
    return df_opt

def get_model_dataset(path_opt, filenames_opt, path_r, filenames_r, call = True):
    df_opt = read_files(path_opt, filenames_opt)
    df_r = read_files(path_r, filenames_r)




path = "C:/Users/Erlend/Google Drive/NTNU/5. Klasse/Prosjektoppgave/Data/option data/"
filenames = ["spx_eod_" + str(year) + (str(month) if month >= 10 else "0"+str(month)) +".txt" for year in range(2010, 2022) for month in range(1, 13)] + ["spx_eod_2022" + (str(month) if month >= 10 else "0"+str(month)) +".txt" for month in range(1, 10)]
filenames = ["spx_eod_202209.txt"]
columns = ["QUOTE_DATE", "EXPIRE_DATE", "UNDERLYING_LAST", "STRIKE", "C_ASK", "C_BID", "P_ASK", "P_BID"]
df_opt = read_files(path, filenames)
df_opt = process_options(df_opt, columns)

path = "C:/Users/Erlend/Google Drive/NTNU/5. Klasse/Prosjektoppgave/Data/"
filenames = ["yield-curve-rates-2022.csv", "yield-curve-rates-1990-2021.csv"]
columns =["Date", "3 Mo"]
df_rates = read_files(path, filenames)
df_r = process_rates(df_rates, columns)

print(df_opt)
df_opt.info()
print(df_r)
df_r.info()

df_opt = combine_opt_r(df_opt, df_r)
print(df_opt)
df_opt.info()

print(df_opt[df_opt.isna().any(axis=1)])

#df_opt = combine_opt_r(df_opt, df_r)
#print(df_opt)
#df_opt.info()

#plt.plot(df["QUOTE_DATE"].tolist(), df["UNDERLYING_LAST"].tolist())
#plt.show(block = True)


"""
df_3500 = df.loc[(df["STRIKE"] == 3800) & (df["EXPIRE_DATE"] == "2022-09-30")]
print(df_3500)
plt.plot(df["QUOTE_DATE"].tolist(), df["UNDERLYING_LAST"].tolist())
plt.plot(df_3500["QUOTE_DATE"].tolist(), df_3500["C_ASK"].tolist())
plt.show(block = True)
"""