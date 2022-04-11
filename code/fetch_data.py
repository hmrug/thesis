#!/usr/bin/env python
# --- Fetch and clean all (free) data needed for the project
# --- Autor: Hubert Mrugala (hubertmrugala.com) ---

import numpy as np
import pandas as pd
import pandas_datareader as pdr

# Functions

# Convert all data in a pandas df to numeric (eliminate string)
def to_num(df):
    l = []
    for i in range(len(df.columns)):
        series = pd.to_numeric(df.iloc[:,i],errors="coerce")
        l.append(series)
    df = pd.concat(l,axis=1)
    return df

# A function for getting multi-series PD data
def clean_dealers(url):
    df = pd.read_csv(url,index_col="As Of Date")
    id = df["Time Series"].unique()
    series = []
    for i in range(len(id)):
        s = df[df["Time Series"]==id[i]].iloc[:,-1]
        series.append(s)
    series = pd.concat(series,axis=1)
    series.columns = id
    # series["All"] = series.iloc[:,0] + series.iloc[:,1]
    return series

start_date = "2008-01-02"
end_date = "2021-12-31"
# Pretend to be a browser when scraping
hdr = {'User-Agent': 'Mozilla/5.0'}

# Fred data
fred_series = ['TREAST','RRPONTSYD','EFFR','DFEDTARU','T10Y2Y']
fred_data = pdr.DataReader(fred_series,"fred",start_date,end_date)

fed_rate = pdr.DataReader("DFEDTAR","fred",start_date,end_date)\
                        .dropna().rename({"DFEDTAR": "Target Rate"},axis=1)
fed_rate_u =pdr.DataReader("DFEDTARU","fred",start_date,end_date)\
                        .dropna().rename({"DFEDTARU": "Target Rate"},axis=1)

fed_target = pd.concat([fed_rate,fed_rate_u])

# Repo rates data
# GCF Repo Service Average Rate: U.S. Treasury Securities (Final)
ofr_gcf_rate = pd.read_csv("https://data.financialresearch.gov/v1/series/timeseries?mnemonic=REPO-GCF_AR_T-F&output=csv",
                           index_col="date")
ofr_gcf_rate.index = pd.to_datetime(ofr_gcf_rate.index)
ofr_gcf_rate.columns = ["OFR_GCF_RATE"]

# GCF Repo Service Outstanding Volume: U.S. Treasury Securities (Final)
ofr_gcf_vol = pd.read_csv("https://data.financialresearch.gov/v1/series/timeseries?mnemonic=REPO-GCF_OV_T-F&output=csv",
                          index_col="date")
ofr_gcf_vol.index = pd.to_datetime(ofr_gcf_vol.index)
ofr_gcf_vol.columns = ["OFR_GCF_VOLUME"]

# DTCC
dtcc_url = "https://www.dtcc.com/data/gcfindex.csv"
dtcc  = pd.read_csv(dtcc_url,
                    storage_options=hdr,on_bad_lines="skip",
                    index_col="Date").loc["01/03/2022":]
dtcc.index = pd.to_datetime(dtcc.index)

dtcc_hist_url = "https://www.dtcc.com/-/media/Files/Downloads/Clearing-Services/FICC/GCF-Index-Graph.xlsx"
dtcc_hist = pd.read_excel(dtcc_hist_url,
                          storage_options=hdr,index_col="Date",
                          header=6,sheet_name=0)
dtcc_hist.index = pd.to_datetime(dtcc_hist.index)

dtcc_hist_rates = dtcc_hist.iloc[:,:2].rename(
    {
        dtcc_hist.columns[0]: "MBS",
        dtcc_hist.columns[1]: "Treasury"
    },axis=1)

dtcc_rates = dtcc.iloc[:,[1,3]].rename(
    {
        dtcc.iloc[:,[1,3]].columns[0]: "MBS",
        dtcc.iloc[:,[1,3]].columns[1]: "Treasury"
    },axis=1)

dtcc_alltime = pd.concat([dtcc_hist_rates,dtcc_rates],axis=0)

# Treasury data
# Debt to Penny [fiscaldata.treasury.gov]
debt_zip_url = "https://fiscaldata.treasury.gov/static-data/downloads/zip/cd890c798d9cac3e69ae3ddee083931ee8c637a8cae14d94a209b9002dea7740/DebtPenny_19930401_20220317.zip"
debt = pd.read_csv(debt_zip_url,
                   index_col="Record Date")["Total Public Debt Outstanding"].iloc[::-1]
debt.index = pd.to_datetime(debt.index)

# Yield curve
ust = pd.read_csv("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/all/all?type=daily_treasury_yield_curve&field_tdr_date_value=all&page&_format=csv",
                  index_col="Date").iloc[::-1]
ust.index = pd.to_datetime(ust.index)

# 3m/10y curve
# ust_3m10y = pd.read_excel("https://www.newyorkfed.org/medialibrary/media/research/capital_markets/allmonth.xls",
#                           index_col="Date")["Spread"]
# ust_3m10y.index = pd.to_datetime(ust_3m10y.index)
ust_3m10y = (ust["10 Yr"] - ust["3 Mo"])
ust_3m10y.name = "3m/10y"
ust_3m10y.index = pd.to_datetime(ust_3m10y.index)


# NYFED Primary Dealers Data
dealers_all_url = "https://markets.newyorkfed.org/api/pd/get/all/timeseries.csv"
df_dealers = clean_dealers(dealers_all_url)
dealers = to_num(df_dealers)
dealers.index = pd.to_datetime(dealers.index)

# Net positions
n_bills_id = ["PDPOSGS-B"]
n_coupons_id = ["PDTRGSC-L2","PDTRGSC-G2L3","PDTRGSC-G3L6","PDTRGSC-G6L7","PDTRGSC-G7L11","PDTRGSC-G11"]
# Transactions
t_bills_id = ["PDTRGS-EXTB"]
t_coupons_id = ["PDTRGSC-L2","PDTRGSC-G2L3","PDTRGSC-G3L6","PDTRGSC-G6L7","PDTRGSC-G7L11","PDTRGSC-G11"]
# Repo fails
fails_id = ["PDFTR-USTET","PDFTD-USTET"]

dealers_positions = pd.concat(
    [
        dealers[n_bills_id],
        dealers[n_coupons_id],
        dealers[n_coupons_id].sum(axis=1).rename("Positions: All coupons"),
        pd.concat([dealers[n_bills_id],dealers[n_coupons_id]],axis=1).sum(axis=1).rename("Positions: All")
    ],
    axis=1)
dealers_positions.index = pd.to_datetime(dealers_positions.index)


dealers_transactions= pd.concat(
    [
        dealers[t_bills_id],
        dealers[t_coupons_id],
        dealers[t_coupons_id].sum(axis=1).rename("Transactions: All coupons"),
        pd.concat([dealers[t_bills_id],dealers[t_coupons_id]],axis=1).sum(axis=1).rename("Transactions: All")
    ],
    axis=1)
dealers_transactions.index = pd.to_datetime(dealers_transactions.index)

repo_fails = pd.concat(
    [
        dealers[fails_id],
        dealers[fails_id].sum(axis=1).rename("All fails"),
    ],
    axis=1)
repo_fails.index = pd.to_datetime(repo_fails.index)

dealers_all = pd.concat([dealers_positions,dealers_transactions,repo_fails],axis=1).dropna().resample("W-WED").first()
dealers_selected = dealers_all[["Positions: All","Transactions: All","All fails"]]

# Main: Final dataframe        
def concat_data():

    df = pd.concat(
        [ fred_data,fed_target, ofr_gcf_rate, ofr_gcf_vol, dtcc_alltime,
            debt, ust, ust_3m10y, dealers_selected],axis=1)
    df.index.name = "Date"

    df_final = df.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,23,24,25,26]] \
                    [start_date:end_date].resample("W-WED").first()
    df_final["RRPONTSYD"].replace(np.nan,0,inplace=True)

    return df_final

df_final = concat_data()

def main():
    return concat_data().to_csv("./resources/data/fetched_data.csv")

if __name__ == "__main__":
    main()

# Add:
#   - libor rates
#   - repo rates
#   - repo volatility
