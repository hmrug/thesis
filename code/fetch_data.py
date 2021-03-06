#!/usr/bin/env python
# --- Fetch and clean all (free) data needed for the project
# --- Autor: Hubert Mrugala (hubertmrugala.com) ---

# Packages
import os
import requests
import csv
import numpy as np
import pandas as pd
import pandas_datareader as pdr

# Path to the project directory
cwd = os.getenv("THESIS_PATH")
os.chdir(cwd)

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
                        .dropna()
fed_rate_u =pdr.DataReader("DFEDTARU","fred",start_date,end_date)\
                        .dropna().rename({"DFEDTARU": "DFEDTAR"},axis=1)

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
        dtcc_hist.columns[0]: "GCF_MBS",
        dtcc_hist.columns[1]: "GCF_TREASURY"
    },axis=1)

dtcc_rates = dtcc.iloc[:,[1,3]].rename(
    {
        dtcc.iloc[:,[1,3]].columns[0]: "GCF_MBS",
        dtcc.iloc[:,[1,3]].columns[1]: "GCF_TREASURY"
    },axis=1)

dtcc_alltime = pd.concat([dtcc_hist_rates,dtcc_rates],axis=0)

# Treasury data
# Debt to Penny [fiscaldata.treasury.gov]
# url: https://fiscaldata.treasury.gov/datasets/debt-to-the-penny/debt-to-the-penny
debt_to_penny_url = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/debt_to_penny?filter=record_date:gte:2006-01-01&page[size]=5000&format=csv"
debt = pd.read_csv(debt_to_penny_url,
                   index_col="record_date",
                   storage_options=hdr)\
                .iloc[:,:3]
debt.index = pd.to_datetime(debt.index)

# Yield curve
ust = pd.read_csv("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/all/all?type=daily_treasury_yield_curve&field_tdr_date_value=all&page&_format=csv",
                  index_col="Date").iloc[::-1]
ust = ust[["1 Mo","3 Mo","10 Yr","30 Yr"]]
ust.columns = ["UST_1M","UST_3M","UST_10Y","UST_30Y"]
ust.index = pd.to_datetime(ust.index)
# 3m/10y curve
# ust_3m10y = pd.read_excel("https://www.newyorkfed.org/medialibrary/media/research/capital_markets/allmonth.xls",
#                           index_col="Date")["Spread"]
ust["T10Y3M"] = ust["UST_10Y"] - ust["UST_3M"]

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

dealers_all = pd.concat(
    [dealers_positions,dealers_transactions,repo_fails],
    axis=1).dropna().resample("W-WED").first()

dealers_selected = dealers_all[["Positions: All","Transactions: All","All fails"]]
dealers_selected.columns = ["PD_POS_ALL","PD_TRANS_ALL","PD_FAILS_ALL"]

# SOMA Holdings
soma_url = 'https://markets.newyorkfed.org/api/soma/summary.csv'
with requests.Session() as s:
    download = s.get(soma_url)
decoded_content = download.content.decode('utf-8')
cr = csv.reader(decoded_content.splitlines(), delimiter=',')
soma_list = list(cr)

soma = pd.DataFrame(soma_list,columns=soma_list[0]).set_index('As Of Date').iloc[1:,:]
soma.index = pd.to_datetime(soma.index)
        
soma['CMBS'] = soma['CMBS'].str[:-1]
soma['Agencies'] = soma['Agencies'].str.replace('','0')
soma['FRN'] = soma['FRN'].str.replace('','0')

soma = soma.astype(float)
soma.columns = ["SOMA_TOTAL","SOMA_MBS","SOMA_TIPS","SOMA_FRN",
    "SOMA_TIPS_INFCOMP","SOMA_NOTES+BONS","SOMA_BILLS","SOMA_AGENCIES",
    "SOMA_CMBS"]

## USD ON LIBOR
libor = pd.read_csv("./data/USD_ON_LIBOR.csv",
                    index_col="DATE")\
                    .rename({"USDONTD156N": "USD_ON_LIBOR"},axis=1)
libor.index = pd.to_datetime(libor.index)
#libor = libor.resample("W-WED").first()

## Collateral Spread
coll_spread = pd.concat([libor, dtcc_alltime["GCF_TREASURY"]],axis=1)
coll_spread = coll_spread["USD_ON_LIBOR"] - coll_spread["GCF_TREASURY"]

# Main: Final dataframe        
def concat_data():

    df = pd.concat(
        [
            fred_data,fed_target, ofr_gcf_rate,
            ofr_gcf_vol, dtcc_alltime, debt,
            ust, dealers_selected,soma, libor],
        axis=1)
    df.index.name = "Date"

    df = df[start_date:end_date].resample("W-WED").first()
    df["RRPONTSYD"].replace(np.nan,0,inplace=True)

    df["COL_SPREAD"] = df["USD_ON_LIBOR"] - df["GCF_TREASURY"]

    return df

#df = concat_data()

def main():
    return concat_data().to_csv("./data/fetched_data.csv")

if __name__ == "__main__":
    main()
