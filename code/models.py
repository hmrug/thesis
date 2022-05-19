#!/usr/bin/env python
# --- Econometics models --- 
# --- UZH MA Thesis : Scarcity channel of QE in the US --- 
# --- Autor: Hubert Mrugala (hubertmrugala.com) ---

# --- Packages --- 
import os
from datetime import datetime as dtt
from datetime import timedelta as tdelta
import numpy as np
import pandas as pd
from pandas.core.arrays.datetimes import sequence_to_datetimes
import statsmodels.api as sm
from statsmodels.formula.api import ols
from stargazer.stargazer import Stargazer
from statsmodels.iolib.summary2 import summary_col
import pandas_datareader.data as pdr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import seaborn as sns

# --- Options ---
pd.set_option('display.precision',4)
plt.rcParams.update({
    "figure.dpi": 100,
    "figure.figsize": [16,9],
    "font.size": 16,
    "axes.titlesize": 24,
    "axes.titleweight": "bold",
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
sns.set_style('whitegrid')
#pd.set_option("max_rows",40)
pd.options.mode.chained_assignment = None  # default='warn'

# Path to the project directory
cwd = os.getenv("THESIS_PATH")
os.chdir(cwd)
fig_path = cwd+"resources/figures/"
tab_path = cwd+"resources/tables/"

# --- Functions --- 
def hist_vol(series):
    """
    Compute 10 points (weeks) volatility
    """
    vols = [series[i-10:i].std() for i in range(10,len(series))]
    return pd.Series(vols,name="Volatility",index=series.index[10:])

def log_vol(series):
    vols = (series/series.shift()).apply(np.log)
    finite_vols = vols[np.isfinite(vols)]
    return finite_vols

start_date = dtt(2008,1,2)
end_date = dtt(2022,1,5)

# --- Data --- 
df = pd.read_csv("data/fetched_data.csv",
                 index_col="Date")
df.index = pd.to_datetime(df.index)

# VIX
vix = pdr.DataReader('^VIX','yahoo',start_date,end_date)['Adj Close']\
                .resample("W-WED").first()\
                .rename("VIX")


# RRP Rate (in bps)
rrp_rate = pdr.DataReader('RRPONTSYAWARD','fred',start_date,end_date)\
                .resample("W-WED").first()\
                .rename({"RRPONTSYAWARD": "RRP_RATE"},axis=1)*100

# Add bills repo
# OFR U.S. Repo Markets Data Release
# DVP Service Average Rate: Term, <=30 Days (Final)
# url: https://www.financialresearch.gov/short-term-funding-monitor/datasets/repo-single/?mnemonic=REPO-DVP_AR_LE30-F
ofr_st_url = "https://data.financialresearch.gov/v1/series/timeseries?mnemonic=REPO-DVP_AR_LE30-F&output=csv"
ofr_st = pd.read_csv(ofr_st_url,index_col='date')
ofr_st.columns = ["REPO_ST"]
ofr_st = ofr_st * 100
ofr_st.index = pd.to_datetime(ofr_st.index)
ofr_st = ofr_st.resample("W-WED").first()

# Add bid-ask data
bidask = pd.read_excel("data/bid_ask_data.xlsx",
                       header=1,index_col="Date")\
                .iloc[:,-4:]\
                .iloc[::-1]\
                .loc[start_date:end_date]\
                .resample("W-WED").first()
bidask.index = pd.to_datetime(bidask.index)
bidask["UST10Y_10W_VOL"] = hist_vol(bidask["UST10Y_MID"])
bidask["UST3M_10W_VOL"] = hist_vol(bidask["UST3M_MID"])
# All rates in basis points
bidask = bidask * 100

df = pd.concat([df,vix,ofr_st,bidask,rrp_rate],axis=1)

# Change measure
# All in trillion USD
#df["SOMA_TREASURY"] = df["SOMA_TOTAL"] / 1000000000000
df["SOMA_TREASURY"] = (df["SOMA_NOTES+BONS"]+df["SOMA_BILLS"]) / 1000000000000
df["DEBT"] = df["debt_held_public_amt"] / 1000000000000
df["RRP"] = df["RRPONTSYD"] / 1000
df["PD_FAILS"] = df["PD_FAILS_ALL"] / 1000000
df["FED_TREASURY"] = df["TREAST"] / 1000000
# Rates to basis points
df["YIELD_CURVE"] = df["T10Y3M"] * 100
df["REPO_TREASURY"] = df["GCF_TREASURY"] * 100
df["LIBOR"] = df["USD_ON_LIBOR"] * 100
df["COLLATERAL_SPREAD"] = df["COL_SPREAD"] * 100
df["GCF_TREASURY"] = df["GCF_TREASURY"] * 100
df["UST_1M"] = df["UST_1M"] * 100

# Fed dates dummies
qe_dates = [
    dtt(2008,11,25), # QE1 Announced
    dtt(2009,3,16), # QE1 Expanded
    dtt(2010,8,10), # QE1 Rollover
    dtt(2010,11,3), # QE2 Announced
    dtt(2011,9,21), # Operation Twist Announced
    dtt(2012,6,20), # Operation Twist Extended
    dtt(2012,9,13), # QE3 Announced & Initiated
    dtt(2012,12,12), # QE3 Expanded
    # dtt(2019,3,8), # Powell: BS endpoint will be highert than before the recession
    dtt(2019,3,20), # Fed announces intent to slow its balance sheet wind-down and then to end it
    # dtt(2019,7,31), # FOMC announces end to balance sheet winddown two months earlier than previously indicated
    dtt(2019,9,18), # Overnight lending repo facility opened
    # dtt(2019,10,11), # FOMC reaffirms Fed’s intention to conduct policy that provides for an ample supply of reserves that does not require active management
    dtt(2020,3,15), # QE4
]

qt_dates = [
    dtt(2010,3,31), # QE1 Terminated
    dtt(2012,12,31), # Operation Twist Terminated
    # dtt(2013,6,19), # QE3 Tapering Discussed
    dtt(2013,12,18), # QE3 Tapering Begins
    # dtt(2014,9,16), # Fed issues normalization plan
    dtt(2014,10,29), # QE3 Terminated
    # dtt(2017,6,14), # Fed signals BS normalization (QT)
    # dtt(2017,6,14), # Fed details normalization plan
    # dtt(2017,9,20), # Fed announces QT to take place in October
    dtt(2017,11,1), # QT has already begun
    # dtt(2018,12,19), # Powell says BS runoff will be on autopilot
    dtt(2020,6,11) # Fed tightenes opeartions in the repo market
]

dates = pd.date_range(start_date,end_date,freq="D")

rate_changes = df["DFEDTAR"].diff()[df["DFEDTAR"].diff() != 0]

rate_up = rate_changes[rate_changes>0].index
rate_up_dates = dates.isin(rate_up)
rate_up_df = pd.Series(data=rate_up_dates,
                       index=dates)\
                    .resample("W-WED").sum()

rate_down = rate_changes[rate_changes<0].index
rate_down_dates = dates.isin(rate_down)
rate_down_df = pd.Series(data=rate_down_dates,
                       index=dates)\
                    .resample("W-WED").sum()

fed_easening = dates.isin(qe_dates)
fed_easening_df = pd.Series(data=fed_easening,
                            index=dates)\
                    .resample("W-WED").sum()

fed_tightening = dates.isin(qt_dates)
fed_tightening_df = pd.Series(data=fed_tightening,
                            index=dates)\
                    .resample("W-WED").sum()

# Taking first differences of the whole data set
ddf = df.diff()

# Add more dummies
df['D_RRP'] = df.eval('RRPONTSYD>104')
ddf["D_RRP"] = df["D_RRP"]

ddf["RRP_RATE"] = df["RRP_RATE"]

# df["REPO_VOL"] = log_vol(df["GCF_TREASURY"])
# ddf["REPO_VOL"] = df["REPO_VOL"] 
#
# df["D_REPO_VOL"] = (df["REPO_VOL"].apply(np.abs) > 0.35) # 80% quantile
# ddf["D_REPO_VOL"] = df["D_REPO_VOL"] 
#
# df["REPO_VOL2"] = hist_vol(df["GCF_TREASURY"])
# ddf["REPO_VOL2"] = df["REPO_VOL2"].apply(np.log).shift()
#
# df["UST10Y_VOL"] = log_vol(df["UST10Y_MID"])
# ddf["UST10Y_VOL"] = df["UST10Y_VOL"] 
#
# df["UST3M_VOL"] = log_vol(df["UST3M_MID"])
# ddf["UST3M_VOL"] = df["UST3M_VOL"] 
#
# df["D_PD_FAILS_ALL"] = (df["PD_FAILS_ALL"].apply(np.abs)>180000) # ?% quantile
# ddf["D_PD_FAILS_ALL"] = df["D_PD_FAILS_ALL"] 

df["RATE_UP"] =  rate_up_df.astype("bool")
ddf["RATE_UP"] =  df["RATE_UP"]

df["RATE_DOWN"] =  rate_down_df
ddf["RATE_DOWN"] =  df["RATE_DOWN"]

df["FED_EASENING"] =  fed_easening_df.astype("bool")
ddf["FED_EASENING"] =  df["FED_EASENING"]

df["FED_TIGHTENING"] =  fed_tightening_df.astype("bool")
ddf["FED_TIGHTENING"] =  df["FED_TIGHTENING"]

ddf["SLR"] = 0
ddf["SLR"]["01-01-2018":] = 1
ddf["SLR"]["01-04-2020":"31-03-2021"] = 0
ddf["SLR"] = ddf["SLR"].astype(bool)

# --- OLS specifications --- 
n_lag = int(726**(1/4))

# 1_1: Base
spec11 = ols("REPO_TREASURY ~\
                SOMA_TREASURY +\
                DEBT",
            data=ddf,missing='drop',hasconst=True).\
                fit(cov_type="HAC",cov_kwds={'maxlags': n_lag})
res11 = spec11.summary()
res11

# 1_2: Base 2
spec12 = ols("REPO_TREASURY ~\
                SOMA_TREASURY +\
                DEBT +\
                RRP +\
                UST_1M +\
                YIELD_CURVE +\
                C(RATE_DOWN) +\
                C(RATE_UP)",
            data=ddf,missing='drop',hasconst=True)\
                .fit(cov_type="HAC",cov_kwds={'maxlags': n_lag})
res12 = spec12.summary()
res12

# 1_3: Add LIBOR
spec13 = ols("REPO_TREASURY ~\
                SOMA_TREASURY +\
                DEBT +\
                RRP +\
                UST_1M +\
                YIELD_CURVE +\
                LIBOR +\
                C(RATE_DOWN) +\
                C(RATE_UP)",
            data=ddf,missing='drop',hasconst=True)\
                .fit(cov_type="HAC",cov_kwds={'maxlags': n_lag})
res13 = spec13.summary()
res13

# 1_4: Add VIX and Fed dates
spec14 = ols("REPO_TREASURY ~\
                SOMA_TREASURY +\
                DEBT +\
                RRP +\
                UST_1M +\
                YIELD_CURVE +\
                LIBOR +\
                VIX +\
                C(RATE_DOWN) +\
                C(RATE_UP) +\
                C(FED_EASENING) +\
                C(FED_TIGHTENING)",
            data=ddf,missing='drop',hasconst=True)\
                .fit(cov_type="HAC",cov_kwds={'maxlags': n_lag})
res14 = spec14.summary()
res14

# 2_1: Collateral Spread Specfication
spec21 = ols("COLLATERAL_SPREAD ~\
                SOMA_TREASURY +\
                DEBT",
            data=ddf,missing='drop',hasconst=True)\
                .fit(cov_type="HAC",cov_kwds={'maxlags': n_lag})
res21 = spec21.summary()
res21

# 2_2: Add significant variables
spec22 = ols("COLLATERAL_SPREAD ~\
                SOMA_TREASURY +\
                DEBT +\
                RRP +\
                UST_1M +\
                YIELD_CURVE +\
                LIBOR +\
                C(RATE_DOWN) +\
                C(RATE_UP)",
            data=ddf,missing='drop',hasconst=True)\
                .fit(cov_type="HAC",cov_kwds={'maxlags': n_lag})
res22 = spec22.summary()
res22

# 2_3: Add VIX and Fed dates
spec23 = ols("COLLATERAL_SPREAD ~\
                SOMA_TREASURY +\
                DEBT +\
                RRP +\
                UST_1M +\
                YIELD_CURVE +\
                LIBOR +\
                VIX +\
                C(RATE_DOWN) +\
                C(RATE_UP) +\
                C(FED_EASENING) +\
                C(FED_TIGHTENING)",
            data=ddf,missing='drop',hasconst=True)\
                .fit(cov_type="HAC",cov_kwds={'maxlags': n_lag})
res23 = spec23.summary()
res23

# 2_4: Add PD Fails
spec24 = ols("COLLATERAL_SPREAD ~\
                SOMA_TREASURY +\
                DEBT +\
                RRP +\
                UST_1M +\
                YIELD_CURVE +\
                LIBOR +\
                C(RATE_DOWN) +\
                C(RATE_UP) +\
                PD_FAILS",
            data=ddf,missing='drop',hasconst=True)\
                .fit(cov_type="HAC",cov_kwds={'maxlags': n_lag})
res24 = spec24.summary()
res24

# 2_5: Add 3M UST Vol
spec25 = ols("COLLATERAL_SPREAD ~\
                SOMA_TREASURY +\
                DEBT +\
                RRP +\
                UST_1M +\
                YIELD_CURVE +\
                LIBOR +\
                C(RATE_DOWN) +\
                C(RATE_UP) +\
                UST3M_10W_VOL",
            data=ddf,missing='drop',hasconst=True)\
                .fit(cov_type="HAC",cov_kwds={'maxlags': n_lag})
res25 = spec25.summary()
res25

# 2_6: Add UST 3M Bid-Ask Spread
spec26 = ols("COLLATERAL_SPREAD ~\
                SOMA_TREASURY +\
                DEBT +\
                RRP +\
                UST_1M +\
                YIELD_CURVE +\
                LIBOR +\
                C(RATE_DOWN) +\
                C(RATE_UP) +\
                UST3M_BIDASK",
            data=ddf,missing='drop',hasconst=True)\
                .fit(cov_type="HAC",cov_kwds={'maxlags': n_lag})
res26 = spec26.summary()
res26

# 4: Bills only
spec4 = ols("GCF_TREASURY ~\
                SOMA_BILLS +\
                DEBT +\
                RRP +\
                UST_1M +\
                YIELD_CURVE +\
                LIBOR +\
                C(RATE_DOWN) +\
                C(RATE_UP)",
            data=ddf,missing='drop',hasconst=True)\
                .fit(cov_type="HAC",cov_kwds={'maxlags': n_lag})
res4 = spec4.summary()
res4

# 5: Explaining RRP
spec5 = ols("RRP ~\
                C(RRP_RATE) +\
                UST_1M",
            data=ddf,missing='drop',hasconst=True)\
                .fit(cov_type="HAC",cov_kwds={'maxlags': 4})
res5 = spec5.summary()
res5
df[["RRP","UST_1M"]].corr()

# Export results
sg1 = Stargazer([spec11,spec12,spec13,spec14])
sg2 = Stargazer([spec21,spec22,spec23])
sg24 = Stargazer([spec24,spec25,spec26])

with open(tab_path+'reg1.txt', 'w') as f:
     f.write(sg1.render_latex())
with open(tab_path+'reg2.txt', 'w') as f:
     f.write(sg2.render_latex())
with open(tab_path+'reg24.txt', 'w') as f:
     f.write(sg24.render_latex())

# --- Statistics --- 
df_stats = df[["GCF_TREASURY","COLLATERAL_SPREAD","LIBOR",
    "UST_1M", "YIELD_CURVE","SOMA_TREASURY",
    "DEBT", "RRP","VIX","PD_FAILS"]]
stats = df_stats.describe().T\
                [["mean","min","max","std","count"]]\
                .rename({"count": "obs"},axis=1)\
                .round(2)
stats.obs = stats.obs.astype(int)

with open(tab_path+'stats.txt', 'w') as f:
     f.write(stats.to_latex())

coll = pd.read_excel("data/repledged.xlsx",
                          index_col="Date")
coll["sum"] = coll.iloc[:,0] + coll.iloc[:,1]

# --- Figures --- 


# dtcc = pd.read_csv("data/dtcc.csv",index_col="Date") / 1000000000
# dtcc.index = pd.to_datetime(dtcc.index)
# dtcc = dtcc.loc[:"2022-01-02"]
# f, ax = plt.subplots(1,1)
# ax.stackplot(dtcc.index,
#              dtcc["Treasury Total PAR Value"])

# Figure 2: Main variables
f_2, ax = plt.subplots(4,1,figsize=[15,12])
for i in range(4):
    ax[i].set_xlim(dtt(2009,1,1),dtt(2021,12,30))
ax[0].plot(df["GCF_TREASURY"],c='k',
           label="_nolegend_")
ax[0].plot(df["DFEDTAR"]*100,c='gray',
           ls='--',alpha=0.6,
           label="Upper target of the Fed funds rate, bps")
ax[0].legend()
ax[0].set_ylim(-30,300)
ax[1].plot(df["COLLATERAL_SPREAD"],c='k')
ax[1].axhline(0,c='gray',ls='--',alpha=0.6)
ax[1].set_ylim(-47,47)
ax[2].plot(df["SOMA_TREASURY"],c='k')
ax[3].plot(df["DEBT"],c='k')
ax[0].set_title("GCF Treasury Repo Rate, bps")
ax[1].set_title("US Collateral Spread, bps")
ax[2].set_title("SOMA Treasury holdings, tril USD")
ax[3].set_title("Treasury debt outstanding, tril USD")
sns.despine()
plt.tight_layout()
f_1.savefig(fig_path+"main_vars.pdf")






