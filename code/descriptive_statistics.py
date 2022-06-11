#!/usr/bin/env python
# --- Data statistics ---
# --- UZH MA Thesis : Scarcity channel of QE in the US --- 
# --- Autor: Hubert Mrugala (hubertmrugala.com) ---

# Packages
from datetime import datetime as dtt
from datetime import timedelta as tdelta
import numpy as np
import pandas as pd
from pandas.core.arrays.datetimes import sequence_to_datetimes
import statsmodels.api as sm
import pandas_datareader.data as pdr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import seaborn as sns
import os

# Options
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
# Path to the project directory
cwd = os.getenv("THESIS_PATH")
os.chdir(cwd)
fig_path = cwd+"resources/figures/"

start = pd.to_datetime("01-01-2009")
# end = pd.to_datetime("30-05-2021")
end = pd.to_datetime("23-05-2022")
hdr = {'User-Agent': 'Mozilla/5.0'}

## Data
df = pd.read_csv("data/fetched_data.csv",index_col="Date")
df.index = pd.to_datetime(df.index)
# FED BS
fed_bs = pdr.DataReader("WALCL","fred",start,end)/1000000 # in trill
# UT 10Y
us10y =pdr.DataReader("DGS10","fred",start,end)\
        .resample("W-WED").first()
# US CPI
us_cpi = pdr.DataReader("CPIAUCSL","fred",start,end)\
        .pct_change(12)*100
# UST Yields
ust = pd.read_csv("https://home.treasury.gov/system/files/276/yield-curve-rates-1990-2021.csv",
                  index_col="Date",storage_options=hdr).iloc[::-1]
ust.index = pd.to_datetime(ust.index)
# 4-week T-bill Yield
tbill4w = pdr.DataReader("DTB4WK","fred",start,end)
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
dtcc = pd.concat([dtcc_hist_rates,dtcc_rates],axis=0)
dtcc["Spread"] = dtcc["GCF_MBS"] - dtcc["GCF_TREASURY"]
# Collateral allowed to be reused: JPM and GS
coll = pd.read_excel("data/repledged.xlsx",
                          index_col="Date")
coll["sum"] = coll.iloc[:,0] + coll.iloc[:,1]
# Fed's RRP volume
rrp = pdr.DataReader("RRPONTSYD","fred",start,end)
# Fed's RRP rate
rrp_rate = pdr.DataReader("RRPONTSYAWARD","fred",start,end)
# Interest on Reserve Balances
iorb = pdr.DataReader(["IORB","IORR"],"fred",start,end)
iorb_plus = iorb.fillna(0)
iorb_plus["combined"] = iorb_plus["IORB"] + iorb_plus["IORR"]
iorb_plus = iorb_plus["combined"]

## Figure 1: FED BS, 10Y T-Yield & US CPI
f_1, ax = plt.subplots(1,1,)
ax2 = ax.twinx()
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%Y"))
ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
nticks = 6
ax.yaxis.set_major_locator(mtick.LinearLocator(nticks))
ax2.yaxis.set_major_locator(mtick.LinearLocator(nticks))
ax.tick_params(axis='x', rotation=45)
ax.fill_between(fed_bs.index,fed_bs.iloc[:,0],
                alpha=0.4,color='C7')
ax2.plot(us10y,c='k',lw=3)
ax2.plot(us10y.rolling(50).mean(),
         c='gray',ls='--')
ax.set_ylim(0,9)
ax.set_ylabel("Trillion USD",labelpad=15)
ax2.set_ylabel("\%",rotation=-180,labelpad=15)
ax.legend(["The Fed\'s Balance Sheet Size (left)"],
          loc="upper center",
          bbox_to_anchor=[0.15,1.12],
          frameon=False,ncol=1)
ax2.legend(["Yield on 10-year US Treasury (right)",
               "50-week Moving Average of 10Y UST (right)"],
          loc="upper center",
          bbox_to_anchor=[0.65,1.12],
          frameon=False,ncol=2)
ax.set_xlim(start,dtt(2022,2,1))
ax2.set_xlim(start,dtt(2022,2,1))
ax.text(x=0,y=-0.17,s='Source: Federal Reserve Economic Data', transform=ax.transAxes)
plt.tight_layout()
f_1.savefig(fig_path+"fed_bs.pdf")

## Figure 2: Collateral Spread
f_2, ax = plt.subplots(1,1)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.tick_params(axis='x', rotation=45)
ax.plot(df["COL_SPREAD"].loc[start:end]*100,c='k',lw=2)
ax.set_ylabel("bps")
ax.axhline(0,c='gray',ls='--')
ax.set_xlim(start)
ax.text(x=0,y=-0.17,transform=ax.transAxes,
        s='Note: USD collatral spread is defined in this work as\
        a difference between the overnight USD LIBOR rate\
        and the DTCC GCF Treasury Repo rate')
sns.despine()
plt.tight_layout()
f_2.savefig(fig_path+"collateral_spread.pdf")

## Figure 3: Repo Rate and cash market rates
f_3, ax = plt.subplots(2,1)
for i in range(2):
    ax[i].set_xlim(dtt(2020,3,20),dtt(2022,1,1))
    ax[i].set_ylim(-0.04,0.21)
    ax[i].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax[i].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax[0].plot(iorb_plus.interpolate(),c='gray',ls="-.",alpha=0.9)
ax[0].plot(rrp_rate.interpolate(),c='gray',ls="--",alpha=0.9)
ax[0].plot(dtcc,c='k',lw=2)
ax[1].plot(iorb_plus.interpolate(),c='gray',ls="-.",alpha=0.9)
ax[1].plot(rrp_rate.interpolate(),c='gray',ls="--",alpha=0.9)
ax[1].plot(tbill4w.interpolate(),c='k',lw=2)
ax[0].set_title("DTCC O/N GC Treasury Repo Rate")
ax[1].set_title("Yield of the 4-week US Treasury Bill")
ax[1].legend(["Interest on Reserve Balances",
                 "Overnight Reverse Repurchase Agreements Award Rate"],
          loc="upper center",
          bbox_to_anchor=[0.5,-.15],
          frameon=False,ncol=2)
sns.despine()
plt.tight_layout()
f_3.savefig(fig_path+"rates.pdf")

## Figure 4: RRP and GS Repledgeable collateral
f_4, ax = plt.subplots(1,2)
for i in range(2):
    ax[i].tick_params(axis='x', rotation=45)
ax[0].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax[0].set_xlim(dtt(2020,3,1),dtt(2022,1,1))
ax[1].set_xlim(dtt(2008,3,1),dtt(2022,1,1))
ax[0].plot(rrp.interpolate(),c='k',lw=2)
ax[1].plot(coll.iloc[:,1],c='k',lw=2)
ax[0].set_title("The Fed: Treasuries Sold Through O/N RRP")
ax[1].set_title("Goldman Sachs: Collateral available to be repledged")
ax[0].text(x=-0.04,y=-0.17,transform=ax[0].transAxes,
        s='Figures in billions of US dollars.')
sns.despine()
plt.tight_layout()
f_4.savefig(fig_path+"rrp+coll.pdf")

## Figure 5: GCF Repo MBS - Treasury Spread
f_5, ax = plt.subplots(1,1)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.tick_params(axis='x', rotation=45)
ax.plot(dtcc["Spread"]*100,c='k',lw=2)
ax.set_ylabel("bps")
ax.axhline(0,c='gray',ls='--')
ax.set_xlim(dtt(2016,1,1),dtt(2022,1,1))
ax.set_ylim(-75,71)
ax.text(x=0,y=-0.17,s='Source: Depository Trust \& Clearing Corporation', transform=ax.transAxes)
sns.despine()
plt.tight_layout()
f_5.savefig(fig_path+"gcf_spread.pdf")

rrp_spread = pd.concat([tbill4w,rrp_rate],axis=1)
rrp_spread = (rrp_spread["DTB4WK"] - rrp_spread["RRPONTSYAWARD"]).dropna()

rrp_spread_repo = pd.concat([dtcc["GCF_TREASURY"],rrp_rate],axis=1)
rrp_spread_repo = (rrp_spread_repo.iloc[:,0] - rrp_spread_repo["RRPONTSYAWARD"]).dropna()

f_, ax = plt.subplots(2,1)
for i in range(2):
    ax[i].set_xlim(dtt(2016,3,20),dtt(2022,9,15))
    ax[i].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax[i].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax[i].axhline(0,ls='--',color='gray')
ax[0].set_ylim(-0.72,0.55)
ax[1].set_ylim(-0.08,0.7)
ax[0].plot(rrp_spread,c='k',lw=2)
ax[1].plot(rrp_spread_repo,c='k',lw=2)
ax[0].set_title("4-week Treasury bill yield less the RRP award rate")
ax[1].set_title("DTCC GCF Treasury repo rate less the RRP award rate")
sns.despine()
plt.tight_layout()

f_, ax = plt.subplots(2,1)
for i in range(2):
    ax[i].set_xlim(dtt(2014,3,20),dtt(2022,9,15))
    ax[i].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax[i].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax[i].axhline(0,ls='--',color='gray')
ax[0].set_ylim(-0.08,0.7)
ax[0].plot(rrp_spread_repo,c='k',lw=2)
ax[0].set_title("DTCC GCF Treasury repo rate less the RRP award rate")
ax[1].plot(rrp,c='k',lw=2)
sns.despine()
plt.tight_layout()


rrp_spread_repo[rrp_spread_repo<0].loc["2021-06-01":"2022-01-01"]

(rrp_spread_repo[rrp_spread_repo<0].loc["2021-06-01":"2022-01-01"]==-0.035)

rrp_spread_repo.loc["2021-06-01":"2022-01-01"]
