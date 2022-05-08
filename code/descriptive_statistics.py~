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

## Data
## Load data
df = pd.read_csv("data/fetched_data.csv",index_col="Date")
df.index = pd.to_datetime(df.index)
# FRED data for Figure 1
fed_bs = pdr.DataReader("WALCL","fred",start,end)/1000000 # in trill
us10y =pdr.DataReader("DGS10","fred",start,end)\
        .resample("W-WED").first()
us_cpi = pdr.DataReader("CPIAUCSL","fred",start,end)\
        .pct_change(12)*100

start = pd.to_datetime("01-01-2009")
end = pd.to_datetime("30-12-2021")

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
ax.text(x=0,y=-0.17,s='Source: FRED', transform=ax.transAxes)
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
# f_2.savefig(fig_path+"collateral_spread.pdf")



    




