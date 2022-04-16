#!/usr/bin/env python
# --- Data statistics ---
# --- UZH MA Thesis : Scarcity channel of Quantitative Easing in the US --- 
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
os.chdir("/home/hmrug/projects/thesis/")
wd = os.getcwd()
fig_path = wd+"/resources/figures/"

pd.set_option('display.precision',4)
plt.rcParams.update({
    "figure.dpi": 100,
    "figure.figsize": [15,6],
    "font.size": 14,
    "axes.titlesize": 24,
    "axes.titleweight": "bold",
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
sns.set_style('whitegrid')

start = pd.to_datetime("01-01-2009")
end = pd.to_datetime("30-12-2021")

## Data

## Fetched data
fd = pd.read_csv(wd+"/data/fetched_data.csv",index_col="Date")
fd.index = pd.to_datetime(fd.index)

## UST 
ust = pd.read_csv("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/all/all?type=daily_treasury_yield_curve&field_tdr_date_value=all&page&_format=csv",
                  index_col="Date").iloc[::-1]
ust.index = pd.to_datetime(ust.index)
ust = ust.resample("W-WED").first()

## USD ON LIBOR
libor = pd.read_csv("./data/USD_ON_LIBOR.csv",
                    index_col="DATE")\
                    .rename({"USDONTD156N": "USD_ON_LIBOR"},axis=1)
libor.index = pd.to_datetime(libor.index)
libor = libor.resample("W-WED").first()

## Collateral Spread
coll_spread = pd.concat([libor, fd["Treasury"]],axis=1)
coll_spread = coll_spread["USD_ON_LIBOR"] - coll_spread["Treasury"]

#fd.plot(subplots=True,layout=(4,4))

## Figure 1: Collateral Spread
f_1, ax = plt.subplots(1,1)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.tick_params(axis='x', rotation=45)
ax.plot(coll_spread.loc[start:end]*100,c='k',lw=2)
ax.set_ylabel("bps")
ax.axhline(0,c='gray',ls='--')
ax.set_xlim(start)
#ax.text(x=0,y=-0.3,
        s='USD collatral spread is defined in this work as\
        a difference between the overnight USD LIBOR rate\
        and the DTCC GCF Treasury Repo rate')
sns.despine()
plt.tight_layout()
# f_1.savefig(fig_path+"collateral_spread.pdf")



    



