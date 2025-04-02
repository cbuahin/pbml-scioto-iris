import pandas as pd
import os
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np


# looking at the date range, and select the ones that have data from
# at least 2001
files = os.listdir("./data/precip/raw")
selected = []
for fname in files:
    df = pd.read_csv(f"./data/precip/raw/{fname}")
    start = df.Reading.iloc[-1]
    end = df.Reading.iloc[0]
    if start < "2001-09":
        selected.append(fname)
        print("✓", end=" ")
    else:
        print("×", end=" ")
    print(start, end, fname.rsplit(".", maxsplit=1)[0])


# if you have partially done the processing, do this to skip them
# files2 = os.listdir("./data/precip/hourly/")
# selected = list(set(selected).difference(set(files2)))


def load_increm_prec(filename):
    df = pd.read_csv(f"./data/precip/raw/{filename}")
    df.index = pd.to_datetime(df.Reading)
    df.sort_index(inplace=True)
    s = df.Value.copy(deep=True)
    s.name = fname.rsplit(".", maxsplit=1)[0].replace(".", "")


    lower = (s - s.shift(1)) < -1
    blocks = lower.cumsum()
    blk = blocks.reset_index().groupby(s.name).agg(dict(Reading=["min", "max"]))
    blk.columns = blk.columns.droplevel()

    s2 = s.copy(deep=True)
    for d in blk.loc[:, "max"]:
        s2.loc[s2.index > d] += s.loc[d]

    vals = s2.groupby(s2.index.strftime("%Y-%m-%d %H:%M")).mean()
    vals.index = pd.to_datetime(vals.index)
    accum = vals.resample('1min').interpolate(limit=60*12).resample("1h").asfreq()
    increm = accum.diff()
    increm.index.name = "datetime"
    return increm


# # manual fix for errors; as you find more cases, add them to the function

# ASCS.csv: the values between 2006-09-10 and 2006-09-12 (three days)
# alternative between 34 and 35, making a huge amount of fantom
# precipitation; it was manually deleted
# ASCS.csv, Kenlawn_Park.csv, Jackson_Twp_Admin.csv, Johnathan_Alder.csv, Jefferson_Twp_Sewer.csv, Three_Creeks_Park.csv, Alton-Darby.csv, Larue.csv: same, Headley_Park.csv, Mt._Gilead_S.P..csv


def manual_fix(fname, increm):
    match fname:
        case "Kenlawn_Park.csv":
            # the value jumps from 26 to 39 on 2013-07-12
            increm.loc["2013-07-12"] = pd.NA
        case 'Blendon_Woods_Park.csv':
            # same problem here value jumps from 29 to 77 inch in 12 hour
            increm.loc["2013-07-12"] = pd.NA
        case 'Pharisburg.csv':
            # 2014-09-11 has a really high precipitation, 10in per hour, but is divided between multiple readings, there is also no corresponding streamflow
            increm.loc["2014-09-11"] = pd.NA
        case 'Jackson_Twp_Admin.csv':
            # oct 8 to dec 14, lot's of readings with values > 10 per day, then same in feb
            for day in range(8, 32):
                increm.loc[f"2018-10-{day:02}"] = pd.NA
            increm.loc["2018-11"] = pd.NA
            increm.loc["2018-12"] = pd.NA
            increm.loc["2019-01"] = pd.NA
            for day in range(1, 27):
                increm.loc[f"2019-02-{day:02}"] = pd.NA
        case 'Johnathan_Alder.csv':
            # lots of observation in a single hour worth of around 9 inch of precip
            increm.loc["2011-07-22 19:00"] = pd.NA
            increm.loc["2011-07-22 20:00"] = pd.NA
        case 'Three_Creeks_Park.csv':
            increm.loc["2012-03-16"] = pd.NA
            increm.loc["2012-06-01"] = pd.NA
            increm.loc["2013-01-11"] = pd.NA
        case 'Mt._Gilead_S.P..csv':
            # there is a significant negative value
            increm.loc[increm < 0] = 0
    return increm


# len(selected) = 25
ind = 0                         # change ind to try diff files
fname = selected[ind]

# # You can do one by one, or run a loop to save it
# for fname in selected:
increm = load_increm_prec(fname)

# extreme values from noaa
# 1hr 4.05 in; 2hr 5.16 in; 3hr 5.61; 6hr 6.70; 12hr 7.86
extremes = [(1, 4.05), (2, 5.16), (3, 5.61), (6, 6.70), (12, 7.86)]
# and streamflow values to see if precipitation pattern is outlier
sf = pd.read_csv("data/streamflow-filled.csv", index_col="datetime").iloc[:, 0]
sf.index = pd.to_datetime(sf.index)


rollings = {i: increm.rolling(i).sum() for i, _ in extremes}

fig = make_subplots(rows=3 + len(rollings), cols=1,
                    shared_xaxes=True,
                    shared_yaxes=True,
                    vertical_spacing=0.02)
fig.add_trace(go.Scatter(x=sf.index, y=sf, name="streamflow"), row=1, col=1)
fig.add_trace(go.Scatter(x=s.index, y=s, name="raw"), row=2, col=1)
fig.add_trace(go.Scatter(x=s2.index, y=s2, name="accum 1"), row=2, col=1)
fig.add_trace(go.Scatter(x=accum.index, y=accum, name="accum"), row=2, col=1)
fig.add_trace(go.Scatter(x=increm.index, y=increm, name="increm"), row=3, col=1)
for i, (hr, v) in enumerate(extremes):
    rol = rollings[hr]
    fig.add_trace(go.Scatter(x=rol.index, y=rol, name=f"rol {hr}"), row=4+i, col=1)
    fig.add_hline(v, row=4+i, col=1)

for _, r in blk.iterrows():
    # fig.add_vline(r.loc["min"], line_color="purple")
    fig.add_vline(r.loc["max"], line_color="red", line_dash="dash", row=2, col=1)

fig.show()

fig.write_html(f"plots/{fname}.html")
increm.to_csv(f"data/precip/hourly/{fname}")

# IF the manual process with plotting was done already, and you're
# sure there is no problem, you can run the complete loop like this to
# generate all stations at once; uncomment and run

# for fname in selected:
#     increm = load_increm_prec(fname)
#     mfix = manual_fix(fname, increm)
#     mfix.to_csv(f"./hourly/{fname}")

# to combine all the processed precipitation values
precips = []
for fname in os.listdir("./data/precip/hourly"):
    precp = pd.read_csv(f"./data/precip/hourly/{fname}", index_col="datetime").iloc[:, 0]
    precp.index = pd.to_datetime(precp.index)
    precips.append(precp)

# make them into same time range; the range was found looking at their
# available range with the commented line below
precip = pd.DataFrame({s.name: s for s in precips}, index=pd.date_range("2001-08-30", "2024-09-23", freq="1h"))
# precip.isna().sum()

precip.index.name = "datetime"
# save the precipitations
precip.to_csv('data/precip-hourly.csv')

# join the streamflow and save that
combined = precip.join(sf)
combined.to_csv("data/precip-and-sf.csv")
