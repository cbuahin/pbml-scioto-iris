import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np


# 1629 square miles
nodear = 1629
df = pd.read_csv("data/streamflow/03227500.csv", header=None)
raw = df.iloc[:, 4]
raw.index = pd.to_datetime(df.iloc[:, 2])

inp1ar = 980
inp1ratio = nodear / inp1ar
inp1 = pd.read_csv("data/streamflow/03221000.csv", header=None)
rawinp1 = inp1.iloc[:, 4]
rawinp1.index = pd.to_datetime(inp1.iloc[:, 2])

inp2ar = 533
inp2ratio = nodear / inp2ar
inp2 = pd.read_csv("data/streamflow/03227107.csv", header=None)
rawinp2 = inp2.iloc[:, 4]
rawinp2.index = pd.to_datetime(inp2.iloc[:, 2])

outar = 2272
outratio = nodear / outar
out = pd.read_csv("data/streamflow/03229610.csv", header=None)
rawout = out.iloc[:, 4]
rawout.index = pd.to_datetime(out.iloc[:, 2])

node = raw.resample('15min').mean().interpolate(limit=8).resample("1h").asfreq()
inps = pd.DataFrame({
    "inp1": rawinp1.resample('15min').mean().interpolate(limit=8).resample("1h").asfreq(),
    "inp2": rawinp2.resample('15min').mean().interpolate(limit=8).resample("1h").asfreq()
},
                    index = node.index
)
# inp = (inps.inp1.fillna(0) + inps.inp2.fillna(0)) / (inps.inp1.notna() * inp1ar + inps.inp2.notna() * inp2ar) * nodear
inp = (inps.inp1 + inps.inp2) / ( inp1ar + inp2ar) * nodear
inp.index.name = "datetime"
out = outratio * rawout.resample('15min').mean().interpolate(limit=8).resample("1h").asfreq()


plt.plot(inp.index, inp)
plt.plot(node.index, node)

fillfrom = pd.DataFrame({"node": node, "inputs": inp, "output": out, "input 1": inps.inp1, "input 2": inps.inp2}, index=node.index)
corrs = fillfrom.corr().loc["node"]

fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
for i, col in enumerate(['inputs', 'output', 'input 1', 'input 2']):
    j = i // 2
    i = i % 2
    corr = corrs.loc[col]
    ax[i, j].scatter(fillfrom.node, fillfrom.loc[:, col], s= 2, alpha = 0.3)
    ax[i, j].plot([0, 7e4], [0, 7e4], c="red")
    ax[i, j].set_title(f"{col} (r = {corr:.3})")
    if i == 1:
        ax[i, j].set_xlabel("node discharge")
    if j == 0:
        ax[i, j].set_ylabel("inputs discharge")

plt.show()



filled = node.fillna(inp).fillna(out).fillna(inps.inp1 / inp1ar * nodear).fillna(inps.inp2 / inp2ar * nodear)

cat = lambda v: (lambda s: pd.NA if pd.isna(s) else v)

fillcat = node.map(cat("node")).fillna(inp.map(cat("inputs"))).fillna(out.map(cat("output"))).fillna(inps.inp1.map(cat("input 1"))).fillna(inps.inp2.map(cat("input 2"))).fillna("N/A")

pd.DataFrame({"count" : fillcat.value_counts(), "correlation": corrs}).sort_values("count", ascending=False)
#             count  correlation
# node     280237.0     1.000000
# input 1   14611.0     0.947286
# N/A        2656.0          NaN
# output     1324.0     0.963151
# inputs      155.0     0.981310
# input 2       NaN     0.776993

node.isna().sum()
filled.isna().sum()

fig = make_subplots(rows=2, cols=1,
                    shared_xaxes=True,
                    shared_yaxes=True,
                    vertical_spacing=0.02)
fig.add_trace(go.Scatter(x=raw.index, y=raw, name="raw"), row=1, col=1)
fig.add_trace(go.Scatter(x=filled.index, y=filled, name="filled"), row=2, col=1)
fig.add_trace(go.Scatter(x=node.index, y=node, name="node"), row=2, col=1)

fig.show()
# fig.write_html("/tmp/streamflow.html")

fig = make_subplots(rows=4, cols=1,
                    shared_xaxes=True,
                    shared_yaxes=True,
                    vertical_spacing=0.02)
# fig.add_trace(go.Scatter(x=raw.index, y=raw, name="raw"), row=1, col=1)
fig.add_trace(go.Scatter(x=node.index, y=node, name="node"), row=1, col=1)
fig.add_trace(go.Scatter(x=inp.index, y=inp, name="inp"), row=2, col=1)
fig.add_trace(go.Scatter(x=out.index, y=out, name="out"), row=3, col=1)
# fig.add_trace(go.Scatter(x=inp.index, y=inp, name="inp"), row=4, col=1)
# fig.add_trace(go.Scatter(x=out.index, y=out, name="out"), row=4, col=1)
fig.add_trace(go.Scatter(x=filled.index, y=filled, name="filled"), row=4, col=1)
fig.add_trace(go.Scatter(x=node.index, y=node, name="node"), row=4, col=1)

fig.show()

filled.to_csv("data/streamflow-filled.csv")
