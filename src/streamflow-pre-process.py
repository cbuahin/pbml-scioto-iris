import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

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
inp = (inps.inp1.fillna(0) + inps.inp2.fillna(0)) / (inps.inp1.notna() * inp1ar + inps.inp2.notna() * inp2ar) * nodear
inp.index.name = "datetime"
out = outratio * rawout.resample('15min').mean().interpolate(limit=8).resample("1h").asfreq()


filled = node.fillna(out).fillna(inp)
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
