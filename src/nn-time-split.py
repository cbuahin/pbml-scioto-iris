import pandas as pd
import numpy as np
import seaborn as sns
import setuptools.dist
from tensorflow.keras import layers
from tensorflow.keras import Model

import matplotlib.pyplot as plt


df_org = pd.read_csv("data/combined-all.csv", index_col="datetime")
df_org.index = pd.to_datetime(df_org.index)

df = df_org.copy(deep=True)

sf_org = df.pop("streamflow")
sf = sf_org.copy(deep=True)

trend = sf_org.rolling(24, center=False).mean().shift(24)
s2 = (sf - trend)

y_mean = np.mean(sf)
y_std = np.std(sf)
sf_norm = lambda f: (f - y_mean)/y_std

y_mean = np.mean(s2)
y_std = np.std(s2)
s2_norm = lambda f: (f - y_mean)/y_std
s = s2.map(s2_norm)

rol03 = df.rolling(3).mean()
rol06 = df.rolling(6).mean()
rol12 = df.rolling(12).mean()
rol24 = df.rolling(24).mean()
rol24007 = df.rolling(24*7).mean()

lag06 = rol06.shift(6)
lag0612 = rol06.shift(12)
lag12 = rol12.shift(12)

df = df.join(rol03, rsuffix="r03")
df = df.join(rol06, rsuffix="r06")
df = df.join(rol12, rsuffix="r12")
df = df.join(lag06, rsuffix="l06")
df = df.join(lag0612, rsuffix="l0612")
df = df.join(lag12, rsuffix="l12")
df = df.join(rol24, rsuffix="r24")
df = df.join(rol24007, rsuffix="r24007")

# scale precipitation
# df = df.apply(np.sqrt)
Prec_mean = np.mean(df)
Prec_std = np.std(df.dropna().values)
df = (df - Prec_mean)/Prec_std

sf_lag_24 = sf.shift(24).map(sf_norm)
sf_lag_24_06 = sf.shift(24).rolling(6).mean().map(sf_norm)
sf_lag_24_24 = sf.shift(24).rolling(24).mean().map(sf_norm)

df = df.join(sf_lag_24, rsuffix="sf_l24")
df = df.join(sf_lag_24_06, rsuffix="sf_l24_06")
df = df.join(sf_lag_24_24, rsuffix="sf_l24_24")

# seasonality and day of time proxy
numdays = pd.Series(df.index.is_leap_year).map({True: 366, False: 365})
seasonality = (df.index.day_of_year / numdays) * np.pi * 2
timeofday = (df.index.hour + df.index.minute / 60) / 24 * np.pi * 2
proxies = pd.DataFrame(dict(
    season_y=seasonality.map(np.sin),
    season_x=seasonality.map(np.cos),
    time_y=timeofday.map(np.sin),
    time_x=timeofday.map(np.cos),
))
proxies.index = df.index

df = df.join(proxies)

nonaind = df.join(s2, rsuffix="sth").dropna().index
df = df.loc[nonaind, :]
sf = s2.loc[nonaind]

X = df.to_numpy()
y = s.loc[df.index].values
N = X.shape[1]

X.shape, y.shape

# random split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
split = X.shape[0] * 70 // 100
splitdt = nonaind[split]
# temporal split
X_train = X[:split,]
X_test = X[split:,]
y_train = y[:split]
y_test = y[split:]
traindt = nonaind[:split,]
testdt = nonaind[split:,]
# X_train = X[-split:,]
# X_test = X[:-split,]
# y_train = y[-split:]
# y_test = y[:-split]
# traindt = sel.index[sel][-split:,]
# testdt = sel.index[sel][:-split]


inp = layers.Input(shape=(N, ))
x = layers.Dense(128, activation='sigmoid')(inp)
x = layers.Dropout(0.4)(x)
x = layers.Dense(16, activation='linear')(x)
x = layers.Dropout(0.4)(x)
output = layers.Dense(1, activation='linear')(x)

model = Model(inp, output)

model.compile(optimizer="adam", metrics=['mse'], loss="mse")
model.summary()


batch_size = 64
epochs = 20

split2 = X_train.shape[0] * 70 // 100
split2dt = nonaind[split2]
# temporal split
X_train_t = X_train[:split2,]
X_train_v = X_train[split2:,]
y_train_t = y_train[:split2]
y_train_v = y_train[split2:]

hist = model.fit(X_train_t, y_train_t, batch_size=batch_size, epochs=epochs, validation_data = (X_train_v, y_train_v))

plt.plot(hist.history['loss'], label='loss')
plt.plot(hist.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


def errors(true_vals, calc_vals):
    pearsonr_err = np.corrcoef(true_vals, calc_vals)[0,1]
    rmse_err = np.sqrt(((true_vals - calc_vals)**2).sum())
    norm_rmse_err = np.sqrt(
        (((true_vals - calc_vals)/true_vals)**2
         ).sum())
    nse_err = 1 - (
        ((true_vals - calc_vals)**2).sum()
        / ((true_vals - true_vals.mean())**2).sum()
        )
    r = np.corrcoef(true_vals, calc_vals)[1,0]
    α = np.var(calc_vals) / np.var(true_vals)
    β = np.mean(calc_vals) / np.mean(true_vals)
    kge_err = 1 - np.sqrt((1 - r)**2 + (1 - α)**2 + (1 - β)**2)
    return dict(
        pearsonr = pearsonr_err,
        r_square = pearsonr_err**2,
        rmse = rmse_err,
        norm_rmse = norm_rmse_err,
        nse = nse_err,
        kge = kge_err,
    )

y_train_val = (y_train * y_std + y_mean)
y_train_pred_val = (y_train_pred[:,0] * y_std + y_mean)
y_test_val = (y_test * y_std + y_mean)
y_test_pred_val = (y_test_pred[:,0] * y_std + y_mean)

# y_train_pred_val[y_train_pred_val < 0] = 0
# y_test_pred_val[y_test_pred_val < 0] = 0

y1 = y_train_val
y2 = y_train_pred_val
y3 = y_test_val
y4 = y_test_pred_val



# y_train_pred[y_train_pred < 0] = 0
# y_test_pred[y_test_pred < 0] = 0
min_y = np.min(np.concat([y1, y2, y3, y4]))
max_y = np.max(np.concat([y1, y2, y3, y4]))
plt.scatter(y1, y2, label="train", s=0.2)
plt.scatter(y3, y4, label="test", s=0.2)
plt.plot([min_y, max_y], [min_y, max_y], label="1:1", c="red")
plt.legend()
plt.xlabel("Observed Flow (cfs)")
plt.ylabel("Simulated Flow (cfs)")

plt.show()

# th = np.quantile(y1, 0.98)

pd.DataFrame(dict(
    train = errors(y1, y2),
    test = errors(y3, y4),
    # train_high = errors(y1[y1 > th], y2[y1 > th]),
    # test_high = errors(y3[y3 > th], y4[y3 > th]),
    # train_log = errors(np.log10(y1 + 0.01), np.log10(y2 + 0.01)),
    # test_log = errors(np.log10(y3 + 0.01), np.log10(y4 + 0.01)),
)).T

#             pearsonr           rmse   norm_rmse       nse       kge
# train       0.761948  774354.831335  389.194691  0.377393  0.473765
# test        0.870868  205711.619556   58.651485  0.751351  0.725983
# train_high  0.867805  143632.458480    7.806992  0.569324  0.477226
# test_high  -0.068099  129653.627462    6.376401 -4.571290 -0.625438
# train_log   0.926848      52.799216   17.911180  0.855479  0.877141
# test_log    0.919624      25.730268    8.037608  0.838026  0.881235

obs = pd.concat([
    pd.Series(y_train_val, index=traindt),
    pd.Series(y_test_val, index=testdt)
])
sim = pd.concat([
    pd.Series(y_train_pred_val, index=traindt),
    pd.Series(y_test_pred_val, index=testdt)
])

obs = obs + trend[obs.index]
sim = sim + trend[obs.index]

AGG_DAILY = False
# if we want to see aggregates
if AGG_DAILY:
    obs = obs.resample("1d").mean().dropna()
    sim = sim.resample("1d").mean().dropna()

result = pd.DataFrame(dict(observed=obs, simulated=sim))


def category(dt):
    if dt < split2dt:
        return "train"
    elif dt < splitdt:
        return "test"
    return "validation"


result.loc[:, "category"] = result.index.map(category)
result.loc[:, "trend"] = trend
result.to_csv(f"pbml-results{'-daily' if AGG_DAILY else ''}.csv")

# sns.lineplot(sf)
# sns.scatterplot(result, x = "datetime", y="observed", hue="category")
fig, ax1 = plt.subplots() # initializes figure and plots
ax2 = ax1.twinx()

sns.lineplot(result, x="datetime", y="observed", hue="category", ax=ax1)
sns.lineplot(trend.reset_index(), x="datetime", y="streamflow", ax=ax1, dashes=[2, 2], label="trend", color="green")
sns.scatterplot(result, x="datetime", y="simulated", color="red", ax=ax1)
mean_precip = df_org.iloc[:, :-1].mean(axis=1)
mean_precip.name = "precip"
# sns.barplot(pd.DataFrame(mean_precip), x="datetime", y="precip")
plt.show()


sea = sns.FacetGrid(result, col="category")
sea.map(sns.scatterplot, "observed", "simulated", alpha=.8, color=None)
plt.show()

trainx = result.observed.loc[result.category == "train"]
trainy = result.simulated.loc[result.category == "train"]
testx = result.observed.loc[result.category == "test"]
testy = result.simulated.loc[result.category == "test"]
validx = result.observed.loc[result.category == "validation"]
validy = result.simulated.loc[result.category == "validation"]

high_threshold = result.observed.quantile(0.9)

pd.DataFrame(dict(
    train=errors(trainx, trainy),
    test=errors(testx, testy),
    valid=errors(validx, validy),
    train_high=errors(trainx.loc[trainx > high_threshold], trainy.loc[trainx > high_threshold]),
    test_high=errors(testx.loc[testx > high_threshold], testy.loc[testx > high_threshold]),
    valid_high=errors(validx.loc[validx > high_threshold], validy.loc[validx > high_threshold]),
)).T

#        pearsonr           rmse   norm_rmse       nse       kge
# train  0.989274  139680.528093  144.087171  0.978500  0.953362
# test   0.917972  217229.655145  114.806614  0.841511  0.867988
# valid  0.904803  279245.076570   98.015559  0.818137  0.813765

# dfs = pd.DataFrame(dict(obs=sf_org, sim=trend)).dropna()

# pd.DataFrame(dict(
#     train=errors(
#         dfs.obs,
#         dfs.sim
#     ),
# )).T

# #        pearsonr           rmse   norm_rmse       nse      kge
# # train  0.791343  813252.292677  252.063574  0.588785  0.78913

# # So we already have a good enough model with predicting a flat mean for future, our above model should have better results than this


# sns.scatterplot(dfs, x="obs", y="sim")
# plt.show()

# # sensitivity analysis
# sensitivity = {}
# for col in df.columns:
#     df2 = df.loc[sel, :].copy(deep=True)
#     df2.loc[:, col] = np.random.rand(len(df.index))
#     vals = model.predict(df2.values)
#     sim2 = pd.Series(vals[:, 0], index=sel.index[sel])
#     sim2 = sim2 + trend[sim2.index]
#     sensitivity[col] = errors(obs.values, sim2.values)


# base_errors = pd.Series(errors(obs.values, sim.values))
# new_errors = pd.DataFrame(sensitivity).T

# new_errors.nse.plot()

# t = new_errors.nse.quantile(0.25)
# new_errors.sort_values(by="nse", ascending=False).head(20)

# plt.show()
