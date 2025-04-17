import os
from typing import List
import pandas as pd
import numpy as np
import seaborn as sns
import setuptools.dist
from tensorflow.keras import layers, activations, callbacks, optimizers, models
from tensorflow.keras import Model
from tensorflow.keras.activations import relu
import tensorflow_probability as tfp
import tensorflow.keras.backend as K
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# this library is not upto date and couldn't use monotonicity using this
# from airt.keras.layers import MonoDense

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

precip_fields = ['group_0','group_1','group_2','group_3']
endogenous_fields = ["streamflow"]

df_org = pd.read_csv(os.path.join(HERE, "../data/combined-all.csv"), index_col="datetime", parse_dates=True)
df = df_org.copy(deep=True)

trend = df[endogenous_fields].rolling(pd.Timedelta(hours=24), center=True).mean()

new_columns = {}

for field in precip_fields:
    new_columns[f'{field}_3h'] = df[field].rolling(pd.Timedelta(hours=3)).sum()
    new_columns[f'{field}_6h'] = df[field].rolling(pd.Timedelta(hours=6)).sum()
    new_columns[f'{field}_9h'] = df[field].rolling(pd.Timedelta(hours=9)).sum()
    new_columns[f'{field}_12h'] = df[field].rolling(pd.Timedelta(hours=12)).sum()
    new_columns[f'{field}_15h'] = df[field].rolling(pd.Timedelta(hours=15)).sum()
    new_columns[f'{field}_18h'] = df[field].rolling(pd.Timedelta(hours=18)).sum()
    new_columns[f'{field}_21h'] = df[field].rolling(pd.Timedelta(hours=21)).sum()
    new_columns[f'{field}_24h'] = df[field].rolling(pd.Timedelta(hours=24)).sum()
    new_columns[f'{field}_36h'] = df[field].rolling(pd.Timedelta(hours=36)).sum()
    new_columns[f'{field}_48h'] = df[field].rolling(pd.Timedelta(hours=48)).sum()
    new_columns[f'{field}_72h'] = df[field].rolling(pd.Timedelta(hours=72)).sum()
    new_columns[f'{field}_7d'] = df[field].rolling(pd.Timedelta(days=7)).sum()
    new_columns[f'{field}_14d'] = df[field].rolling(pd.Timedelta(days=14)).sum()
    new_columns[f'{field}_30d'] = df[field].rolling(pd.Timedelta(days=30)).sum()
    new_columns[f'{field}_60d'] = df[field].rolling(pd.Timedelta(days=60)).sum()

# Add all new columns to the DataFrame at once
df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)

# Update precip_fields
precip_fields.extend(new_columns.keys())
exogenous_fields = [*precip_fields]

def add_seasonality_signals(
        data: pd.DataFrame,
        seasonality_signals: List[str],
        add_sine: bool = True,
        add_cosine: bool = True
):
    """
    Add seasonality signals to the data
    :return:
    """
    d = data.copy()

    for signal in seasonality_signals:
        if signal == 'hour_of_day':
            hour_of_day = d.index.hour + d.index.minute / 60 + d.index.second / 3600
            d['hour_of_day'] = hour_of_day
            if add_sine:
                d['hour_of_day_sin'] = np.sin(2 * np.pi * hour_of_day / 24)
            if add_cosine:
                d['hour_of_day_cos'] = np.cos(2 * np.pi * hour_of_day / 24)
        elif signal == 'day_of_week':
            hour_of_day = d.index.hour + d.index.minute / 60 + d.index.second / 3600
            d['day_of_week'] = d.index.dayofweek + hour_of_day / 24
            if add_sine:
                d['day_of_week_sin'] = np.sin(2 * np.pi * d['day_of_week'] / 7)
            if add_cosine:
                d['day_of_week_cos'] = np.cos(2 * np.pi * d['day_of_week'] / 7)
        elif signal == 'day_of_year':
            hour_of_day = d.index.hour + d.index.minute / 60 + d.index.second / 3600
            day_of_year = d.index.dayofyear + hour_of_day / 24
            d['day_of_year'] = day_of_year
            if add_sine:
                d['day_of_year_sin'] = np.sin(2 * np.pi * day_of_year / 365)
            if add_cosine:
                d['day_of_year_cos'] = np.cos(2 * np.pi * day_of_year / 365)

    return d

df = add_seasonality_signals(df, seasonality_signals =['day_of_year'])
df.dropna(inplace=True)

exogenous_fields.append('day_of_year_sin')
exogenous_fields.append('day_of_year_cos')

x = df[exogenous_fields]
y = df[endogenous_fields]

from sklearn import set_config
set_config(transform_output="pandas")

scalar_x = MinMaxScaler().fit(x)
scalar_y = MinMaxScaler().fit(y)

x_scaled = scalar_x.transform(x)
y_scaled = scalar_y.transform(y)

# X = df.to_numpy()
# y = s.loc[df.index].values
# N = X.shape[1]
#
# X.shape, y.shape
dropout = 0.3

# random split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.3, shuffle=False)
x_train_a, x_valid, y_train_a, y_valid = train_test_split(x_train, y_train, test_size=0.3, shuffle=False)

precip_perturbations = np.arange(1.0, 1.5+0.05, 0.05)
x_train_a= np.repeat(x_train_a.values[:, np.newaxis, :], len(precip_perturbations), axis=1)


def probabilistic_layer(y_pred):
    distribution = tfp.distributions.Normal(loc=layers.Dense(1)(y_pred[...,0]),
                                            scale=tf.exp(layers.Dense(1)(y_pred[...,1])))
    return distribution

# perturb the rainfall fields
for i, perturbation in enumerate(precip_perturbations):
    x_train_a[:, i, :len(precip_fields)] *= perturbation

model = models.Sequential([
    layers.Input(shape=(None, len(exogenous_fields))),
    layers.Dense(64),
    layers.PReLU(shared_axes=[-1]),
    layers.Dropout(dropout),
    layers.Dense(128),
    layers.PReLU(shared_axes=[-1]),
    layers.Dropout(dropout),
    layers.Dense(128),
    layers.PReLU(shared_axes=[-1]),
    layers.Dropout(dropout),
    layers.Dense(32),
    layers.PReLU(shared_axes=[-1]),
    layers.Dropout(dropout),
    # output layer
    layers.Dense(1, activation="relu"),
    # probabilistic layer
    # layers.Lambda(probabilistic_layer)
    ]
)
# inp_layer = layers.Input(shape=(None, len(exogenous_fields)))
#
# model_layer = layers.Dense(64)(inp_layer)
# model_layer = layers.PReLU(shared_axes=[-1])(model_layer)
# model_layer = layers.Dropout(dropout)(model_layer)
#
# model_layer = layers.Dense(128)(model_layer)
# model_layer = layers.PReLU(shared_axes=[-1])(model_layer)
# model_layer = layers.Dropout(dropout)(model_layer)
#
# model_layer = layers.Dense(128)(model_layer)
# model_layer = layers.PReLU(shared_axes=[-1])(model_layer)
# model_layer = layers.Dropout(dropout)(model_layer)
#
# model_layer = layers.Dense(32)(model_layer)
# model_layer = layers.PReLU(shared_axes=[-1])(model_layer)
# model_layer = layers.Dropout(dropout)(model_layer)
#
# output = layers.Dense(1, activation=layers.PReLU(shared_axes=[-1]))(model_layer)
# model = Model(inp_layer, output)


# test model output
# model.predict(x_train[:5])


def nse_loss(y_true, y_pred):
    if isinstance(y_pred, tfp.distributions.Distribution):
        y_pred_mean = y_pred.mean()
    else:
        y_pred_mean = y_pred

    if len(y_pred_mean.shape) > 2:
        y_pred_mean = y_pred_mean[:,0,...]

    return K.sum((y_pred_mean-y_true)**2)/K.sum((y_true-K.mean(y_true))**2)

def mse_loss(y_true, y_pred):
    if isinstance(y_pred, tfp.distributions.Distribution):
        y_pred_mean = y_pred.mean()
    else:
        y_pred_mean = y_pred

        if len(y_pred_mean.shape) > 2:
            y_pred_mean = y_pred_mean[:,0,...]

    return K.mean(K.square(y_pred_mean - y_true))


def monotonicity_loss(y_true, y_pred):
    """
    Monotonicity loss function
    :param y_true:
    :param y_pred:
    :return:
    """
    if isinstance(y_pred, tfp.distributions.Distribution):
        y_pred_mean = y_pred.mean()
    else:
        y_pred_mean = y_pred

    int_mse_loss = 0


    # y_pred = tf.squeeze(y_pred, axis=-1)
    return int_mse_loss

def negative_log_likelihood(y_true, y_pred):
    """
    Negative log likelihood loss function
    :param y_true:
    :param y_pred:
    :return:
    """
    if isinstance(y_pred, tfp.distributions.Distribution):
        # check shape of y_pred scale
        if len(y_pred.scale.shape) > 2:
            y_true = tf.repeat(y_true[:, tf.newaxis , ...], y_pred.scale.shape[1], axis=1)
            log_likelihood = y_pred.log_prob(y_true)
        else:
            log_likelihood = y_pred.log_prob(y_true)

        return -tf.reduce_mean(log_likelihood)

    return 0.0

def combined_loss(y_true, y_pred):
    """
    Combined loss function
    :param y_true:
    :param y_pred:
    :return:
    """
    mse = mse_loss(y_true, y_pred)
    ngll
    return mse + nse

model.compile(
    optimizer="adam",
    metrics=[nse_loss, mse_loss],
    loss=mse_loss,
    # run_eagerly=True
)

# model._train_counter = 0
# model._test_counter = 0
# model._is_graph_network = False
model.summary()


batch_size = 64
epochs = 100
#
# split2 = X_train.shape[0] * 70 // 100
# split2dt = nonaind[split2]
# # temporal split
# X_train_t = X_train[:split2,]
# X_train_v = X_train[split2:,]
# y_train_t = y_train[:split2]
# y_train_v = y_train[split2:]

hist = model.fit(
    x_train_a, y_train_a,
    batch_size=batch_size,
    epochs=epochs,
    validation_data = (x_valid.values[:, np.newaxis, ...], y_valid),
    callbacks=[
        callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
        ),
        callbacks.CSVLogger(
            os.path.join(HERE, "training.log"), append=True
        ),
        callbacks.TensorBoard(
            log_dir=os.path.join(HERE, "logs"), update_freq="epoch"
        ),
        callbacks.ModelCheckpoint(
            os.path.join(HERE, "model.h5"), save_best_only=True
        ),
        callbacks.TerminateOnNaN(),
    ],

)

plt.plot(hist.history['loss'], label='loss')
plt.plot(hist.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

y_train_pred = model.predict(x_train)
y_test_pred = model.predict(y_train)


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

streamflow = df[endogenous_fields]
y_mean = np.mean(streamflow)
y_std = np.std(streamflow)
sf_norm = lambda f: (f - y_mean)/y_std


y_train_val = (y_train * y_std + y_mean)
y_train_pred_val = (y_train_pred[:,0] * y_std + y_mean)
y_test_val = (y_test * y_std + y_mean)
y_test_pred_val = (y_test_pred[:,0] * y_std + y_mean)

y_train_pred_val[y_train_pred_val < 0] = 0
y_test_pred_val[y_test_pred_val < 0] = 0

y1 = y_train_val
y2 = y_train_pred_val
y3 = y_test_val
y4 = y_test_pred_val

# pd.DataFrame(dict(
#     train = errors(y1, y2),
#     test = errors(y3, y4),
# )).T

# y_train_pred[y_train_pred < 0] = 0
# y_test_pred[y_test_pred < 0] = 0
min_y = np.min(np.concatenate([y1, y2, y3, y4]))
max_y = np.max(np.concatenate([y1, y2, y3, y4]))
plt.scatter(y1, y2, label="train", s=0.2)
plt.scatter(y3, y4, label="test", s=0.2)
plt.plot([min_y, max_y], [min_y, max_y], label="1:1", c="red")
plt.legend()
plt.xlabel("Observed Flow (cfs)")
plt.ylabel("Simulated Flow (cfs)")

plt.show()

obs = pd.concat([
    pd.Series(y_train_val, index=traindt),
    pd.Series(y_test_val, index=testdt)
])
sim = pd.concat([
    pd.Series(y_train_pred_val, index=traindt),
    pd.Series(y_test_pred_val, index=testdt)
])

# obs = obs + trend[obs.index]
# sim = sim + trend[obs.index]

AGG_DAILY = False
# if we want to see aggregates
if AGG_DAILY:
    obs = obs.resample("1d").mean().dropna()
    sim = sim.resample("1d").mean().dropna()

# result = pd.DataFrame(dict(observed=obs, simulated=sim))


# def category(dt):
#     if dt < split2dt:
#         return "train"
#     elif dt < splitdt:
#         return "test"
#     return "validation"


# result.loc[:, "category"] = result.index.map(category)
# result.loc[:, "trend"] = trend
# result.to_csv(f"pbml-results{'-daily' if AGG_DAILY else ''}.csv")

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

high_threshold = result.observed.quantile(0.99)

# pd.DataFrame(dict(
#     train=errors(trainx, trainy),
#     test=errors(testx, testy),
#     valid=errors(validx, validy),
#     train_high=errors(trainx.loc[trainx > high_threshold], trainy.loc[trainx > high_threshold]),
#     test_high=errors(testx.loc[testx > high_threshold], testy.loc[testx > high_threshold]),
#     valid_high=errors(validx.loc[validx > high_threshold], validy.loc[validx > high_threshold]),
# )).T
