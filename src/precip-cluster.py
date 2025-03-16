import pandas as pd
import geopandas as gpd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


df = pd.read_csv("data/precip-and-sf.csv", index_col="datetime")
df.index = pd.to_datetime(df.index)


# some stations correlated with each other
sns.heatmap(df.corr())
plt.title("Hourly Correlation")
plt.show()

sns.heatmap(df.groupby(df.index.year).mean().corr())
plt.title("Annual Aggregated Correlation")
plt.show()

df.groupby(df.index.year).mean().corr().loc["streamflow"]
df.rolling(24*300).mean().corr().loc["streamflow"]
df.corr().loc["streamflow"]

sf = df.pop("streamflow")

zeros = df.apply(lambda x: all(v == 0 for v in x), axis=1)
sum(zeros) / len(zeros)         # 35% of the data are all zeros

zeros = df.rolling(12).mean().apply(lambda x: all(v == 0 for v in x), axis=1)
sum(zeros) / len(zeros)         # 25% of the data are zeros for 12 hours

zeros = df.rolling(48).mean().apply(lambda x: all(v < 0.005 for v in x), axis=1)
sum(zeros) / len(zeros)         # 54% of the data are <0.005 for 48 hours

# invalids = pd.date_range('2009-12-10 20:00:00', '2010-04-01 07:00:00', freq="1h")

# df = df.loc[df.index.map(lambda x: x not in invalids),]


pca = PCA(n_components=5)

pca.fit(df.dropna().values)
comps = pca.components_[0:5, :]

km = KMeans(4, max_iter=1000, n_init=100)
km.fit(np.transpose(comps))
km.labels_


plt.scatter(comps[0], comps[1], c=km.labels_)
plt.show()

gages = gpd.read_file("~/class/fall-2024/iris/project/data/gages.gpkg", layer="storms2")

gages.index = gages.name.map(lambda n: n.replace(".", "").replace(" ", "_"))
gages.index.name = "ind"
selected = list(df.columns)
# selected.remove("streamflow")
gages.loc[selected, "cluster"] = km.labels_
gages.to_file("/tmp/clusters.gpkg")

names = pd.Series(gages.index).groupby(gages.cluster.to_list()).agg(list)
for c, n in names.items():
    print(c, n)

# clusters from above, also look at the GIS file for their locations
group1 = ['Big_Walnut_Park', 'Headley_Park', 'Jackson_Twp_Admin', 'Jefferson_Twp_Sewer', 'Pataskala', 'Three_Creeks_Park']
group2 = ['Camp_Lazarus', 'Delaware', 'Mt_Gilead_SP', 'Pharisburg']
group3 = ['Alton-Darby', 'Blendon_Woods_Park', 'Dexter_Falls_Park', 'Frazell_Road', 'Highbanks_Metro_Park', 'Johnathan_Alder', 'Kenlawn_Park', 'Ohio_EMA', 'Raymond_Mem_GC', 'Walnut_St', 'Westerville_Water', 'Woodward_Park']
group4 = ['ASCS', 'Killdeer_Plains_SP', 'Larue']
groups = [group1, group2, group3, group4]

sns.heatmap(df.loc[:, group4].corr())
plt.title("Hourly Correlation")
plt.show()
pass

# from the correlation plots: this was from the old interpolation method; the outliers must have messed things up
# - Alton-Darby has neg correlation with streamflow annually, drop it
# group2.remove('Alton-Darby')
# group2.remove('Headley_Park')
# - group1 even though clustered together don't correlate with each other, keep them separated
# - group 3 has high correlation with each other, average them
# - group 2 show good correlation with each other on most cases, average them

df = pd.read_csv("data/precip-and-sf.csv", index_col="datetime")
df.index = pd.to_datetime(df.index)

# fillval = df.rolling(240).mean()

# for col in df.columns[:-1]:
#     # https://hdsc.nws.noaa.gov/pfds/pfds_map_cont.html?bkmrk=oh
#     outliers = df.loc[:, col] > 4
#     df.loc[outliers, col] = fillval.loc[outliers, col]

dfnew = pd.DataFrame(index=df.index, columns=["g1", "g2", "g3", "g4"])

for col, g in zip(dfnew.columns, groups):
    dfnew.loc[:, col] = df.loc[:, g].mean(axis=1)
# dfnew.loc[:, "group2_q10"] = df.loc[:, group2].quantile(0.1, axis=1)
# dfnew.loc[:, "group2_q50"] = df.loc[:, group2].quantile(0.5, axis=1)
# dfnew.loc[:, "group2_q90"] = df.loc[:, group2].quantile(0.9, axis=1)

len(dfnew.index)
(dfnew.isna().sum(axis=1) == 0).sum()
dfnew.fillna(0.0, limit=4, inplace=True)


sns.heatmap(dfnew.loc[:, ["g1", "g2", "g3", "g4"]].corr())
plt.title("Hourly Correlation")
plt.show()

fillval = dfnew.loc[:, ["g1", "g2", "g3"]].mean(axis=1)
for col in ["g1", "g2", "g3"]:
    ind = dfnew.loc[:, col].isna()
    dfnew.loc[ind, col] = fillval.loc[ind]


dfnew.fillna(0.0, limit=4, inplace=True)


combined = dfnew.join(df.streamflow)
combined.streamflow.isna().sum()
combined.streamflow = combined.streamflow.interpolate(limit=4)
combined.streamflow.isna().sum()

combined.to_csv("data/combined-aggregate-new.csv")
