import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("data/field-obs.csv", index_col="date")
rating = pd.read_csv("data/rating.csv")

df.index = pd.to_datetime(df.index)

plt.plot(rating.discharge, rating.depth, c='gray')
plt.scatter(df.discharge, df.height, c=df.index.year, s=5)
plt.colorbar()
# plt.semilogx()
# plt.loglog()
plt.xlabel("Discharge (m³/s)")
plt.ylabel("Height (ft)")
plt.show()
