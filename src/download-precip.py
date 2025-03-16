from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
import pandas as pd


fire = webdriver.Firefox()

## First Step is to get a list of all stations

fire.get("https://storms2.storms.ohio.gov/list/")

elements = fire.find_elements(By.TAG_NAME, "a")

sites = []
for elem in elements:
    if elem.get_property("href").startswith("https://storms2.storms.ohio.gov/site/?site_id="):
        sites.append((elem.text, elem.get_property("href")))


df = pd.DataFrame(index=range(len(sites)), columns=["name", "lat", "lon", "elev", "url"])
for i, (name, url) in enumerate(sites):
    fire.get(url)
    lat, lon = next(l for l in fire.find_elements(By.TAG_NAME, "a") if l.get_property("href").startswith("https://storms2.storms.ohio.gov/map/?find_site_id")).text.split(",")
    try:
        elev = float(next(l for l in fire.find_elements(By.CLASS_NAME, "nowrap") if l.text.startswith("Elevation:")).text.split(":")[1].strip())
    except StopIteration:
        elev = pd.NA
    df.iloc[i, :] = [name, float(lat), float(lon), elev, url]

df.to_csv("data/gis/storms-stations.csv", index=False)
# we import that to GIS and look for only the stations that are in/around our basin

## after spatial analysis, only related gages are taken
# 52 out of 419

import geopandas as gpd
import requests

# Assuming you have the storms points that are relevant in this GIS file
df2 = gpd.read_file("data/gis/gages.gpkg", layer="storms2")


def dataurl(url):
    (a, b) = url.split("/sensor/")
    # I got this link by sneaking around the website network traffic
    return "https://storms2.storms.ohio.gov/export/file/" + b + "&mode=&hours=&data_start=1990-01-01%2000:00:00&data_end=2024-12-31%2023:59:59&tz=" + "US%2FEastern&format_datetime=%25Y-%25m-%25d+%25H%3A%25i%3A%25S&mime=txt&delimiter=comma"


for i, row in df2.iterrows():
    fname = row.loc["name"].replace(" ", "_")
    fire.get(row.url)
    try:
        accum = fire.find_element(By.LINK_TEXT, "Rain Accumulation")
        url = accum.get_property("href")
        r = requests.get(dataurl(url))
        if r.status_code != 200:
            continue
        with open(f"data/precip/raw/{fname}.csv", "wb") as w:
            w.write(r.content)
    except NoSuchElementException:
        pass
