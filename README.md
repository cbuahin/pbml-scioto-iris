# pbml-scioto-iris
Codes for the Physics based Machine Learning Model for Scioto River for IRiS Fellowship

# Setting Up

## Install Python
use python 3.12 as tensorflow doesn't have support for 3.13 yet (as of writing this).

If you already have Python 3.13 in your system you can use `pyenv` to run different python version than your system and make virtual environments for them.

## Install required packages

Use the following commands to install required python packages for the project. The versions are frozen for reproducibility.

```bash
pip install -r requirements.txt
```

## Download Input Data
We are using streamflow data from USGS, and Precipitation data from https://storms2.storms.ohio.gov/ website. You can download the data from them following the scripts in the `download` directory. You'll need more libraries and setup (selenium for browser automation) for this.

You can also simply use the downloaded and processed data provided with the repository on the `processed` directory and skip those process.

If you want to download data yourself, follow these steps:
### Precipitation Data
Precipitation data were not easy to dowload as there was no clear cut API from the site. So we used `selenium` to automate a browser to download the stations details as well as the precipitation data.

### Streamflow Data
For streamflow, download the data through USGS website: https://dashboard.waterdata.usgs.gov/app/nwd/en/

Here, if you zoom into the map near scioto river, you can see a bunch of streamflow stations. The one that is close to the city is station: 03227500

https://dashboard.waterdata.usgs.gov/api/gwis/2.1.1/service/site?agencyCode=USGS&siteNumber=03227500&open=110328

And to fill the gaps in this, we used the streamflow data for stations upsteam and downstream of it.

For the initial analysis of the network and range of data, streamflow was downloaded for the following list of stations. You can follow the same steps to get all of them.
```python
["03217424", "03219500", "03220000", "03221646", "03225500",
 "03227107", "03228300", "03228689", "03228805", "03229610",
 "03217500", "03219781", "03221000", "03223425", "03226800",
 "03227500", "03228500", "03228750", "03229500"]
```

# Running scripts
## Download Raw data
If you use the data from `processed` directory skip this step.

The script `src/download-precip.py` can be run interactively to open a browser and remotely control it using python. The codes there can extract the stations co-ordinates, which can be put in a GIS software to find the stations that are useful, and the second part will use that list and download the precipitation data for those stations.

## Processing Raw Data
The raw data for streamflow and precipitation are not hourly so we need to work on that.

### Streamflow
For streamflow, out of all the input data, there are different time range and the frequency.


## Running the clustering algorithm

