# -*- coding: utf-8 -*-
"""CS246 - Colab 9.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/173RadLSqWkfvJTVHFXVlfILfnF6shj4g

# CS246 - Colab 9
## Studying COVID-19

### Setup

Let's setup Spark on your Colab environment.  Run the cell below!
"""

!pip install pyspark
!pip install -U -q PyDrive
!apt install openjdk-8-jdk-headless -qq
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"

"""Now we authenticate a Google Drive client to download the files we will be processing in our Spark job.

**Make sure to follow the interactive instructions.**
"""

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# Authenticate and create the PyDrive client
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

id='1qgUxXG2rVDPuTvvoamqDQVNh0DDZ4LUh'
downloaded = drive.CreateFile({'id': id})
downloaded.GetContentFile('time_series_19-covid-Confirmed.csv')

id='1KMR3I0Mz6XHtv5tsjGIIgyLNS0jmniaA'
downloaded = drive.CreateFile({'id': id})
downloaded.GetContentFile('time_series_19-covid-Deaths.csv')

id='1wKgm-A6p6K79hmtDJKRKuK-Cf1kzsIhw'
downloaded = drive.CreateFile({'id': id})
downloaded.GetContentFile('time_series_19-covid-Recovered.csv')

"""If you executed the cells above, you should be able to see the dataset we will use for this Colab under the "Files" tab on the left panel.

Next, we import some of the common libraries needed for our task.
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

import pyspark
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf

"""Let's initialize the Spark context."""

# create the session
conf = SparkConf().set("spark.ui.port", "4050")

# create the context
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()

"""You can easily check the current version and get the link of the web interface. In the Spark UI, you can monitor the progress of your job and debug the performance bottlenecks (if your Colab is running with a **local runtime**)."""

spark

"""If you are running this Colab on the Google hosted runtime, the cell below will create a *ngrok* tunnel which will allow you to still check the Spark UI."""

!wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
!unzip ngrok-stable-linux-amd64.zip
get_ipython().system_raw('./ngrok http 4050 &')
!curl -s http://localhost:4040/api/tunnels | python3 -c \
    "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"

"""### Data Loading

In this Colab, we will be analyzing the timeseries data of the Coronavirus COVID-19 Global Cases, collected by Johns Hopkins CSSE.

Here you can check a realtime dashboard based on this dataset: [https://www.arcgis.com/apps/opsdashboard/index.html?fbclid=IwAR2hQKsEZ3D38wVtXGryUhP9CG0Z6MYbUM_boPEaV8FBe71wUvDPc65ZG78#/bda7594740fd40299423467b48e9ecf6](https://www.arcgis.com/apps/opsdashboard/index.html?fbclid=IwAR2hQKsEZ3D38wVtXGryUhP9CG0Z6MYbUM_boPEaV8FBe71wUvDPc65ZG78#/bda7594740fd40299423467b48e9ecf6)

---



*   ```confirmed```: dataframe containing the total number of confirmed COVID-19 cases per day, divided by geographical area
*   ```deaths```: dataframe containing the number of deaths per day due to COVID-19, divided by geographical area
*   ```recovered```: dataframe containing the number of recoevered patients per day, divided by geographical area
"""

confirmed = spark.read.csv('time_series_19-covid-Confirmed.csv', header=True)
deaths = spark.read.csv('time_series_19-covid-Deaths.csv', header=True)
recovered = spark.read.csv('time_series_19-covid-Recovered.csv', header=True)

confirmed.printSchema()

"""### Your Task

We are aware of the great deal of stress we are all experiencing because of the spread of the Coronavirus. As such, we decided to conclude our series of Colabs with a **lightweight task** -- given everything you have learned about Spark during the quarter, this Colab should take you just a few minutes to complete.

At the same time, we turned this into an opportunity to raise awareness about the extent of the COVID-19 epidemic.

[Stay healthy, wash your hands often](https://www.cdc.gov/coronavirus/2019-ncov/about/index.html), and invest the time you saved on this Colab to be emotionally supportive to your friends and family!

Signed, *the CS246 teaching staff*
"""

# YOUR CODE HERE
def md5(s):
  import hashlib
  m = hashlib.md5()
  m.update(s.encode())
  return m.hexdigest()[:2]

md5('3/5/20')

confirmed.show()

"""Consider only the most recent data points in the timeseries, and compute:


*   number of confirmed COVID-19 cases across the globe
*   number of deaths due to COVID-19 (across the globe)
*   number of recovered patients across the globe
"""

# YOUR CODE HERE
def list_sum(l):
  sum = 0
  for x in l:
    sum += int(x)
  return sum
confirmed_list = confirmed.select('3/5/20').rdd.flatMap(lambda x: x).collect()
deaths_list = deaths.select('3/5/20').rdd.flatMap(lambda x: x).collect()
recovered_list = recovered.select('3/5/20').rdd.flatMap(lambda x: x).collect()

print("number of confirmed COVID-19 cases across the globe is: {}".format(list_sum(confirmed_list)))
print("number of deaths due to COVID-19 cases across the globe is: {}".format(list_sum(deaths_list)))
print("number of recovered patients across the globe is: {}".format(list_sum(recovered_list)))

"""Consider only the most recent data points in the timeseries, and filter out the geographical locations where less than 50 cases have been confirmed.
For the areas still taken into consideration after the filtering step, compute the ratio between number of recovered patients and number of confirmed cases.
"""

# YOUR CODE HERE
confirmed_list = confirmed.select('Province/State', 'Country/Region', col('3/5/20').alias('confirmed'))
deaths_list = deaths.select('Province/State', 'Country/Region', col('3/5/20').alias('deaths'))
recovered_list = recovered.select('Province/State', 'Country/Region', col('3/5/20').alias('recovered'))
result = confirmed_list.join(deaths_list, ['Province/State', 'Country/Region'])
result = result.join(recovered_list, ['Province/State', 'Country/Region'])
result = result.withColumn("%recovered", result.recovered / result.confirmed)
result = result.withColumn("%deaths", result.deaths / result.confirmed)

"""Following the same filtering strategy as above, now compute the ratio between number of deaths and number of confirmed cases."""

result = result.filter(result.confirmed >= 50).sort("%recovered", ascending = False)
result.show()

result = result.filter(result.confirmed >= 50).sort("%deaths", ascending = False)
result.show()

"""Once you have working code for each cell above, **head over to Gradescope, read carefully the questions, and submit your solution for this Colab**!"""