# -*- coding: utf-8 -*-
"""CS246 - Colab 4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1UWeDiyXiwDDqe7ksN2kt-myHsuSLObv8

# CS246 - Colab 4
## Collaborative Filtering

### Setup

Let's setup Spark on your Colab environment.  Run the cell below!
"""

!pip install pyspark
!pip install -U -q PyDrive
!apt install openjdk-8-jdk-headless -qq
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"

"""Now we authenticate a Google Drive client to download the filea we will be processing in our Spark job.

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

id='1QtPy_HuIMSzhtYllT3-WeM3Sqg55wK_D'
downloaded = drive.CreateFile({'id': id})
downloaded.GetContentFile('MovieLens.training')

id='1ePqnsQTJRRvQcBoF2EhoPU8CU1i5byHK'
downloaded = drive.CreateFile({'id': id})
downloaded.GetContentFile('MovieLens.test')

id='1ncUBWdI5AIt3FDUJokbMqpHD2knd5ebp'
downloaded = drive.CreateFile({'id': id})
downloaded.GetContentFile('MovieLens.item')

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

In this Colab, we will be using the [MovieLens dataset](https://grouplens.org/datasets/movielens/), specifically the 100K dataset (which contains in total 100,000 ratings from 1000 users on ~1700 movies).

We load the ratings data in a 80%-20% ```training```/```test``` split, while the ```items``` dataframe contains the movie titles associated to the item identifiers.
"""

schema_ratings = StructType([
    StructField("user_id", IntegerType(), False),
    StructField("item_id", IntegerType(), False),
    StructField("rating", IntegerType(), False),
    StructField("timestamp", IntegerType(), False)])

schema_items = StructType([
    StructField("item_id", IntegerType(), False),
    StructField("movie", StringType(), False)])

training = spark.read.option("sep", "\t").csv("MovieLens.training", header=False, schema=schema_ratings)
test = spark.read.option("sep", "\t").csv("MovieLens.test", header=False, schema=schema_ratings)
items = spark.read.option("sep", "|").csv("MovieLens.item", header=False, schema=schema_items)

training.printSchema()

items.printSchema()

"""### Your task

Let's compute some stats!  What is the number of ratings in the training and test dataset? How many movies are in our dataset?
"""

# YOUR CODE HERE
print("training set shape is: ", (training.count(), len(training.columns)))
print("testing set shape is: ", (test.count(), len(test.columns)))

"""Using the training set, train a model with the Alternating Least Squares method available in the Spark MLlib: [https://spark.apache.org/docs/latest/ml-collaborative-filtering.html](https://spark.apache.org/docs/latest/ml-collaborative-filtering.html)"""

# YOUR CODE HERE
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

# Build the recommendation model using ALS on the training data
# Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
als = ALS(rank=10, maxIter=10, regParam=0.1, userCol="user_id", itemCol="item_id", ratingCol="rating",
          coldStartStrategy="drop")
model = als.fit(training)

"""Now compute the RMSE on the test dataset."""

# YOUR CODE HERE
# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

# Build the recommendation model using ALS on the training data
# Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
als = ALS(rank=100, maxIter=10, regParam=0.1, userCol="user_id", itemCol="item_id", ratingCol="rating",
          coldStartStrategy="drop")
model = als.fit(training)

# YOUR CODE HERE
# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

regParam_list = [1, 0.3, 0.1, 0.03, 0.01]

for regParam in regParam_list:
  # Build the recommendation model using ALS on the training data
  # Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
  als = ALS(rank=100, maxIter=10, regParam=regParam, userCol="user_id", itemCol="item_id", ratingCol="rating",
            coldStartStrategy="drop")
  model = als.fit(training)

  # YOUR CODE HERE
  # Evaluate the model by computing the RMSE on the test data
  predictions = model.transform(test)
  evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                  predictionCol="prediction")
  rmse = evaluator.evaluate(predictions)
  print("Regularization parameter = {}, Root-mean-square error = {}".format(regParam, str(rmse)))

"""At this point, you can use the trained model to produce the top-K recommendations for each user."""

# YOUR CODE HERE
# Build the recommendation model using ALS on the training data
# Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
als = ALS(rank=100, maxIter=10, regParam=0.1, userCol="user_id", itemCol="item_id", ratingCol="rating",
          coldStartStrategy="drop")
model = als.fit(training)

# YOUR CODE HERE
# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)

# Generate top 1 movie recommendations for each user
userRecs = model.recommendForAllUsers(1)

userRecs.select('user_id', expr("recommendations['item_id']")[0].alias("item_id")).groupBy('item_id').count().sort(desc('count')).take(5)

items.filter(items.item_id == 1449).take(5)

"""Once you have working code for each cell above, **head over to Gradescope, read carefully the questions, and submit your solution for this Colab**!"""