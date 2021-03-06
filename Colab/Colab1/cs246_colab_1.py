# -*- coding: utf-8 -*-
"""CS246 - Colab 1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1OYY1n6iSrpP7ET5H2PUchl6xEMnC1boD

# CS246 - Colab 1
## Wordcount in Spark

### Setup

Let's setup Spark on your Colab environment.  Run the cell below!
"""

!pip install pyspark
!pip install -U -q PyDrive
!apt install openjdk-8-jdk-headless -qq
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"

"""Now we authenticate a Google Drive client to download the file we will be processing in our Spark job.

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

id='1SE6k_0YukzGd5wK-E4i6mG83nydlfvSa'
downloaded = drive.CreateFile({'id': id})
downloaded.GetContentFile('pg100.txt')

"""If you executed the cells above, you should be able to see the file *pg100.txt* under the "Files" tab on the left panel.

### Your task

If you run successfully the setup stage, you are ready to work on the *pg100.txt* file which contains a copy of the complete works of Shakespeare.

Write a Spark application which outputs the number of words that start with each letter. This means that for every letter we want to count the total number of (non-unique) words that start with a specific letter. In your implementation **ignore the letter case**, i.e., consider all words as lower case. Also, you can ignore all the words **starting** with a non-alphabetic character.
"""

from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark import SparkContext
import pandas as pd

# create the Spark Session
spark = SparkSession.builder.getOrCreate()

# create the Spark Context
sc = spark.sparkContext

# YOUR
words = sc.textFile("pg100.txt").map(lambda line: line.lower()).flatMap(lambda line: line.split(" "))
words_mapped = words.map(lambda x: (x[0:1],1))

# CODE
words_grouped = words_mapped.reduceByKey(lambda a,b:a +b).collect()

# HERE
keys = ['a', 'd', 'e', 'j', 'k', 'n', 'q', 't', 'y', 'z']

for k_ in keys:
  for k, v in words_grouped:
    if k_ == k:
      print(k, v)

"""Once you obtained the desired results, **head over to Gradescope and submit your solution for this Colab**!"""