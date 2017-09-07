import numpy as np
import pandas as pd
import time
import math
import ibmos2spark

import requests
import json

from pyspark import SparkContext
from pyspark import Row
from pyspark.sql import SQLContext
from pyspark.ml.feature import Word2Vec
from pyspark.ml.clustering import KMeans
from pyspark.mllib.linalg import Vectors
from pyspark.ml.feature import PCA

import warnings
warnings.filterwarnings("ignore")
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
import numpy as np
import matplotlib.image as mpimg

import matplotlib.pyplot as plt
%matplotlib inline
from mpl_toolkits.mplot3d import Axes3D

# @hidden_cell
credentials = {
    'auth_url': 'https://identity.open.softlayer.com',
    'project_id': '20a837077b8e4e48b126f71bdef51d70',
    'region': 'dallas',
    'user_id': '25cdd071b2734ecbb079a2029398961d',
    'username': 'member_97487509d827bc34148f5866a767ada3d5dbac5f',
    'password': 'jw8*9)eDpD4!)y,]'
}

configuration_name = 'os_6442159b15f642e6a3fc67707069b86a_configs'
bmos = ibmos2spark.bluemix(sc, credentials, configuration_name)

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
df_data_1 = spark.read\
  .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')\
  .option('header', 'true')\
  .load(bmos.url('WaterOilconsumption', 'Podatki_brez_vikendov.csv'))
df_data_1.take(5)


df_data_1.take(5)

#hidden_cell
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

# This function includes credentials to your Object Storage.
# You might want to remove those credentials before you share your notebook.
def set_hadoop_config(name, creds):
    """This function sets the Hadoop configuration so it is possible to
    access data from Bluemix Object Storage V3 using Spark"""
    prefix = 'fs.swift.service.' + name
    hconf = sc._jsc.hadoopConfiguration()
    hconf.set(prefix + '.auth.url', 'https://identity.open.softlayer.com'+'/v3/auth/tokens')
    hconf.set(prefix + '.auth.endpoint.prefix', 'endpoints')
    hconf.set(prefix + '.tenant', creds['project_id']) 
    hconf.set(prefix + '.username', creds['user_id'])  
    hconf.set(prefix + '.password', creds['password'])  
    hconf.setInt(prefix + '.http.port', 8080)
    hconf.set(prefix + '.region', 'dallas')
    hconf.setBoolean(prefix + '.public', True)

name = 'keystone'
set_hadoop_config(name, credentials)

t0 = time.time()
#datapath = 'swift://'+'VrtecTest'+'.keystone/PRIPRAVLJENI - Podatki za WatsonAnalytics_Jun_2017.xls - List1.csv'
datapath = 'swift://'+'Water_Oil_consumption'+'.keystone/Podatki_brez_vikendov.csv'
#tweets = sqlContext.read.csv(datapath)
#tweets = spark.read.format("csv").option("header", "true").load(datapath) 
#tweets.registerTempTable("tweets")
#twr = tweets.count()
tweets = df_data_1
tweets.show()
#print "Number of tweets read: ", twr 
#print "Elapsed time (seconds): ", time.time() - t0

from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

mapTable = {"Rain": 1.0, "Fog": 2.0, "Clear": 3.0, "Sunny": 4.0, "Thunderstorm": 5.0, "Snow": 6.0}

# define cleaning functions
def mapWeather(v): # reformat the values to get an actual number (e.g., 117,870 kWh to 117870)
    return mapTable[v]
def doNothing(v):
    return v
# Define udf's to apply the defined function to the Spark dataframe
udfMapWeather = udf(mapWeather, DoubleType())
udfDoNothing = udf(doNothing, DoubleType())

df = tweets
#df = tweets.withColumn("Conditions_forecast", udfMapWeather("Conditions_forecast"))
#df = tweets.drop("Conditions_forecast")

df.show()

# use the .toPandas() function to map Spark dataframes to pandas dataframes
dfNp = df.toPandas()

# get the column names of the concatenated dataframe
cols = dfNp.columns
# scale data to prepare for regression model 
from sklearn import preprocessing
scaler = preprocessing.MaxAbsScaler() 


#EDIT:
#delete column that is string
#del dfNp['Property Name']
# get the column names of the concatenated dataframe
cols = dfNp.columns

feat = scaler.fit_transform(dfNp)
# define a new dataframe with the scaled data
dfScaled = pd.DataFrame(feat,columns=cols)

import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')
ff = pd.tools.plotting.scatter_matrix(dfScaled, diagonal='hist',figsize=(30,30))

# get a list of the features used to explain energy
features = dfScaled.columns.tolist()
response = ['Oil_consumption', 'Hot_water_consumption']
features.remove(response[0])
features.remove(response[1])
#features.remove('Conditions_forecast')
# import regression solver
from sklearn import linear_model
from sklearn import isotonic
# declare a linear regression model 
#lr = linear_model.LinearRegression(fit_intercept=False)
lr = linear_model.Ridge (alpha = .9, fit_intercept=True)
# define response variable: energy usage
y = np.asarray(dfScaled[response]) 
# define features
X = dfScaled[features]
# fit regression model to the data
regr = lr.fit(X,y)
coefs = regr.coef_[0]
# collect regression coefficients
#dataRegQ = []
#dataRegQ.append(('Intercept', regr.intercept_[0]))
#for i in range(len(features)):
#    dataRegQ.append((features[i],coefs[i]))
# compute energy predictions using our fitted model 
print features
yh = regr.predict(X)
# import package to compute the R-squared quality metric
from sklearn.metrics import r2_score
# print results
print 'R-Squared: ', r2_score(y,yh)
#pd.DataFrame(dataRegQ,columns=['feature_name','coefficient']) #.head()

fig, ax = plt.subplots()
ax.scatter(y, yh)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Energy Observed',fontsize=20)
ax.set_ylabel('Energy Predicted',fontsize=20)
ax.axis([-0.1, 1.1, -0.1, 1.1])
plt.gcf().set_size_inches( (6, 6) )
plt.show()

def getWeatherData():


    # Weather company data API credentials
    username='b2a97e28-4998-4e2b-9d80-9c148005ff94'
    password='yktrohq4xi'

    # Request forecast for London
    lat = '51.49999473'
    lon = '-0.116721844'
    line='https://'+username+':'+password+'@twcservice.mybluemix.net/api/weather/v1/geocode/46.066798/14.541449/forecast/daily/5day.json?&units=m'
    r=requests.get(line)
    weather = json.loads(r.text)
    #print r
    #print weather

    wdata = []
    
    for i in range(0, 5):
        tmpdata = []
        tmpdata.append(weather["forecasts"][i+1]["max_temp"])
        tmpdata.append(weather["forecasts"][i+1]["min_temp"])
        wdata.append(tmpdata)
    
    #print wdata
    return wdata

#myX = [31.0648,16.0648,24.7518,52.3744,26.9004,28,14,1]
#
#myY = regr.predict(myX)
#print myY

wData = getWeatherData()


myX = []
for i in range(0, 5):
    tmpdata = [0, 0, wData[i][0], wData[i][1]]
    myX.append(tmpdata)
print myX
myX = scaler.transform(myX)
print myX


#myX_scaled = [myX[0,0], myX[0,3], myX[0,4], myX[0,5]]
myX_scaled = [[myX[0,2], myX[0,3]], [myX[1,2], myX[1,3]], [myX[2,2], myX[2,3]], [myX[3,2], myX[3,3]], [myX[4,2], myX[4,3]]]

myY = regr.predict(myX_scaled)
print myY

def convertPrediction(Y, s):
    trainPredict_dataset_like = np.zeros(shape=(s, 4) )
    # put the predicted values in the right field
    for i in range(0, s):
        trainPredict_dataset_like[i,0] = Y[i][0]
        trainPredict_dataset_like[i,1] = Y[i][1]
    print trainPredict_dataset_like
    yhInverse = scaler.inverse_transform(trainPredict_dataset_like)

    return yhInverse


predictions = convertPrediction(myY, 5)
print predictions

for i in range(0, len(predictions)):
    if predictions[i][0] < 0:
        predictions[i][0] = 0.0
    if predictions[i][1] < 0:
        predictions[i][1] = 0.0
print predictions

cloudantdata = spark.read.format("com.cloudant.spark")\
.option("cloudant.host","c09994d8-d57d-4131-aa4a-5dfb5753d009-bluemix.cloudant.com")\
.option("cloudant.username", "c09994d8-d57d-4131-aa4a-5dfb5753d009-bluemix")\
.option("cloudant.password","024a9cc35c0832bbdde4a7375085597ced78e2cb8c4bfcd4d7a9513649c4f2bb")\
.load("analytics")

cloudantdata.printSchema()
cloudantdata.count()
cloudantdata.show()

from pyspark.sql import functions as F
import time
timestamp = int(time.time())

def cloudantSaveData(columnName, saveData):
    global cloudantdata
    cloudantdata = cloudantdata.withColumn(columnName,
        F.when(cloudantdata[columnName] != saveData,saveData).otherwise(cloudantdata[columnName]))

cloudantSaveData("timestamp", timestamp)

cloudantSaveData("day1_water", predictions[0][1])
cloudantSaveData("day1_oil", predictions[0][0])
cloudantSaveData("day1_temp_max", wData[0][0])
cloudantSaveData("day1_temp_min", wData[0][1])

cloudantSaveData("day2_water", predictions[1][1])
cloudantSaveData("day2_oil", predictions[1][0])
cloudantSaveData("day2_temp_max", wData[1][0])
cloudantSaveData("day2_temp_min", wData[1][1])


cloudantSaveData("day3_water", predictions[2][1])
cloudantSaveData("day3_oil", predictions[2][0])
cloudantSaveData("day3_temp_max", wData[2][0])
cloudantSaveData("day3_temp_min", wData[2][1])

cloudantSaveData("day4_water", predictions[3][1])
cloudantSaveData("day4_oil", predictions[3][0])
cloudantSaveData("day4_temp_max", wData[3][0])
cloudantSaveData("day4_temp_min", wData[3][1])

cloudantSaveData("day5_water", predictions[4][1])
cloudantSaveData("day5_oil", predictions[4][0])
cloudantSaveData("day5_temp_max", wData[4][0])
cloudantSaveData("day5_temp_min", wData[4][1])


#Because of 'exceeded 5 calls per minute' limit
time.sleep(1)

cloudantdata.show()

cloudantdata.write.format("com.cloudant.spark")\
.option("cloudant.host","c09994d8-d57d-4131-aa4a-5dfb5753d009-bluemix.cloudant.com")\
.option("cloudant.username", "c09994d8-d57d-4131-aa4a-5dfb5753d009-bluemix")\
.option("cloudant.password","024a9cc35c0832bbdde4a7375085597ced78e2cb8c4bfcd4d7a9513649c4f2bb")\
.save("analytics")
