#import findspark
#findspark.init()
import pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
from pyspark.sql.functions import *

df = spark.read.csv("data/loan.csv", inferSchema = True, header = True)
df.printSchema()

df.show(5)

len(df.columns)