from lib import ConfigReader

from pyspark.sql import SparkSession

def get_spark_session(env):
    if env=="LOCAL":
        return SparkSession.builder\
            .config(conf=ConfigReader.get_pyspark_config(env))\
                .master("local[2]")\
                    .getOrCreate()
    else:
        return SparkSession.builder\
            .config(conf=ConfigReader.get_pyspark_config(env))\
                .enableHiveSupport()\
                    .getOrCreate()
