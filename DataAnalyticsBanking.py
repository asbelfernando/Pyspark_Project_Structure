# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Create a SparkSession
spark = SparkSession.builder.appName("BankingProject").getOrCreate()

# Load the data into a DataFrame
df = spark.read.csv("path/to/data.csv", header=True, inferSchema=True)

# Data cleaning
df = df.dropna()
df = df.dropDuplicates()
df = df.withColumn("age", col("age").cast("integer"))

# Feature engineering
df = df.withColumn("avg_balance", (col("balance") / col("transactions")))
df = df.withColumn("customer_age", (col("year") - col("birth_year")))

# Exploratory Data Analysis
df.describe().show()
df.groupBy("churn").count().show()
df.select("avg_balance").show()

# Prepare data for model building
assembler = VectorAssembler(inputCols=["avg_balance", "customer_age"], outputCol="features")
data = assembler.transform(df)

# Split data into training and test sets
train, test = data.randomSplit([0.7, 0.3])

# Build the model
lr = LogisticRegression(labelCol="churn")

# Set up the parameter grid for cross-validation
paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .addGrid(lr.elasticNetParam, [0.5, 0.8]) \
    .build()

# Set up the evaluator
evaluator = BinaryClassificationEvaluator(labelCol="churn")

# Set up the cross-validator
cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator)

# Train the model
model = cv.fit(train)

# Make predictions on the test set
predictions = model.transform(test)

# Evaluate the model
print("Accuracy: ", evaluator.evaluate(predictions))