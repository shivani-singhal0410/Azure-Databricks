# Databricks notebook source
# MAGIC %md
# MAGIC # Running experiments in Azure Machine Learning
# MAGIC 
# MAGIC In this lab, you will learn to run experiments in Azure Machine Learning from Azure Databricks. This lab will cover following exercises:
# MAGIC 
# MAGIC - Exercise 1: Running an Azure ML experiment on Databricks
# MAGIC - Exercise 2: Reviewing experiment metrics in Azure ML Studio
# MAGIC 
# MAGIC To install the required libraries please follow the instructions in the lab guide.
# MAGIC 
# MAGIC **Required Libraries**: 
# MAGIC * `azureml-sdk[databricks]` via PyPI
# MAGIC * `sklearn-pandas==2.1.0` via PyPI
# MAGIC * `azureml-mlflow` via PyPI

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Run the following cell to load common libraries.

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

import os
import numpy as np
import pandas as pd
import pickle
import sklearn
import joblib
import math

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn_pandas import DataFrameMapper
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import matplotlib
import matplotlib.pyplot as plt

import azureml
from azureml.core import Workspace, Experiment, Run
from azureml.core.model import Model

print('The azureml.core version is {}'.format(azureml.core.VERSION))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Connect to the AML workspace

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC In the following cell, be sure to set the values for `subscription_id`, `resource_group`, and `workspace_name` as directed by the comments. Please note, you can copy the `subscription ID` and `resource group` name from the **Overview** page on the blade for the Azure ML workspace in the Azure portal.

# COMMAND ----------

#Provide the Subscription ID of your existing Azure subscription
subscription_id = "1e17398b-590b-4bc5-b425-d1e505573710"

#Replace the name below with the name of your resource group
resource_group = "ss_my_resgrp"

#Replace the name below with the name of your Azure Machine Learning workspace
workspace_name = "ss_my_aml"

print("subscription_id:", subscription_id)
print("resource_group:", resource_group)
print("workspace_name:", workspace_name)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC **Important Note**: You will be prompted to login in the text that is output below the cell. Be sure to navigate to the URL displayed and enter the code that is provided. Once you have entered the code, return to this notebook and wait for the output to read `Workspace configuration succeeded`.
# MAGIC 
# MAGIC *Also note that the sign-on link and code only appear the first time in a session. If an authenticated session is already established, you won't be prompted to enter the code and authenticate when creating an instance of the Workspace.*

# COMMAND ----------

ws = Workspace(subscription_id, resource_group, workspace_name)
print(ws)
print('Workspace region:', ws.location)
print('Workspace configuration succeeded')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the training data
# MAGIC 
# MAGIC In this notebook, we will be using a subset of NYC Taxi & Limousine Commission - green taxi trip records available from [Azure Open Datasets]( https://azure.microsoft.com/en-us/services/open-datasets/). The data is enriched with holiday and weather data. Each row of the table represents a taxi ride that includes columns such as number of passengers, trip distance, datetime information, holiday and weather information, and the taxi fare for the trip.
# MAGIC 
# MAGIC Run the following cell to load the table into a Spark dataframe and reivew the dataframe.

# COMMAND ----------

dataset = spark.sql("select * from nyc_taxi").toPandas()
display(dataset)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Exercise 1: Running an Azure ML experiment on Databricks

# COMMAND ----------

# MAGIC %md ### Use MLflow with Azure Machine Learning for Model Training
# MAGIC 
# MAGIC In the subsequent cells you will learn to do the following:
# MAGIC - Set up MLflow tracking URI so as to use Azure ML
# MAGIC - Create MLflow experiment – this will create a corresponding experiment in Azure ML Workspace
# MAGIC - Train a model on Azure Databricks cluster while logging metrics and artifacts using MLflow
# MAGIC 
# MAGIC After this notebook, you should return to the **lab guide** and follow instructions to review the model performance metrics and training artifacts in the Azure Machine Learning workspace.

# COMMAND ----------

# MAGIC %md #### Set MLflow tracking URI
# MAGIC 
# MAGIC Set the MLflow tracking URI to point to your Azure ML Workspace. The subsequent logging calls from MLflow APIs will go to Azure ML services and will be tracked under your Workspace.

# COMMAND ----------

import mlflow
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
print("MLflow tracking URI to point to your Azure ML Workspace setup complete.")

# COMMAND ----------

# MAGIC %md #### Configure experiment

# COMMAND ----------

experiment_name = 'MLflow-AML-Exercise'
mlflow.set_experiment(experiment_name)
print("Experiment setup complete.")

# COMMAND ----------

# MAGIC %md #### Train Model and Log Metrics and Artifacts
# MAGIC 
# MAGIC Now you are ready to train the model. Run the cell below to do the following:
# MAGIC -	Train model
# MAGIC -	Evaluate model
# MAGIC -	Log evaluation metrics
# MAGIC -   Log artifact: Evaluation graph
# MAGIC -   Save model
# MAGIC -   Log artifact: Trained model
# MAGIC 
# MAGIC Note that the metrics and artifacts will be saved in your `AML Experiment Run`.

# COMMAND ----------

print("Training model...")
output_folder = 'outputs'
model_file_name = 'nyc-taxi.pkl'
dbutils.fs.mkdirs(output_folder)
model_file_path = os.path.join('/dbfs', output_folder, model_file_name)

with mlflow.start_run() as run:
  df = dataset.dropna(subset=['totalAmount'])
  x_df = df.drop(['totalAmount'], axis=1)
  y_df = df['totalAmount']

  X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=0)

  numerical = ['passengerCount', 'tripDistance', 'snowDepth', 'precipTime', 'precipDepth', 'temperature']
  categorical = ['hour_of_day', 'day_of_week', 'month_num', 'normalizeHolidayName', 'isPaidTimeOff']

  numeric_transformations = [([f], Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])) for f in numerical]
    
  categorical_transformations = [([f], OneHotEncoder(handle_unknown='ignore', sparse=False)) for f in categorical]

  transformations = numeric_transformations + categorical_transformations

  clf = Pipeline(steps=[('preprocessor', DataFrameMapper(transformations, df_out=True)), 
                        ('regressor', GradientBoostingRegressor())])

  clf.fit(X_train, y_train)
  
  y_predict = clf.predict(X_test)
  y_actual = y_test.values.flatten().tolist()
  
  rmse = math.sqrt(mean_squared_error(y_actual, y_predict))
  mlflow.log_metric('rmse', rmse)
  mae = mean_absolute_error(y_actual, y_predict)
  mlflow.log_metric('mae', mae)
  r2 = r2_score(y_actual, y_predict)
  mlflow.log_metric('R2 score', r2)
  
  plt.figure(figsize=(10,10))
  plt.scatter(y_actual, y_predict, c='crimson')
  plt.yscale('log')
  plt.xscale('log')

  p1 = max(max(y_predict), max(y_actual))
  p2 = min(min(y_predict), min(y_actual))
  plt.plot([p1, p2], [p1, p2], 'b-')
  plt.xlabel('True Values', fontsize=15)
  plt.ylabel('Predictions', fontsize=15)
  plt.axis('equal')
  
  results_graph = os.path.join('/dbfs', output_folder, 'results.png')
  plt.savefig(results_graph)
  mlflow.log_artifact(results_graph)
  
  joblib.dump(clf, open(model_file_path,'wb'))
  mlflow.log_artifact(model_file_path)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC #### View the Experiment Run in Azure Machine Learning Workspace
# MAGIC 
# MAGIC Run the cell below and then **right-click** on **Link to Azure Machine Learning studio** link below to open the `AML Experiment Run Details` page in a **new browser tab**.

# COMMAND ----------

list(ws.experiments[experiment_name].get_runs())[0]

# COMMAND ----------

# MAGIC %md ## Exercise 2: Reviewing experiment metrics in Azure ML Studio
# MAGIC 
# MAGIC Return to the `lab guide` and follow instructions to review the model performance metrics and training artifacts in the Azure Machine Learning workspace.