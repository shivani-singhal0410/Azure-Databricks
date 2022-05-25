# Databricks notebook source
# MAGIC %md
# MAGIC # Deploying Models in Azure Machine Learning
# MAGIC 
# MAGIC In this lab, you will learn to deploy models in Azure Machine Learning. This lab will cover following exercises:
# MAGIC 
# MAGIC - Exercise 1: Register a databricks-trained model in AML
# MAGIC - Exercise 2: Deploy a service that uses the model
# MAGIC - Exercise 3: Consume the deployed service
# MAGIC 
# MAGIC To install the required libraries please follow the instructions in the lab guide.
# MAGIC 
# MAGIC **Required Libraries**: 
# MAGIC * `azureml-sdk[databricks]` via PyPI
# MAGIC * `sklearn-pandas==2.1.0` via PyPI
# MAGIC * `azureml-mlflow` via PyPI

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
# MAGIC ### Connect to the AML workspace

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC In the following cell, be sure to set the values for `subscription_id`, `resource_group`, and `workspace_name` as directed by the comments. Please note, you can copy the subscription ID and resource group name from the **Overview** page on the blade for the Azure ML workspace in the Azure portal.

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
# MAGIC ### Load the training data
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
# MAGIC ### Use MLflow with Azure Machine Learning for Model Training
# MAGIC 
# MAGIC In the subsequent cells you will learn to do the following:
# MAGIC - Set up MLflow tracking URI so as to use Azure ML
# MAGIC - Create MLflow experiment – this will create a corresponding experiment in Azure ML Workspace
# MAGIC - Train a model on Azure Databricks cluster while logging metrics and artifacts using MLflow
# MAGIC - Save the trained model to Databricks File System (DBFS)

# COMMAND ----------

import mlflow
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
experiment_name = 'MLflow-AML-Exercise'
mlflow.set_experiment(experiment_name)

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
# MAGIC Run the cell below to list the experiment run in Azure Machine Learning Workspace that you just completed.

# COMMAND ----------

aml_run = list(ws.experiments[experiment_name].get_runs())[0]
aml_run

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Exercise 1: Register a databricks-trained model in AML
# MAGIC 
# MAGIC Azure Machine Learning provides a Model Registry that acts like a version controlled repository for each of your trained models. To version a model, you use the SDK as follows. Run the following cell to register the model with Azure Machine Learning.

# COMMAND ----------

model_name = 'nyc-taxi-fare'
model_description = 'Model to predict taxi fares in NYC.'
model_tags = {"Type": "GradientBoostingRegressor", 
              "Run ID": aml_run.id, 
              "Metrics": aml_run.get_metrics()}

registered_model = Model.register(model_path=model_file_path, #Path to the saved model file
                                  model_name=model_name, 
                                  tags=model_tags, 
                                  description=model_description, 
                                  workspace=ws)

print(registered_model)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Exercise 2: Deploy a service that uses the model

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Create the scoring script

# COMMAND ----------

script_dir = 'scripts'
dbutils.fs.mkdirs(script_dir)
script_dir_path = os.path.join('/dbfs', script_dir)
print("Script directory path:", script_dir_path)

# COMMAND ----------

# MAGIC %%writefile $script_dir_path/score.py
# MAGIC import json
# MAGIC import numpy as np
# MAGIC import pandas as pd
# MAGIC import sklearn
# MAGIC import joblib
# MAGIC from azureml.core.model import Model
# MAGIC 
# MAGIC columns = ['passengerCount', 'tripDistance', 'hour_of_day', 'day_of_week', 
# MAGIC            'month_num', 'normalizeHolidayName', 'isPaidTimeOff', 'snowDepth', 
# MAGIC            'precipTime', 'precipDepth', 'temperature']
# MAGIC 
# MAGIC def init():
# MAGIC     global model
# MAGIC     model_path = Model.get_model_path('nyc-taxi-fare')
# MAGIC     model = joblib.load(model_path)
# MAGIC     print('model loaded')
# MAGIC 
# MAGIC def run(input_json):
# MAGIC     # Get predictions and explanations for each data point
# MAGIC     inputs = json.loads(input_json)
# MAGIC     data_df = pd.DataFrame(np.array(inputs).reshape(-1, len(columns)), columns = columns)
# MAGIC     # Make prediction
# MAGIC     predictions = model.predict(data_df)
# MAGIC     # You can return any data type as long as it is JSON-serializable
# MAGIC     return {'predictions': predictions.tolist()}

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Create the deployment environment

# COMMAND ----------

from azureml.core import Environment
from azureml.core.environment import CondaDependencies

my_env_name="nyc-taxi-env"
myenv = Environment.get(workspace=ws, name='AzureML-Minimal').clone(my_env_name)
conda_dep = CondaDependencies()
conda_dep.add_pip_package("numpy==1.18.1")
conda_dep.add_pip_package("pandas==1.1.5")
conda_dep.add_pip_package("joblib==0.14.1")
conda_dep.add_pip_package("scikit-learn==0.24.1")
conda_dep.add_pip_package("sklearn-pandas==2.1.0")
conda_dep.add_pip_package("azure-ml-api-sdk")
myenv.python.conda_dependencies=conda_dep

print("Review the deployment environment.")
myenv

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Create the inference configuration

# COMMAND ----------

from azureml.core.model import InferenceConfig
inference_config = InferenceConfig(entry_script='score.py', source_directory=script_dir_path, environment=myenv)
print("InferenceConfig created.")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Create the deployment configuration
# MAGIC 
# MAGIC In this exercise we will use the Azure Container Instance (ACI) to deploy the model

# COMMAND ----------

from azureml.core.webservice import AciWebservice, Webservice

description = 'NYC Taxi Fare Predictor Service'

aci_config = AciWebservice.deploy_configuration(
                        cpu_cores=3, 
                        memory_gb=15, 
                        location='eastus', 
                        description=description, 
                        auth_enabled=True, 
                        tags = {'name': 'ACI container', 
                                'model_name': registered_model.name, 
                                'model_version': registered_model.version
                                }
                        )

print("AciWebservice deployment configuration created.")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Deploy the model as a scoring webservice
# MAGIC 
# MAGIC Please note that it can take **10-15 minutes** for the deployment to complete.

# COMMAND ----------

aci_service_name='nyc-taxi-service'

service = Model.deploy(workspace=ws,
                       name=aci_service_name,
                       models=[registered_model],
                       inference_config=inference_config,
                       deployment_config= aci_config, 
                       overwrite=True)

service.wait_for_deployment(show_output=True)
print(service.state)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Exercise 3: Consume the deployed service

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC **Review the webservice endpoint URL and API key**

# COMMAND ----------

api_key, _ = service.get_keys()
print("Deployed ACI test Webservice: {} \nWebservice Uri: {} \nWebservice API Key: {}".
      format(service.name, service.scoring_uri, api_key))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC **Prepare test data**

# COMMAND ----------

#['passengerCount', 'tripDistance', 'hour_of_day', 'day_of_week', 'month_num', 
# 'normalizeHolidayName', 'isPaidTimeOff', 'snowDepth', 'precipTime', 'precipDepth', 'temperature']

data1 = [2, 5, 9, 4, 5, 'Memorial Day', True, 0, 0.0, 0.0, 65]
data2 = [[3, 10, 15, 4, 7, 'None', False, 0, 2.0, 1.0, 80], 
         [2, 5, 9, 4, 5, 'Memorial Day', True, 0, 0.0, 0.0, 65]]

print("Test data prepared.")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Consume the deployed webservice over HTTP

# COMMAND ----------

import requests
import json

headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}
response = requests.post(service.scoring_uri, json.dumps(data1), headers=headers)
print('Predictions for data1')
print(response.text)
print("")
response = requests.post(service.scoring_uri, json.dumps(data2), headers=headers)
print('Predictions for data2')
print(response.text)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Clean-up
# MAGIC 
# MAGIC When you are done with the exercise, delete the deployed webservice by running the cell below.

# COMMAND ----------

service.delete()
print("Deployed webservice deleted.")

# COMMAND ----------

