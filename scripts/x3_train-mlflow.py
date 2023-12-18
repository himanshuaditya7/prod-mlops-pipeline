#Importing the libraries
##################################### mlflow-integration ########################
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from mlflow.models import infer_signature

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Create a new MLflow Experiment
mlflow.set_experiment("MLflow Quickstart")


# Load data
fulldf = pd.read_csv('data_processed.csv')

# Extract X and y variables and values
x = fulldf.drop(['satisfaction'], axis = 1)
y = fulldf['satisfaction']

z = x.values
min_max_scaler = MinMaxScaler()
z_scaled = min_max_scaler.fit_transform(z)
x = pd.DataFrame(z_scaled)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.11, random_state = 88)

# - - - - APPLY CLASSIFIERS -->
# ::: Apply Dummy Classifier
dummyModel = DummyClassifier(strategy="most_frequent")
dummyModel.fit(X_train, y_train)
predictionsDummy = dummyModel.predict(X_test)

accuracyDummy = metrics.accuracy_score(y_test, predictionsDummy)       # One of the results


# ::: Apply KNN
def chooseKNN():
    maxi = 1 #saving the index of the highest score
    max = 0 #saving the value of the highest score
    for i in range(1,20):
        modelKNN = KNeighborsClassifier(n_neighbors = i, weights='distance')
        modelKNN.fit(X_train, y_train)
        accuracy = modelKNN.score(X_test, y_test)
        if (accuracy > max):
            maxi = i
            max = accuracy
    print(maxi, "  ", max)
# Start an MLflow run
with mlflow.start_run():
  modelKNN = KNeighborsClassifier(n_neighbors = 9, weights='distance')
  modelKNN.fit(X_train, y_train)
  predictionsKNN = modelKNN.predict(X_test)
  accuracyKNN = metrics.accuracy_score(y_test, predictionsKNN)          # One of the results
    # Log the loss metric
  mlflow.log_metric("accuracyknn", accuracyKNN)

  # ::: Apply Logistic Regression
  modelLogReg = LogisticRegression()
  modelLogReg.fit(X_train, y_train)
  predictionsLogReg = modelLogReg.predict(X_test)
  accuracyLogReg = modelLogReg.score(X_test, y_test)                    # One of the results
    # Log the loss metric
  mlflow.log_metric("accuracylog", accuracyLogReg)

  # Infer the model signature
  signatureknn = infer_signature(X_train, modelKNN.predict(X_train))
  signaturelog = infer_signature(X_train, modelLogReg.predict(X_train))
  # Log the model
  modelknn_info = mlflow.sklearn.log_model(
      sk_model=modelKNN,
      artifact_path="knn_model",
      signature=signatureknn,
      input_example=X_train,
      registered_model_name="knn-model",
  )
  modellog_info = mlflow.sklearn.log_model(
      sk_model=modelLogReg,
      artifact_path="logreg_model",
      signature=signaturelog,
      input_example=X_train,
      registered_model_name="log-reg-model",
  )
# - - - - - - - MAKING PREDICTIONS
probsKNN = modelKNN.predict_proba(X_test)[:, 1]
probsLogReg = modelLogReg.predict_proba(X_test)[:, 1]
dummyProbs = dummyModel.predict_proba(X_test)[:, 1]


# Plot ROC
fprLR, tprLR, thresholdsLR = metrics.roc_curve(y_test, probsLogReg)
fprKNN, tprKNN, thresholdsKNN = metrics.roc_curve(y_test, probsKNN)
fprDummy, tprDummy, thresholdsDummy = metrics.roc_curve(y_test, dummyProbs)
# fig = plt.figure()
# axes = fig.add_axes([0,0,1,1])
# axes.plot(fprLR, tprLR, label = "LogReg")
# axes.plot(fprKNN, tprKNN, label = "KNN")
# axes.plot(fprDummy, tprDummy, label = "Dummy")
# axes.set_xlabel("False positive rate")
# axes.set_ylabel("True positive rate")
# axes.set_title("ROC Curve for KNN, Logistic regression, Dummy")
# axes.grid(which = 'major', c='#cccccc', linestyle='--', alpha=0.5)
# axes.legend(shadow=True)
# plt.savefig('ROC.png', dpi=120)


# Calculate AUC values for the classifiers
auc_dummy               = metrics.auc(fprDummy, tprDummy)
auc_logistic_regression = metrics.auc(fprLR, tprLR)
auc_knn                 = metrics.auc(fprKNN, tprKNN)


# - - - - - - - GENERATE METRICS FILE
with open("metrics.json", 'w') as outfile:
        json.dump(
        	{ "accuracy_dummy"                 : accuracyDummy,
        	  "accuracy_KNN"                   : accuracyKNN,
        	  "accuracy_logistic-regression"   : accuracyLogReg,
        	  "AUC_dummy"                      : auc_dummy,
        	  "AUC_logistic-regression"        : auc_logistic_regression,
        	  "AUC_KNN"                        : auc_knn}, 
        	  outfile
        	)
