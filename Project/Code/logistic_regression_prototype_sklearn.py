import numpy as np
import pandas as pa
import math
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Read data, separate features and the labels then split to train and test data
data = pa.read_csv("../Data/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn_Processed.csv")

features = data.iloc[:, 2:21].copy()
features = np.squeeze(np.asarray(features))

label_Column = data.iloc[:, 21].copy()
# Using Sklearn data split function
features_train, features_test, label_train, label__test = train_test_split(features, label_Column, test_size=0.80)

# Feature scaling. Needed since features vary currently a lot
scaler = StandardScaler()
scaler.fit(features_train)

features_train = scaler.transform(features_train) #scale along features axis
features_test = scaler.transform(features_test)

# Training
classifier = LogisticRegression()
classifier.fit(features_train, label_train)

# Prediction. Predict test data
predictions = classifier.predict(features_test)

# Evaluations
print(confusion_matrix(label__test, predictions)) 
# matrix containing elements Cij where Cij is known to be in group i but predicted to be in group j
# thus, in binary classification true negatives C00, false negatives C10, true positives C11, false positives C01

print(classification_report(label__test, predictions)) 
# y_true : Ground truth target values.
# y_pred: Estimated targets as returned by a classifier
# labels: Optional list of label indices to include in the report
# target_names: Optional display names matching the labels
# sample weight: Sample weights
# digits: Number of digits for formatting output floating point values
# output_dict: If True, return output as dict
# report: Text summary of the precision, F1, for each class

print("Accuracy:" + str(accuracy_score(label__test, predictions, normalize=True, sample_weight=None) * 100) + "%")

y_pred_proba = classifier.predict_proba(features_test)[:,1]
fpr, tpr, _ = metrics.roc_curve(label__test,  y_pred_proba) # receiver operating characteristics for binary classification
plt.plot(tpr, label=", True Positives rate")
plt.plot(fpr,label=", False Positives rate")
plt.legend(loc=4)
plt.show()