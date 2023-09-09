import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec


from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest,RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef
from datadetection import *

X_train ,X_test,Y_train,Y_test=train_test_split(X_data,Y_data,test_size=0.2,random_state=42)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

# Your code for data loading and splitting here

# Isolation Forest
ifc = IsolationForest(max_samples=len(X_train), contamination=outlier_fraction, random_state=1)
ifc.fit(X_train)
y_pred = ifc.predict(X_test)
y_pred[y_pred == 1] = 0
y_pred[y_pred == -1] = 1

# Calculate confusion matrix
conf_matrix = confusion_matrix(Y_test, y_pred)

# Random Forest Classifier
rfc = RandomForestClassifier()
rfc.fit(X_train, Y_train)
y_pred_rf = rfc.predict(X_test)

# Calculate confusion matrix for Random Forest
conf_matrix_rf = confusion_matrix(Y_test, y_pred_rf)

# Functions to calculate precision matrix and recall matrix
def precision_matrix(conf_matrix):
    return conf_matrix / conf_matrix.sum(axis=0, keepdims=True)

def recall_matrix(conf_matrix):
    return conf_matrix / conf_matrix.sum(axis=1, keepdims=True)

# Calculate precision matrices
precision_matrix_ifc = precision_matrix(conf_matrix)
precision_matrix_rfc = precision_matrix(conf_matrix_rf)

# Calculate recall matrices
recall_matrix_ifc = recall_matrix(conf_matrix)
recall_matrix_rfc = recall_matrix(conf_matrix_rf)

# Calculate accuracy, F1-Score, and MCC for both models
def evaluate_model(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    return acc, prec, rec, f1, mcc

# Evaluate Isolation Forest
acc_ifc, prec_ifc, rec_ifc, f1_ifc, mcc_ifc = evaluate_model(Y_test, y_pred)

# Evaluate Random Forest Classifier
acc_rfc, prec_rfc, rec_rfc, f1_rfc, mcc_rfc = evaluate_model(Y_test, y_pred_rf)

print('Isolation Forest:')
print(f'Accuracy: {acc_ifc}')
print(f'Precision: {prec_ifc}')
print(f'Recall: {rec_ifc}')
print(f'F1-Score: {f1_ifc}')
print(f'Matthews Correlation Coefficient: {mcc_ifc}')

print('\nRandom Forest Classifier:')
print(f'Accuracy: {acc_rfc}')
print(f'Precision: {prec_rfc}')
print(f'Recall: {rec_rfc}')
print(f'F1-Score: {f1_rfc}')
print(f'Matthews Correlation Coefficient: {mcc_rfc}')
