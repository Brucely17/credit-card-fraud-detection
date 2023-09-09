# import the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

data=pd.read_csv('credit_card_data.csv')

# data.head()
# print(data,data.head())

data=data.sample(frac=0.1,random_state=48)
print('Shape:',data.shape)
# to get mean ,and other values
print(data.describe())

features =data.iloc[:,0:28].columns

print(features)

plt.figure(figsize=(12,28*4))
gs=gridspec.GridSpec(28,1)

for i,c in enumerate(data[features]):
    ax=plt.subplot(gs[i])
    sns.distplot(data[c][data.Class==1],bins=50)
    sns.distplot(data[c][data.Class==0],bins=50)
    ax.set_xlabel('')
    ax.set_title('histogram of feature:'+str(c))
# plt.show()

Fraud =data[data['Class']==1]
Customer=data[data['Class']==0]

outlier_fraction=len(Fraud)/float(len(Customer))

print('Fraud Cases: {}'.format(len(data[data['Class'] == 1])))
print('Valid Transactions: {}'.format(len(data[data['Class'] == 0])))

print(Fraud.Amount.describe())

# correlation matrix
corrmat=data.corr()
fig=plt.figure(figsize=(12,9))

sns.heatmap(corrmat,vmax=0.8,square =True)
# plt.show()

# dividing from dataset
X=data.drop(['Class'],axis=1)
Y=data['Class']

print(X.shape)
print(Y.shape)

X_data=X.values
Y_data=Y.values

print('X,Y:',X_data,Y_data)