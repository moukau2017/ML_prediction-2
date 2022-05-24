# -*- coding: utf-8 -*-
"""
Created on Tue May 24 16:43:58 2022

@author: mmoit
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df1 = pd.read_excel('merged-files/7t9j/7t9j_ligand_sorted.xlsx')
df2 = pd.read_excel('merged-files/7t9j/ADMET-7t9j-2-csv.xlsx')

df1.rename(columns={'Ligand' : 'List'}, inplace=True)
df1.columns
df = pd.merge(df1,df2, on='List')
df.columns

df['S+CL_Mech'].replace(['Renal', 'HepUptake'], ['1', '2'], inplace=True)

df = pd.get_dummies(data=df, columns=['BBB_Filter', 'S+MDCK-LE', 'S+CL_Metab', 'S+CL_Renal', 'S+CL_Uptake', 'S+CL_Mech', 'ECCS_Class', 'BSEP_Inh', 
                                               'Chrom_Aberr', 'PLipidosis', 'Repro_Tox', 'Ser_AlkPhos', 'Ser_GGT', 'Ser_LDH']) 

df.head()
df.to_csv('7t9j-merged.csv')
#df.dtypes
df.info
target_features = 'Binding Energy Ligand'

y = df[target_features]
y.head
x = df.drop(target_features, axis=1)
#x = df3.drop(List, axis=1)
x.head

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=7)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(x_train)
X_train_scaled = scaler.transform(x_train)
X_test_scaled = scaler.transform(x_test)

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#Random forest.
#Increase number of tress and see the effect
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators = 50, random_state=15)
model.fit(x_train, y_train)
y_pred_RF = model.predict(x_test)
from sklearn.metrics import mean_squared_error, mean_absolute_error
mse_RF = mean_squared_error(y_test, y_pred_RF)
mae_RF = mean_absolute_error(y_test, y_pred_RF)
print('Mean squared error using Random Forest: ', mse_RF)
print('Mean absolute error Using Random Forest: ', mae_RF)

from sklearn.metrics import r2_score
score = round(r2_score(y_test, y_pred_RF)*100,2)
print('test score', score)
r2 = r2_score(y_test, y_pred_RF)
print('r2 score for perfect model is', r2)
Df1 = pd.DataFrame({'Actual':y_test, 'Predicted':y_pred_RF, 'Variance':y_test-y_pred_RF})
fig,ax = plt.subplots(figsize=(20,10))
x_ax = range(len(x_test))
plt.plot(x_ax, y_test, lw=4, color='green', label='original')
plt.plot(x_ax, y_pred_RF, lw=3, color='red', label='predicted')
plt.legend(fontsize=32)
plt.show()
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#LINEAR REGRESSION
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(x_train,y_train)
train_score = round(regression.score(x_train,y_train)*100,2)
print('training score of linear regression:', train_score)


y_pred = regression.predict(x_test)
from sklearn.metrics import r2_score
score = round(r2_score(y_test, y_pred)*100,2)
print('test score', score)
r2 = r2_score(y_test, y_pred)
print('r2 score for model is', r2)

from sklearn import metrics
print('Mean absolute error on test data of linear regression:', metrics.mean_absolute_error(y_pred,y_test))
print('Mean squared error on test data of linear regression:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean squared error on test data of linear regression:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

Df1 = pd.DataFrame({'Actual':y_test, 'Predicted':y_pred, 'Variance':y_test-y_pred})
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x_train,y_train)

y_pred1 = regressor.predict(x_test)
from sklearn.metrics import r2_score
score1 = round(r2_score(y_test, y_pred1)*100,2)
#print('test score', score)
r2 = r2_score(y_test, y_pred1)
print('r2 score for model is', r2)
from sklearn import metrics
print('Mean absolute error on test data of linear regression:', metrics.mean_absolute_error(y_pred1,y_test))
print('Mean squared error on test data of linear regression:', metrics.mean_squared_error(y_test, y_pred1))
print('Root Mean squared error on test data of linear regression:', np.sqrt(metrics.mean_squared_error(y_test, y_pred1)))

Df1 = pd.DataFrame({'Actual':y_test, 'Predicted':y_pred, 'Variance':y_test-y_pred1})
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#artificial neural network
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(x_train)
X_train_scaled = scaler.transform(x_train)
X_test_scaled = scaler.transform(x_test)



model = Sequential()
model.add(Dense(128, input_dim=73, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
model.summary()

history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs =100)

from matplotlib import pyplot as plt
#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['mean_absolute_error']
val_acc = history.history['val_mean_absolute_error']
plt.plot(epochs, acc, 'y', label='Training MAE')
plt.plot(epochs, val_acc, 'r', label='Validation MAE')
plt.title('Training and validation MAE')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

predictions = model.predict(x_test)
predictions = predictions.flatten()
print("Predicted values are: ", predictions)
print("Real values are: ", y_test[:5])

r2 = r2_score(y_test, y_test-predictions)
print('r2 score for model is', r2)
from sklearn import metrics
print('Mean absolute error on test data of linear regression:', metrics.mean_absolute_error(y_pred1,y_test))
print('Mean squared error on test data of linear regression:', metrics.mean_squared_error(y_test, y_pred1))
print('Root Mean squared error on test data of linear regression:', np.sqrt(metrics.mean_squared_error(y_test, y_pred1)))

Df1 = pd.DataFrame({'Actual':y_test, 'Predicted':y_pred, 'Variance':y_test-y_pred1})
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# load data and arrange into Pandas dataframe
df = read_csv("merged-files/7t9j/7t9j_classification.csv")
print(df.head())

#Split into features and target
X = df.drop('Binding Energy', axis = 1)
y = df['Binding Energy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 40)


####################################### Cross val Q2 ###############################################
#kfold cross validation Q2
#print("Final rmse value is =",np.sqrt(np.mean((y_test, y_pred_lr)**2)))
def Q2_score(y_test,y_pred):
    numerator = ((y_test - y_pred) ** 2).sum()
    denominator = ((y_test - np.average(y_test)) ** 2).sum()
    output_scores = 1 - (numerator / denominator)
    return output_scores 

from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition
df_train, df_test = model_selection.train_test_split(df, test_size=0.2)
X = df.drop('Binding Energy', axis = 1).values
y = df['Binding Energy'].values

## call models one by one decision tree, random forest and XGBoost
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor()
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators = 30, random_state=30)

import numpy as np
import xgboost as xgb
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)

## K fold validation and plot scatter plot for 10 fold
scores = []
cv = model_selection.KFold(n_splits=10, shuffle=True)
fig = plt.figure()
i = 1
for train, test in cv.split(X, y):
    y_pred = model.fit(X[train], y[train]).predict(X[test])
    true = y[test]
    #score = metrics.r2_score(true, y_pred)
    score = Q2_score(true,y_pred)    
    scores.append(score)
    plt.scatter(y_pred, true, lw=2, alpha=0.3, label='Fold %d (Q2 = %0.2f)' % (i,score))
    i = i+1


plt.plot([min(y),max(y)], [min(y_train),max(y)], linestyle='--', lw=2, color='black')
plt.xlabel('Predicted by Random forest')
plt.ylabel('True')
plt.title('K-Fold Validation for 6moj')
plt.legend()
plt.show()
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
import pandas as pd
df = pd.read_csv('6moj_ligand.csv')

#filt = (df['Binding Energy'] <= '-7')
filt = (df['Binding Energy'] <= '-7')
df.head()
filt2 = df[(df['Binding Energy'] < '-7')] 
filt3 =  df[(df['Binding Energy'] > '-3')]   
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#joy plot
import joypy
import pandas as pd
import matplotlib.pyplot as plt

#iris = pd.read_csv("MODIFIED_LIGAND_ENERGY/sorted-peptides1.csv")
iris = pd.read_csv("MODIFIED_LIGAND_ENERGY/sorted-peptides13.csv")
#iris = pd.read_csv("MODIFIED_LIGAND_ENERGY/sorted-peptides-ALL.csv")
fig, axes = joypy.joyplot(iris)

from matplotlib import cm
fig, axes = joypy.joyplot(iris, figsize =(5,3), colormap=cm.autumn, fade=True)
fig, axes = joypy.joyplot(iris, figsize =(5,3), colormap=cm.rainbow, fade=True)

fig, axes = joypy.joyplot(iris, figsize =(5,3), x_range=[-10,15], colormap= cm.viridis)
fig, axes = joypy.joyplot(iris, figsize =(5,3), x_range=[-10,15], colormap= cm.magma)
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#...  DOT+BOX PLOT  ....

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("MODIFIED_LIGAND_ENERGY/sorted-peptides1.csv")
df.head()
sns.set(style='whitegrid')
fig, ax = plt.subplots(figsize=(4,6))
g = sns.boxplot(data=df, width=0.8)#notch=True, ) # df[['p1_satisfaction','p2_satisfaction','p3_satisfaction']]
#this is for box and dot plot
sns.stripplot(data=df, color="grey")
# Titles and labels
plt.title("Sorted peptides", fontsize=16)
plt.ylabel("Ratio", fontsize=14)
plt.tight_layout()
plt.savefig('all_sat_boxplots.png', dpi=500)
plt.show()
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
