'''
@author: Bilal-Issa
'''
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

#read data as a DataFrame
df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')

#visualize the data
print(df.head())

#change data type to True and False
df['male'] = df['Sex'] == 'male'
df = df.drop(columns=['Sex'])
print(df.head())

#make sure there is no missing data
df.isnull().sum()

#set traning data
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
#set data ground truth (labels)
y = df['Survived'].values


#Linear Regression model with Kfold technique in SKlearn
kf = KFold(n_splits=5, shuffle=True)
scores = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = LogisticRegression()
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))

print(scores)
print(np.mean(scores))
final_model = LogisticRegression()
final_model.fit(X, y)






#Decision tree classifier in SKlearn
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=0)
param_grid = {
    'max_depth': [5, 15, 25],
    'min_samples_leaf': [1, 3],
    'max_leaf_nodes': [10, 20, 35, 50]}
dt = DecisionTreeClassifier()
dt_gs = GridSearchCV(dt, param_grid, scoring='f1', cv=5)
dt_gs.fit(X_train, y_train)
print("best params:", dt_gs.best_params_)
print("best score:", dt_gs.best_score_)



#Random Forest classifier in SKlearn
param_grid = {
    'n_estimators': [int(x) for x in np.linspace(5,50,5)],
    'max_depth': [5, 15, 25],
    'min_samples_split':[1,2,3,5],
    'min_samples_leaf': [1, 3, 5],
    'max_leaf_nodes': [10, 20, 35, 50],}
    #'bootstrap': [True, False]}
rf = RandomForestClassifier()
rf_gs = GridSearchCV(rf, param_grid, scoring='f1', cv=5)
rf_gs.fit(X_train, y_train)
print("best params:", rf_gs.best_params_)
print(f'Train accuracy - : {rf_gs.score(X_train,y_train):.3f}')
print(f'Test accuracy - : {rf_gs.score(X_test,y_test):.3f}')
print("best score:", rf_gs.best_score_)



















