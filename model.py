import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv('dataset.csv')
print(df.columns)

df = df.rename(columns={'attendace':'attendance','mathe':'maths','sci':'science','nglish':'english'})

X = df[ ['study_hours', 'attendance', 'parent_edu', 'study_source'] ]
y = df[ ['maths', 'science', 'english'] ]

le = LabelEncoder()
X['parent_edu'] = le.fit_transform(X['parent_edu'])
X['study_source'] = le.fit_transform(X['study_source'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

model = MultiOutputRegressor(RandomForestRegressor())
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print( 'MAE: ',mean_absolute_error(y_test, y_pred))
print('r2_score: ',r2_score(y_test, y_pred))

print('Actal values')
print(y_test.head(5))
print('Predicted Values')
print(y_pred[:5])

model2 = MultiOutputRegressor(LinearRegression())
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)

print('MAE: ', mean_absolute_error(y_test, y_pred2))
print('r2 score: ', r2_score(y_test, y_pred2))

print('Actual Values')
print(y_test.head(5))
print('Pridected Values')
print(y_pred2[:5])

print("COMPARISION")
print('Model1: RANDOM FOREST')
print( 'MAE: ',mean_absolute_error(y_test, y_pred))
print('r2_score: ',r2_score(y_test, y_pred))
print('model2: LINEAR REGRESSION')
print('MAE: ', mean_absolute_error(y_test, y_pred2))
print('r2 score: ', r2_score(y_test, y_pred2))